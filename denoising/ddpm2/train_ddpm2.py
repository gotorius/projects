# train_ddpm.py
import os
import math
import copy
import random
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import pandas as pd

# === 設定 ===
DATA_DIR = '/mnt/data1/gotou/projects/Medical/kaggledata'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
OUT_DIR = os.path.join(DATA_DIR, 'ddpm_out')
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16         # GPUメモリに応じて調整（224だと8でも重い場合あり）
EPOCHS = 100
SAVE_EVERY = 10           # epochごとに保存/サンプル
LEARNING_RATE = 2e-4
NUM_WORKERS = 4
# GPU IDを環境変数から取得（デフォルトは0番）
GPU_ID = int(os.environ.get('GPU_ID', '0'))
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
PRINT_EVERY = 50

# DDPMハイパーパラメータ
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

# === データセット（既存の PCamDataset を使いつつ、画像を [-1,1] に正規化） ===
class PCamDatasetForDDPM(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels = labels_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_id = self.labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # returns tensor in [0,1]
        # scale to [-1, 1]
        image = image * 2.0 - 1.0
        return image

# transforms: 224x224, keep variety for diffusion training
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# load labels and split (use only image list; labels not needed for unconditional DDPM)
labels_df = pd.read_csv(LABELS_CSV)
# optional: shuffle/split if you want validation. Here we use all images for training.
train_df = labels_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

train_dataset = PCamDatasetForDDPM(TRAIN_IMG_DIR, train_df, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

# === DDPM utilities ===
def make_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps)

# 前計算
betas = make_beta_schedule(TIMESTEPS).to(DEVICE)            # (T,)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)              # \bar{\alpha}_t
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# === 時刻埋め込み ===
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,)
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # (batch, dim)

# === 小さめの ResBlock with time embedding を持つ ===
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.SiLU()
            )
        else:
            self.time_mlp = None
        self.act = nn.SiLU()

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            # broadcast to spatial dims
            time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)

# === 簡易 U-Net ===
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # down
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)  # /2
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)  # /4
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)  # /8
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)  # /16

        # bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)

        # up
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch*16, base_ch*8, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        # t: (batch,) long or float
        t_emb = self.time_mlp(t)  # (batch, time_emb_dim)

        # encode
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb)
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        d4 = self.down4(e4)

        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)

        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        u4 = self.dec4(u4, t_emb)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, e3], dim=1)
        u3 = self.dec3(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        out = self.out_conv(u1)
        return out  # predicted noise (same shape as x)

# === モデル/最適化 ===
model = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# EMA（オプション：推論時に安定化）
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().cpu().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new = param.detach().cpu().clone()
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * new

    def store(self, model):
        self.tmp = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.tmp[name] = param.detach().cpu().clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.tmp[name].to(param.device))
        self.tmp = None

ema = EMA(model, decay=0.9995)

# === サンプリング・前進関数 ===
def q_sample(x_start, t, noise=None):
    # x_start: (B, C, H, W) in [-1,1]
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

# p_sample step (one reverse step)
def p_sample(model, x_t, t):
    # t: scalar timestep for this call (int), but we provide batch t as tensor
    betas_t = betas[t].view(-1,1,1,1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
    sqrt_recip_alphas_t = (1.0 / torch.sqrt(alphas[t])).view(-1,1,1,1)

    # predict noise
    eps_pred = model(x_t, t)  # shape (B,C,H,W)
    # compute mean of posterior (see Ho et al., eq)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_pred)
    if t.item() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        posterior_var_t = betas[t].view(-1,1,1,1)
        return model_mean + torch.sqrt(posterior_var_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, device):
    b = shape[0]
    img = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(TIMESTEPS)), desc='sampling loop', total=TIMESTEPS):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t)
    return img

# === 学習ループ ===
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # mixed precision

global_step = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    running_loss = 0.0
    for idx, batch in enumerate(pbar):
        x0 = batch.to(DEVICE)  # [-1,1]
        B = x0.size(0)
        # sample t uniformly for each sample in batch
        t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
        noise = torch.randn_like(x0)
        x_noisy = q_sample(x0, t, noise=noise)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            eps_pred = model(x_noisy, t)
            loss = F.mse_loss(eps_pred, noise)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        global_step += 1

        if global_step % PRINT_EVERY == 0:
            pbar.set_postfix({'loss': running_loss / (idx+1)})

        # EMA 更新
        ema.update(model)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} finished. avg loss: {avg_loss:.6f}")

    # 10 epochごと: 保存 + 生成
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }
        ckpt_path = os.path.join(OUT_DIR, f"ddpm2_epoch{epoch}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # 生成：EMA を使って生成すると安定する
        ema.store(model)
        ema_shadow = model  # EMA stored into params
        # copy shadow params into model for sampling
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(ema.shadow[name].to(param.device))

        model.eval()
        sample_count = 16
        sample_shape = (sample_count, 3, IMAGE_SIZE, IMAGE_SIZE)

        samples = p_sample_loop(model, sample_shape, DEVICE)
        # samples are in [-1,1] -> scale to [0,1]
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0)

        grid = utils.make_grid(samples, nrow=4)
        sample_path = os.path.join(OUT_DIR, f"samples2_epoch{epoch}.png")
        utils.save_image(grid, sample_path)
        print(f"Saved samples to {sample_path}")

        model.train()
        ema.restore(model)

print("Training complete.")
