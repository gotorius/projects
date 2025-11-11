# train_ddpm_v2.py - 自己注意機構を追加したバージョン
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

# === 設定 ===
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
OUT_DIR = os.path.join('/mnt/data1/gotou/kaggle/chestxray', 'ddpm_out_v2')
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 8          # 注意機構でVRAM消費増→バッチサイズ削減
EPOCHS = 150            # より長く学習
SAVE_EVERY = 10
LEARNING_RATE = 2e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50

# DDPMハイパーパラメータ
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
BETA_SCHEDULE = 'cosine'

# === データセット ===
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, extensions=("jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"), transform=None, recursive=True):
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        root_path = Path(root_dir)
        all_extensions = list(extensions) + [ext.upper() for ext in extensions]
        patterns = [f"**/*.{ext}" if recursive else f"*.{ext}" for ext in all_extensions]
        for pat in patterns:
            for p in root_path.glob(pat):
                if p.is_file():
                    self.paths.append(str(p))
        self.paths = sorted(list(set(self.paths)))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Supported extensions: {extensions}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        image = image * 2.0 - 1.0
        return image

# 強化された前処理
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomApply([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))
    ], p=0.3),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.1, contrast=0.15)
    ], p=0.3),
    transforms.ToTensor(),
])

train_dataset = UnlabeledImageDataset(TRAIN_IMG_DIR, transform=train_transform, recursive=True)
print(f"Found {len(train_dataset)} training images under {TRAIN_IMG_DIR}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# === DDPM utilities ===
def make_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END, schedule=BETA_SCHEDULE):
    if schedule == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'cosine':
        s = 0.008
        steps = timesteps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f = (t / steps + s) / (1 + s)
        alphas_bar = torch.cos(f * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(min=1e-8, max=0.999)
        return betas.to(torch.float32)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

betas = make_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
alphas_cumprod_prev = torch.cat([torch.ones(1, device=DEVICE), alphas_cumprod[:-1]], dim=0)
sqrt_alphas = torch.sqrt(alphas)
posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# === 時刻埋め込み ===
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# === 自己注意ブロック ===
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(8 if channels >= 8 else 1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        h = (attn @ v).transpose(2, 3).contiguous().view(B, C, H, W)
        
        return x + self.proj(h)

# === ResBlock ===
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
            time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)

# === U-Net with Attention ===
class UNetWithAttention(nn.Module):
    def __init__(self, in_ch=3, base_ch=96, time_emb_dim=256, attn_resolutions=(28, 14)):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)  # 112
        
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)  # 56
        
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.attn3 = SelfAttention(base_ch*4) if 56 in attn_resolutions else nn.Identity()
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)  # 28
        
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn4 = SelfAttention(base_ch*8) if 28 in attn_resolutions else nn.Identity()
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)  # 14

        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn_bot = SelfAttention(base_ch*8) if 14 in attn_resolutions else nn.Identity()
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch*16, base_ch*8, time_emb_dim)
        self.attn_dec4 = SelfAttention(base_ch*8) if 28 in attn_resolutions else nn.Identity()
        
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.attn_dec3 = SelfAttention(base_ch*4) if 56 in attn_resolutions else nn.Identity()
        
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
        t_emb = self.time_mlp(t)

        # Encode
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        
        e3 = self.enc3(d2, t_emb)
        e3 = self.attn3(e3)
        d3 = self.down3(e3)
        
        e4 = self.enc4(d3, t_emb)
        e4 = self.attn4(e4)
        d4 = self.down4(e4)

        # Bottleneck
        b = self.bot1(d4, t_emb)
        b = self.attn_bot(b)
        b = self.bot2(b, t_emb)

        # Decode
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        u4 = self.dec4(u4, t_emb)
        u4 = self.attn_dec4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, e3], dim=1)
        u3 = self.dec3(u3, t_emb)
        u3 = self.attn_dec3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out_conv(u1)

# === モデル/最適化 ===
model = UNetWithAttention(in_ch=3, base_ch=96, time_emb_dim=256, attn_resolutions=(28, 14)).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE * 0.1)

# EMA
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

ema = EMA(model, decay=0.9999)

# === サンプリング ===
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

def p_sample(model, x_t, t):
    betas_t = betas[t].view(-1, 1, 1, 1)
    alphas_t = alphas[t].view(-1, 1, 1, 1)
    sqrt_alphas_t = sqrt_alphas[t].view(-1, 1, 1, 1)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_prev_t = alphas_cumprod_prev[t].view(-1, 1, 1, 1)

    eps_pred = model(x_t, t)
    x0_pred = (x_t - eps_pred * sqrt_one_minus_alpha_cumprod_t) / (sqrt_alpha_cumprod_t + 1e-8)
    x0_pred = x0_pred.clamp(-1.0, 1.0)

    posterior_mean_coef1 = (betas_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
    posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev_t) * sqrt_alphas_t) / (1.0 - alpha_cumprod_t + 1e-8)
    model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t

    posterior_var_t = posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)

    is_t0 = (t == 0).view(-1, 1, 1, 1)
    noise = torch.randn_like(x_t)
    sampled = model_mean + torch.sqrt(posterior_var_t) * noise
    out = torch.where(is_t0, model_mean, sampled)
    return out.clamp(-2.0, 2.0)

@torch.no_grad()
def p_sample_loop(model, shape, device):
    b = shape[0]
    img = torch.randn(shape, device=device)
    for i in tqdm(reversed(range(TIMESTEPS)), desc='sampling loop', total=TIMESTEPS):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t)
    return img

# === 学習ループ ===
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

if __name__ == "__main__":
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        running_loss = 0.0
        for idx, batch in enumerate(pbar):
            x0 = batch.to(DEVICE)
            B = x0.size(0)
            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE).long()
            noise = torch.randn_like(x0)
            x_noisy = q_sample(x0, t, noise=noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                eps_pred = model(x_noisy, t)
                loss = F.mse_loss(eps_pred, noise)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix({'loss': running_loss / (idx+1)})

            ema.update(model)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} finished. avg loss: {avg_loss:.6f}")
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")

        need_sample = (epoch in {1, 2, 5}) or (epoch % SAVE_EVERY == 0) or (epoch == EPOCHS)
        if need_sample:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            ckpt_path = os.path.join(OUT_DIR, f"ddpm_epoch{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            ema.store(model)
            model.eval()
            sample_count = 16
            sample_shape = (sample_count, 3, IMAGE_SIZE, IMAGE_SIZE)

            samples = p_sample_loop(model, sample_shape, DEVICE)
            samples = (samples + 1.0) / 2.0
            samples = samples.clamp(0.0, 1.0)

            grid = utils.make_grid(samples, nrow=4)
            sample_path = os.path.join(OUT_DIR, f"samples_epoch{epoch}.png")
            utils.save_image(grid, sample_path)
            print(f"Saved samples to {sample_path}")

            ema.restore(model)
            model.train()

    print("Training complete.")
