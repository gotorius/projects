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
import matplotlib.pyplot as plt

# === 設定 ===
DATA_DIR = '/mnt/data1/Public/MedImages/DermMel'
# DermMel 用: train_sep 配下にクラス別フォルダ (e.g., Melanoma/ NotMelanoma/) がある前提
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_sep')
OUT_DIR = os.path.join('/mnt/data1/gotou/project/dermmel', 'ddpm_out2')
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 4         # GPUメモリに応じて調整(224だと8でも重い場合あり)
EPOCHS = 200            # より長く訓練
SAVE_EVERY = 10         # epochごとに保存/サンプル
LEARNING_RATE = 1e-4    # より小さな学習率で安定化
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50

# Early Stopping設定
EARLY_STOP_PATIENCE = 5  # 15エポック改善しなければ停止
EARLY_STOP_MIN_DELTA = 0.0001  # この程度の改善は無視

# DDPMハイパーパラメータ
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
# 線形βでは霧・色かぶりになりやすいため、cosineスケジュールを推奨
BETA_SCHEDULE = 'cosine'  # 'linear' or 'cosine'

# 学習目的の改良: v-prediction + P2 loss weighting (詳細構造の学習を強化)
V_PREDICTION = True
P2_GAMMA = 0.5
P2_K = 1.0

# === データセット（未ラベル画像を再帰的に走査し、画像を [-1,1] に正規化） ===
class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, extensions=("jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"), transform=None, recursive=True):
        self.root_dir = root_dir
        self.transform = transform
        self.paths = []
        root_path = Path(root_dir)
        patterns = [f"**/*.{ext}" if recursive else f"*.{ext}" for ext in extensions]
        for pat in patterns:
            for p in root_path.glob(pat):
                if p.is_file():
                    self.paths.append(str(p))
        # 重複除去＆ソート（安定性のため）
        self.paths = sorted(list(set(self.paths)))
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Supported extensions: {extensions}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # returns tensor in [0,1]
        # scale to [-1, 1]
        image = image * 2.0 - 1.0
        return image

# DermMel向け前処理: より強力なデータ拡張で多様性を確保しつつ、医学的意味を保持
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # まず正方形にリサイズ
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# DermMel: クラス別サブフォルダ（Melanoma/NotMelanoma/ など）を含む未ラベル画像を再帰的に収集
train_dataset = UnlabeledImageDataset(TRAIN_IMG_DIR, transform=train_transform, recursive=True)
print(f"Found {len(train_dataset)} training images under {TRAIN_IMG_DIR}")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

# === DDPM utilities ===
def make_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END, schedule=BETA_SCHEDULE):
    if schedule == 'linear':
        return torch.linspace(beta_start, beta_end, timesteps)
    elif schedule == 'cosine':
        # Nichol & Dhariwal 2021 (Improved DDPM) cosineスケジュール
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

# 前計算
betas = make_beta_schedule(TIMESTEPS).to(DEVICE)            # (T,)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)              # \bar{\alpha}_t
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

# 1つ前の累積 α (t-1)。t=0 では 1.0 とみなす
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

# === 自己注意（2D）詳細構造の学習を強化 ===
class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.ln(x_flat))
        return x_flat.transpose(1, 2).view(b, c, h, w)

# === 簡易 U-Net (チャネル数を増やして表現力向上 + Attention追加) ===
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=256, attn_heads=4):  # base_ch: 64→128
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        def attn(ch):
            return SelfAttention2d(ch, num_heads=attn_heads)

        # down
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)  # /2
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)  # /4
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.attn3 = attn(base_ch*4)  # Attention追加で詳細構造学習
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)  # /8
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn4 = attn(base_ch*8)  # Attention追加で詳細構造学習
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)  # /16

        # bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn_bot = attn(base_ch*8)  # Bottleneck Attention

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
        e3 = self.attn3(e3)  # Attention適用
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        e4 = self.attn4(e4)  # Attention適用
        d4 = self.down4(e4)

        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)
        b = self.attn_bot(b)  # Bottleneck Attention

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
model = SimpleUNet(in_ch=3, base_ch=128, time_emb_dim=256, attn_heads=4).to(DEVICE)  # Attention追加
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # weight decay追加

# 学習率スケジューラ追加: cosine annealing で徐々に学習率を下げる
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

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

# p_sample step (one reverse step with v-prediction)
def p_sample(model, x_t, t):
    # t: scalar timestep for this call (int), but we provide batch t as tensor
    # t may be a scalar-like tensor or a (B,) tensor where all elements are equal
    betas_t = betas[t].view(-1, 1, 1, 1)
    alphas_t = alphas[t].view(-1, 1, 1, 1)
    sqrt_alphas_t = sqrt_alphas[t].view(-1, 1, 1, 1)
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_prev_t = alphas_cumprod_prev[t].view(-1, 1, 1, 1)

    if V_PREDICTION:
        # v-predictionモード: より詳細な構造を学習
        v_pred = model(x_t, t)
        # x0 = sqrt(alpha_bar) * x_t - sqrt(1-alpha_bar) * v
        x0_pred = (sqrt_alpha_cumprod_t * x_t) - (sqrt_one_minus_alpha_cumprod_t * v_pred)
        # eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1-alpha_bar)
        eps_pred = (x_t - sqrt_alpha_cumprod_t * x0_pred) / (sqrt_one_minus_alpha_cumprod_t + 1e-8)
    else:
        # 通常のε-prediction
        eps_pred = model(x_t, t)
        x0_pred = (x_t - eps_pred * sqrt_one_minus_alpha_cumprod_t) / (sqrt_alpha_cumprod_t + 1e-8)
    
    # より滑らかなクリッピングで霧を軽減
    x0_pred = torch.tanh(x0_pred * 0.8) / 0.8

    # 後方平均: p(x_{t-1} | x_t, x0)
    posterior_mean_coef1 = (betas_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
    posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev_t) * sqrt_alphas_t) / (1.0 - alpha_cumprod_t + 1e-8)
    model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t

    # posterior variance は事前計算を使用
    posterior_var_t = posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)

    # t==0 のサンプルにはノイズを加えない
    is_t0 = (t == 0).view(-1, 1, 1, 1)
    noise = torch.randn_like(x_t)
    sampled = model_mean + torch.sqrt(posterior_var_t) * noise
    out = torch.where(is_t0, model_mean, sampled)
    # 数値安定のための軽いクリップ
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
scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  # mixed precision (new API)

if __name__ == "__main__":
    global_step = 0
    loss_history = []  # 各エポックのlossを記録
    best_loss = float('inf')  # ベストloss初期化
    best_epoch = 0
    patience_counter = 0  # Early stopping用カウンター
    
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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                if V_PREDICTION:
                    # v-target: v = sqrt(a_bar)*eps - sqrt(1-a_bar)*x0
                    sqrt_ab_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
                    sqrt_omab_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
                    v_target = sqrt_ab_t * noise - sqrt_omab_t * x0
                    v_pred = model(x_noisy, t)
                    # P2 loss weighting: 詳細構造の学習を強化
                    snr = (alphas_cumprod[t] / (1 - alphas_cumprod[t])).view(-1, 1, 1, 1)
                    p2_weight = (P2_K + snr) ** (-P2_GAMMA)
                    loss = (p2_weight * (v_pred - v_target) ** 2).mean()
                else:
                    eps_pred = model(x_noisy, t)
                    loss = F.mse_loss(eps_pred, noise)

            scaler.scale(loss).backward()
            # 収束安定化のための軽いクリップ
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            global_step += 1

            if global_step % PRINT_EVERY == 0:
                pbar.set_postfix({'loss': running_loss / (idx+1)})

            # EMA 更新 (optimizer step 後)
            ema.update(model)

        # エポック終了後に学習率更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # loss履歴に追加
        print(f"Epoch {epoch} finished. avg loss: {avg_loss:.6f}, lr: {current_lr:.6e}")
        
        # ベストモデルの更新チェック + Early Stopping
        if avg_loss < best_loss - EARLY_STOP_MIN_DELTA:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0  # カウンターリセット
            best_model_path = os.path.join(OUT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
            print(f"✨ New best model saved! (loss: {avg_loss:.6f})")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{EARLY_STOP_PATIENCE} epochs")
            
            # Early Stopping判定
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
                print(f"Best model: Epoch {best_epoch} with loss {best_loss:.6f}")
                break  # 訓練終了

        # 早期の挙動確認のため: 最初の数エポック (1,2,5,10) もサンプルを保存
        need_sample = (epoch in {1, 2, 5, 10, 20, 30}) or (epoch % SAVE_EVERY == 0) or (epoch == EPOCHS)
        if need_sample:
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            ckpt_path = os.path.join(OUT_DIR, f"ddpm_epoch{epoch}.pth")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

            # EMA パラメータでサンプリング
            ema.store(model)          # 現在パラメータ保存 + EMA反映
            model.eval()
            sample_count = 16
            sample_shape = (sample_count, 3, IMAGE_SIZE, IMAGE_SIZE)

            samples = p_sample_loop(model, sample_shape, DEVICE)
            # [-1,1] -> [0,1]
            samples = (samples + 1.0) / 2.0
            samples = samples.clamp(0.0, 1.0)

            grid = utils.make_grid(samples, nrow=4)
            sample_path = os.path.join(OUT_DIR, f"samples_epoch{epoch}.png")
            utils.save_image(grid, sample_path)
            print(f"Saved samples to {sample_path}")

            # 元の学習パラメータへ戻す
            ema.restore(model)
            model.train()

    # === 訓練完了後の処理 ===
    print("Training complete.")
    print(f"\n📊 Best model: Epoch {best_epoch} with loss {best_loss:.6f}")
    print(f"Best model saved at: {os.path.join(OUT_DIR, 'best_model.pth')}")
    
    # Loss推移グラフの描画
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.axhline(y=best_loss, color='r', linestyle='--', linewidth=1, label=f'Best Loss ({best_loss:.6f})')
    plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1, alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('DDPM Training Loss History', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # グラフ保存
    loss_plot_path = os.path.join(OUT_DIR, 'loss_history.png')
    plt.savefig(loss_plot_path, dpi=150)
    print(f"\n📈 Loss history plot saved to: {loss_plot_path}")
    
    # Loss履歴をテキストファイルにも保存
    loss_txt_path = os.path.join(OUT_DIR, 'loss_history.txt')
    with open(loss_txt_path, 'w') as f:
        f.write("Epoch,Loss\n")
        for ep, loss in enumerate(loss_history, 1):
            f.write(f"{ep},{loss:.6f}\n")
    print(f"📝 Loss history text saved to: {loss_txt_path}")
    
    plt.close()

