# generate_samples.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

# === 設定 ===
CHECKPOINT_PATH = '/mnt/data1/gotou/projects/Medical/kaggledata/ddpm_out/ddpm_epoch10.pth'
OUT_DIR = '/mnt/data1/gotou/kaggle/denoising/generated_samples'
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = 224
NUM_SAMPLES = 16  # 生成する画像数
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DDPM utilities ===
def make_beta_schedule(timesteps, beta_start=BETA_START, beta_end=BETA_END):
    return torch.linspace(beta_start, beta_end, timesteps)

betas = make_beta_schedule(TIMESTEPS).to(DEVICE)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# === モデル定義（訓練時と同じ） ===
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
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)

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
        t_emb = self.time_mlp(t)

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
        return out

# === サンプリング関数 ===
@torch.no_grad()
def p_sample(model, x_t, t, t_index):
    """1ステップの逆拡散"""
    betas_t = betas[t].view(-1,1,1,1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
    sqrt_recip_alphas_t = (1.0 / torch.sqrt(alphas[t])).view(-1,1,1,1)

    eps_pred = model(x_t, t)
    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_pred)
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x_t)
        posterior_var_t = posterior_variance[t].view(-1,1,1,1)
        return model_mean + torch.sqrt(posterior_var_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, device):
    """完全な逆拡散サンプリング"""
    b = shape[0]
    img = torch.randn(shape, device=device)
    
    for i in tqdm(reversed(range(TIMESTEPS)), desc='Sampling', total=TIMESTEPS):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i)
    
    return img

# === メイン処理 ===
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    
    # モデル初期化
    model = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(DEVICE)
    
    # チェックポイント読み込み
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"Loaded model from epoch {epoch}")
    
    model.eval()
    
    # サンプル生成
    print(f"Generating {NUM_SAMPLES} samples...")
    sample_shape = (NUM_SAMPLES, 3, IMAGE_SIZE, IMAGE_SIZE)
    samples = p_sample_loop(model, sample_shape, DEVICE)
    
    # [-1, 1] -> [0, 1] に変換
    samples = (samples + 1.0) / 2.0
    samples = samples.clamp(0.0, 1.0)
    
    # グリッド画像として保存
    grid = utils.make_grid(samples, nrow=4)
    grid_path = os.path.join(OUT_DIR, f'samples_epoch{epoch}_grid.png')
    utils.save_image(grid, grid_path)
    print(f"Saved grid image to: {grid_path}")
    
    # 個別画像として保存
    for i in range(NUM_SAMPLES):
        img_tensor = samples[i]
        img_np = img_tensor.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_path = os.path.join(OUT_DIR, f'sample_{i:03d}.png')
        img_pil.save(img_path)
    
    print(f"Saved {NUM_SAMPLES} individual images to: {OUT_DIR}")
    print("Generation complete!")

if __name__ == '__main__':
    main()
