"""
ChestX-ray Dataset - AutoAttack + DDPM Purification Defense (Non-adaptive版)
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
Non-adaptive Attack: 分類器のみを攻撃 → DDPM浄化 → 分類

注意: DDPMはグレースケール（1チャンネル）で訓練されているため、
RGB→グレースケール→DDPM浄化→グレースケール→RGB の変換を行う
"""

"""# 基本実行（デフォルト設定: standard版 = 4つの攻撃）
python ddpm_autoattack_eval.py

# パラメータ指定
python ddpm_autoattack_eval.py \
    --lp_norm Linf \
    --adv_eps 0.03137 \
    --start_t 80 \
    --T_purify 50 \
    --num_samples 500 \
    --gpu 0
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import time
import random
import gc

from autoattack import AutoAttack


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray AutoAttack + DDPM Defense (Non-adaptive)')
    
    # 攻撃設定
    parser.add_argument('--attack_version', type=str, default='standard',
                        choices=['standard', 'rand', 'custom'],
                        help='Attack version: standard (4 attacks), rand (2 attacks + EOT), or custom')
    parser.add_argument('--attack_type', type=str, default='apgd-ce,apgd-t,fab-t,square',
                        help='Attack type for custom version (comma-separated)')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Lp norm for attack')
    parser.add_argument('--adv_eps', type=float, default=8/255,
                        help='Adversarial perturbation epsilon')
    parser.add_argument('--eot_iter', type=int, default=20,
                        help='EOT iterations for rand version')
    
    # DDPM浄化設定
    parser.add_argument('--start_t', type=int, default=80,
                        help='Diffusion start timestep')
    parser.add_argument('--T_purify', type=int, default=50,
                        help='Number of purification steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDPM sampling eta (0=DDIM, 1=DDPM)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0,
                        help='Data random seed')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to evaluate (0 for all)')
    
    # パス設定
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='/mnt/data1/gotou/projects/chestxray/ddpm/ddpm_out3/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/ddpm/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== データセット ==========
class ChestXrayDataset(Dataset):
    """ChestX-rayテストデータセット"""
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        for cfold in class_folders:
            cidx = self.class_to_idx[cfold.name]
            for p in cfold.glob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(p), cidx))
        
        # ソートして順序を固定
        self.samples.sort(key=lambda x: x[0])
        print(f"Collected {len(self.samples)} images from {root_dir}")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ========== DDPMモデル定義（グレースケール用: in_ch=1）==========
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
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_ch), nn.SiLU()) if time_emb_dim else None
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """2D Self-Attention layer"""
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
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)  # (B, HW, C)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.ln(x_flat))
        return x_flat.transpose(1, 2).view(b, c, h, w)


class SimpleUNet(nn.Module):
    """グレースケール用UNet with Attention (in_ch=1)"""
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256, attn_heads=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        def attn(ch):
            return SelfAttention2d(ch, num_heads=attn_heads)
        
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.attn3 = attn(base_ch*4)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn4 = attn(base_ch*8)
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn_bot = attn(base_ch*8)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch*16, base_ch*8, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        e3 = self.attn3(e3)
        e4 = self.enc4(self.down3(e3), t_emb)
        e4 = self.attn4(e4)
        
        # Bottleneck
        b = self.bot1(self.down4(e4), t_emb)
        b = self.bot2(b, t_emb)
        b = self.attn_bot(b)
        
        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)


# ========== DDPM浄化クラス（グレースケール用・v-prediction対応）==========
class DDPMPurifierGray(nn.Module):
    """グレースケールDDPM浄化処理（v-prediction対応）
    
    RGB画像 [0,1] → グレースケール → DDPM浄化 → グレースケール → RGB画像 [0,1]
    
    注意: このDDPMはv-predictionで訓練されています
    v = sqrt(ᾱ)*ε - sqrt(1-ᾱ)*x0
    """
    def __init__(self, ddpm_model, device, T_steps=1000, start_t=80, T_purify=50, eta=0.0):
        super().__init__()
        self.ddpm = ddpm_model
        self.device = device
        self.T_steps = T_steps
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
        
        # βスケジュール（cosine）- 訓練コードと同じ
        betas = self._cosine_beta_schedule(T_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # alphas_cumprod_prev: t=0では1.0
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # posterior variance
        posterior_variance = torch.zeros_like(betas)
        posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod_prev[1:]) / (1.0 - alphas_cumprod[1:])
        posterior_variance[0] = 1e-8
        self.register_buffer('posterior_variance', posterior_variance)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine beta schedule (Improved DDPM) - 訓練コードと同じ"""
        steps = timesteps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f = (t / steps + s) / (1 + s)
        alphas_bar = torch.cos(f * torch.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(min=1e-8, max=0.999)
        return betas.to(torch.float32)
    
    def rgb_to_gray(self, x_rgb):
        """RGB [0,1] → グレースケール [0,1]"""
        weights = torch.tensor([0.299, 0.587, 0.114], device=x_rgb.device).view(1, 3, 1, 1)
        return (x_rgb * weights).sum(dim=1, keepdim=True)
    
    def gray_to_rgb(self, x_gray):
        """グレースケール [0,1] → RGB [0,1]"""
        return x_gray.repeat(1, 3, 1, 1)
    
    def pixel_to_ddpm(self, x_pixel):
        """[0,1] → [-1,1]"""
        return x_pixel * 2.0 - 1.0
    
    def ddpm_to_pixel(self, x_ddpm):
        """[-1,1] → [0,1]"""
        return torch.clamp((x_ddpm + 1.0) / 2.0, 0, 1)
    
    def p_sample_v_pred(self, x_t, t_batch):
        """
        v-predictionを使った1ステップの逆拡散
        訓練コードのp_sample関数と同じロジック
        """
        t = t_batch[0].item()  # スカラー値
        b = x_t.size(0)
        
        # 各種係数
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_alphas_t = self.sqrt_alphas[t].view(-1, 1, 1, 1)
        
        # v-prediction: モデルはvを予測
        v_pred = self.ddpm(x_t, t_batch)
        
        # x0を推定: x0 = sqrt(ᾱ)*x_t - sqrt(1-ᾱ)*v
        x0_pred = sqrt_alpha_cumprod_t * x_t - sqrt_one_minus_alpha_cumprod_t * v_pred
        # 訓練コードと同じクリッピング
        x0_pred = torch.tanh(x0_pred * 0.8) / 0.8
        
        # posterior mean計算
        posterior_mean_coef1 = (betas_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
        posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev_t) * sqrt_alphas_t) / (1.0 - alpha_cumprod_t + 1e-8)
        model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
        
        # posterior variance（訓練コードと同じ80%削減）
        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)
        posterior_var_t = posterior_var_t * 0.8
        
        # t==0ではノイズを加えない
        if t == 0:
            return model_mean, x0_pred
        else:
            noise = torch.randn_like(x_t)
            # eta=0の場合はDDIM（決定的）
            sigma = self.eta * torch.sqrt(posterior_var_t)
            sampled = model_mean + sigma * noise
            return sampled.clamp(-2.0, 2.0), x0_pred
    
    def purify(self, x_pixel_rgb):
        """
        RGB画像 [0,1] を浄化
        RGB → グレースケール → DDPM浄化（v-prediction） → グレースケール → RGB
        """
        b = x_pixel_rgb.size(0)
        device = x_pixel_rgb.device
        
        # RGB → グレースケール
        x_gray = self.rgb_to_gray(x_pixel_rgb)  # [B, 1, H, W]
        
        # [0,1] → [-1,1] (DDPM空間)
        x_ddpm = self.pixel_to_ddpm(x_gray)
        
        # Forward diffusion (ノイズ追加)
        t0 = torch.full((b,), self.start_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_ddpm)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_ddpm + sqrt_one_minus_alpha_bar * noise
        
        # Reverse diffusion (v-prediction)
        x0_pred = None
        for t_ in range(self.start_t, max(self.start_t - self.T_purify, 0), -1):
            t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
            x_t, x0_pred = self.p_sample_v_pred(x_t, t_batch)
        
        # 最終的なx0推定を使用
        x0_hat = x0_pred if x0_pred is not None else x_t
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        # [-1,1] → [0,1]
        x_purified_gray = self.ddpm_to_pixel(x0_hat)
        
        # グレースケール → RGB
        x_purified_rgb = self.gray_to_rgb(x_purified_gray)
        
        return x_purified_rgb
    
    def forward(self, x_pixel_rgb):
        return self.purify(x_pixel_rgb)


# ========== AutoAttack用モデルラッパー ==========
class ClassifierWrapper(nn.Module):
    """分類器のみのラッパー（Non-adaptive用）
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 2クラスロジット"""
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


class DDPMDefenseWrapper(nn.Module):
    """DDPM浄化 + 分類器のラッパー
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, purifier, classifier, mean, std):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.counter = 0
        self.print_freq = 50
    
    def reset_counter(self):
        self.counter = 0
    
    def forward(self, x):
        """x: [0,1]の画像 → DDPM浄化 → 2クラスロジット"""
        if self.counter % self.print_freq == 0:
            print(f'  [DDPMDefense] Forward pass #{self.counter}')
        self.counter += 1
        
        # DDPM浄化（RGB→Gray→DDPM→Gray→RGB）
        x_purified = self.purifier(x)
        
        # 正規化して分類
        x_norm = (x_purified - self.mean) / self.std
        return self.classifier(x_norm)


# ========== データ読み込み ==========
def load_data(args):
    """テストデータを読み込み"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dir = os.path.join(args.data_dir, 'test')
    test_dataset = ChestXrayDataset(test_dir, transform)
    
    # サンプル数制限
    if args.num_samples > 0 and args.num_samples < len(test_dataset):
        np.random.seed(args.data_seed)
        indices = np.random.choice(len(test_dataset), args.num_samples, replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    # 全データをテンソルに変換
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    x_list, y_list = [], []
    for images, labels in tqdm(loader, desc="Loading data"):
        x_list.append(images)
        y_list.append(labels)
    
    x_test = torch.cat(x_list, dim=0)
    y_test = torch.cat(y_list, dim=0)
    
    print(f"Loaded {len(x_test)} test samples")
    return x_test, y_test


# ========== モデル読み込み ==========
def load_models(args, device):
    """分類器とDDPMを読み込み"""
    # 分類器（2クラス: NORMAL, PNEUMONIA）
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    
    # DDPM（グレースケール: in_ch=1）
    ddpm = SimpleUNet(in_ch=1, base_ch=64, time_emb_dim=256).to(device)
    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt and isinstance(ckpt['ema_state_dict'], dict):
            ddpm.load_state_dict(ckpt['ema_state_dict'], strict=False)
            print("Loaded DDPM EMA weights")
        elif 'model_state_dict' in ckpt:
            ddpm.load_state_dict(ckpt['model_state_dict'])
            print("Loaded DDPM model weights")
        else:
            ddpm.load_state_dict(ckpt)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    
    print(f"Loaded classifier from {args.clf_ckpt}")
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return classifier, ddpm


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    """モデルの精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * bs
            end = min(start + bs, len(x))
            x_batch = x[start:end].to(device)
            y_batch = y[start:end].to(device)
            
            out = model(x_batch)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
    
    return correct / len(x)


# ========== AutoAttack評価 ==========
def eval_autoattack(args, classifier_model, defense_model, x_test, y_test, device, log_dir):
    """
    Non-adaptive AutoAttack評価
    分類器のみを攻撃 → DDPM防御を適用 → 精度測定
    """
    # 攻撃設定
    attack_version = args.attack_version
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    
    print(f"\n{'='*70}")
    print(f"Attack Configuration")
    print(f"{'='*70}")
    print(f"Version: {attack_version}")
    print(f"Attacks: {attack_list}")
    print(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps:.4f}")
    print(f"{'='*70}")
    
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    results = {}
    
    # ==================== Non-adaptive Attack ====================
    print(f"\n{'='*70}")
    print("NON-ADAPTIVE ATTACK (Classifier only → DDPM Defense)")
    print(f"{'='*70}")
    print("Attack targets classifier only, then DDPM purification is applied.")
    
    # 初期精度（クリーン画像）
    init_acc_clf = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Initial classifier accuracy (clean): {init_acc_clf:.4f}")
    
    # DDPM防御適用時のクリーン精度
    defense_model.reset_counter()
    init_acc_def = get_accuracy(defense_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Initial defense accuracy (clean + DDPM): {init_acc_def:.4f}")
    
    # AutoAttack（分類器のみを攻撃）
    print(f"\nRunning AutoAttack on classifier...")
    if attack_version == 'custom':
        adversary_clf = AutoAttack(
            classifier_model, norm=args.lp_norm, eps=args.adv_eps,
            version='custom', attacks_to_run=attack_list,
            log_path=os.path.join(log_dir, 'log_autoattack.txt'),
            device=device
        )
        adversary_clf.apgd.n_restarts = 1
    else:
        adversary_clf = AutoAttack(
            classifier_model, norm=args.lp_norm, eps=args.adv_eps,
            version=attack_version,
            log_path=os.path.join(log_dir, 'log_autoattack.txt'),
            device=device
        )
    
    if attack_version == 'rand':
        adversary_clf.apgd.eot_iter = args.eot_iter
    
    start_time = time.time()
    x_adv = adversary_clf.run_standard_evaluation(x_test, y_test, bs=args.batch_size)
    attack_time = time.time() - start_time
    
    # 敵対的精度（防御なし）
    robust_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"\nRobust accuracy (no defense): {robust_acc_no_defense:.4f}")
    
    # DDPM防御を適用した精度
    defense_model.reset_counter()
    defended_acc = get_accuracy(defense_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Defended accuracy (DDPM purification): {defended_acc:.4f}")
    print(f"Defense improvement: {defended_acc - robust_acc_no_defense:+.4f}")
    print(f"Attack time: {attack_time:.2f}s")
    
    results['non_adaptive'] = {
        'clean_acc_classifier': init_acc_clf,
        'clean_acc_defense': init_acc_def,
        'robust_acc_no_defense': robust_acc_no_defense,
        'defended_acc': defended_acc,
        'improvement': defended_acc - robust_acc_no_defense,
        'attack_time': attack_time,
        'attacks': attack_list
    }
    
    # 敵対的サンプル保存
    torch.save({'x_adv': x_adv.cpu(), 'y': y_test.cpu()},
               os.path.join(log_dir, f'x_adv_sd{args.seed}.pt'))
    
    return results, x_adv


# ========== 混同行列出力 ==========
def print_confusion_matrix(y_true, y_pred, title):
    """混同行列をテキスト出力"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        
        print(f"\n{title}:")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'precision': precision, 'recall': recall, 'f1': f1}
    return {}


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, defense_model, y_true, classes,
                       save_dir, device, max_samples=10):
    """サンプル画像を保存（クリーン、敵対的、浄化後）"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    purifier = defense_model.purifier
    
    for i in range(n):
        # 浄化画像を取得
        with torch.no_grad():
            x_purified = purifier(x_adv[i:i+1].to(device))
        
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # clean, adv, purified の3枚を並べて保存
        triplet = torch.cat([
            x_clean[i:i+1],
            x_adv[i:i+1],
            x_purified.cpu()
        ], dim=0)
        grid = make_grid(triplet, nrow=3, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Adversarial | Purified]")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing GPU: {args.gpu}")
    print(f"Device: {device}")
    
    # 出力ディレクトリ
    attack_name = args.attack_version if args.attack_version != 'custom' else args.attack_type
    log_dir = os.path.join(
        args.output_dir,
        f'{attack_name}_{args.lp_norm}_eps{int(args.adv_eps*255)}',
        f'start{args.start_t}_purify{args.T_purify}',
        f'seed{args.seed}_data{args.data_seed}'
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, ddpm = load_models(args, device)
    
    # 浄化器（グレースケール用）
    purifier = DDPMPurifierGray(
        ddpm, device,
        start_t=args.start_t,
        T_purify=args.T_purify,
        eta=args.eta
    ).to(device)
    
    # ラッパー作成
    classifier_model = ClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = DDPMDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test = load_data(args)
    
    # クラス名取得
    test_dir = os.path.join(args.data_dir, 'test')
    classes = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
    print(f"Classes: {classes}")
    
    # AutoAttack評価
    results, x_adv = eval_autoattack(
        args, classifier_model, defense_model, x_test, y_test, device, log_dir
    )
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Non-adaptive Attack)")
    print(f"{'='*70}")
    print(f"Attack: {args.attack_version}, Norm: {args.lp_norm}, Eps: {args.adv_eps:.4f}")
    print(f"Attacks used: {results['non_adaptive']['attacks']}")
    print(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"Note: DDPM is trained on grayscale images")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:          {results['non_adaptive']['clean_acc_classifier']:.4f}")
    print(f"  With DDPM purification:   {results['non_adaptive']['clean_acc_defense']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy:")
    print(f"  Without defense:          {results['non_adaptive']['robust_acc_no_defense']:.4f}")
    print(f"  With DDPM purification:   {results['non_adaptive']['defended_acc']:.4f}")
    print(f"  Defense improvement:      {results['non_adaptive']['improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['non_adaptive']['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # 混同行列
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # バッチ処理で予測を取得（OOM回避）
    def get_predictions_batched(model, x, batch_size=32):
        preds = []
        n_batches = (len(x) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(x))
                x_batch = x[start:end].to(device)
                pred = model(x_batch).argmax(dim=1).cpu()
                preds.append(pred)
                del x_batch
                torch.cuda.empty_cache()
        return torch.cat(preds).numpy()
    
    # Clean (classifier only)
    clean_pred = get_predictions_batched(classifier_model, x_test, batch_size=args.batch_size)
    # Adversarial (no defense)
    adv_pred_no_def = get_predictions_batched(classifier_model, x_adv, batch_size=args.batch_size)
    # Adversarial (with DDPM defense)
    defense_model.reset_counter()
    adv_pred_defended = get_predictions_batched(defense_model, x_adv, batch_size=args.batch_size)
    
    y_true = y_test.cpu().numpy()
    print_confusion_matrix(y_true, clean_pred, f"Clean Images (Classifier) - Classes: {classes}")
    print_confusion_matrix(y_true, adv_pred_no_def, "Adversarial Images (No Defense)")
    print_confusion_matrix(y_true, adv_pred_defended, "Adversarial Images (DDPM Defense)")
    
    # サンプル画像保存
    save_sample_images(
        x_test[:10].cpu(), x_adv[:10].cpu(),
        defense_model, y_test[:10].cpu().numpy(), classes,
        os.path.join(log_dir, 'samples'), device
    )
    
    # 結果をファイルに保存
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - AutoAttack + DDPM Defense (Non-adaptive)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: {args.attack_version}\n")
        f.write(f"Attacks used: {results['non_adaptive']['attacks']}\n")
        f.write(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps:.4f}\n")
        f.write(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}\n")
        f.write(f"Note: DDPM trained on grayscale\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        f.write("-"*70 + "\n")
        f.write("Clean Accuracy:\n")
        f.write(f"  Classifier only:          {results['non_adaptive']['clean_acc_classifier']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['non_adaptive']['clean_acc_defense']:.4f}\n\n")
        f.write("Adversarial Accuracy:\n")
        f.write(f"  Without defense:          {results['non_adaptive']['robust_acc_no_defense']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['non_adaptive']['defended_acc']:.4f}\n")
        f.write(f"  Defense improvement:      {results['non_adaptive']['improvement']:+.4f}\n\n")
        f.write("-"*70 + "\n")
        f.write(f"Attack time: {results['non_adaptive']['attack_time']:.2f}s\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Results saved to: {log_dir}")
    print(f"✅ Summary: {summary_path}")
    
    return results


if __name__ == '__main__':
    main()
