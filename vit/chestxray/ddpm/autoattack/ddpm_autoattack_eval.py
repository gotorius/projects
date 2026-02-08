"""
ChestX-ray Dataset - AutoAttack + DDPM Purification Defense (ViT Classifier)
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. AutoAttack敵対的画像の分類精度（防御なし）
4. AutoAttack敵対的画像を浄化した後の分類精度（防御あり）

注意: DDPMはグレースケール（1チャンネル）で訓練されているため、
RGB→グレースケール→DDPM浄化→グレースケール→RGB の変換を行う

AutoAttack: 
- APGD-CE (Auto-PGD with cross-entropy loss)
- APGD-DLR (Auto-PGD with Difference of Logits Ratio loss)
- FAB (Fast Adaptive Boundary)
- Square Attack (query-based black-box attack)
"""

"""
# 基本実行（デフォルト設定）
python ddpm_autoattack_eval.py

# パラメータ指定
python ddpm_autoattack_eval.py \
    --epsilon 0.03137 \
    --start_t 80 \
    --T_purify 50 \
    --gpu 0

# AutoAttackのバージョン指定
python ddpm_autoattack_eval.py --version standard  # APGD-CE + APGD-DLR + FAB + Square
python ddpm_autoattack_eval.py --version plus      # APGD-CE + APGD-DLR + FAB + Square + Multi-targeted
python ddpm_autoattack_eval.py --version rand      # ランダムバージョン（軽量）
"""

import os
import sys
import argparse
import random
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm

# AutoAttackのインポート
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not installed. Please install it with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray AutoAttack + DDPM Defense (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon (pixel scale 0-1)')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand', 'custom'],
                        help='AutoAttack version')
    parser.add_argument('--norm', type=str, default='Linf',
                        choices=['Linf', 'L2'],
                        help='Attack norm')
    parser.add_argument('--attacks_to_run', type=str, nargs='+', 
                        default=['apgd-ce', 'apgd-dlr', 'fab', 'square'],
                        help='List of attacks to run (for custom version)')
    parser.add_argument('--n_restarts', type=int, default=1,
                        help='Number of restarts for AutoAttack')
    
    # DDPM浄化設定
    parser.add_argument('--start_t', type=int, default=280,
                        help='Diffusion start timestep')
    parser.add_argument('--T_purify', type=int, default=300,
                        help='Number of purification steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDPM sampling eta (0=DDIM, 1=DDPM)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='/mnt/data1/gotou/projects/resnet/chestxray/ddpm/ddpm_out3/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/ddpm/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
            time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
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
            return SelfAttention2d(ch, attn_heads)
        
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
        """Cosine beta schedule (Improved DDPM)"""
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
        """v-predictionを使った1ステップの逆拡散"""
        t = t_batch[0].item()
        b = x_t.size(0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_alphas_t = self.sqrt_alphas[t].view(-1, 1, 1, 1)
        
        # v-prediction
        v_pred = self.ddpm(x_t, t_batch)
        
        # x0を推定
        x0_pred = sqrt_alpha_cumprod_t * x_t - sqrt_one_minus_alpha_cumprod_t * v_pred
        x0_pred = torch.tanh(x0_pred * 0.8) / 0.8
        
        # posterior mean計算
        posterior_mean_coef1 = (betas_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
        posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev_t) * sqrt_alphas_t) / (1.0 - alpha_cumprod_t + 1e-8)
        model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
        
        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)
        posterior_var_t = posterior_var_t * 0.8
        
        if t == 0:
            return model_mean, x0_pred
        else:
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var_t) * noise, x0_pred
    
    def purify(self, x_pixel_rgb):
        """RGB画像 [0,1] を浄化"""
        b = x_pixel_rgb.size(0)
        device = x_pixel_rgb.device
        
        # RGB → グレースケール
        x_gray = self.rgb_to_gray(x_pixel_rgb)
        
        # [0,1] → [-1,1]
        x_ddpm = self.pixel_to_ddpm(x_gray)
        
        # Forward diffusion
        t0 = torch.full((b,), self.start_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_ddpm)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_ddpm + sqrt_one_minus_alpha_bar * noise
        
        # Reverse diffusion
        x0_pred = None
        for t_ in range(self.start_t, max(self.start_t - self.T_purify, 0), -1):
            t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
            x_t, x0_pred = self.p_sample_v_pred(x_t, t_batch)
        
        x0_hat = x0_pred if x0_pred is not None else x_t
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        # [-1,1] → [0,1]
        x_purified_gray = self.ddpm_to_pixel(x0_hat)
        
        # グレースケール → RGB
        x_purified_rgb = self.gray_to_rgb(x_purified_gray)
        
        return x_purified_rgb
    
    def forward(self, x_pixel_rgb):
        return self.purify(x_pixel_rgb)


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
    """ViT分類器のラッパー
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
    """DDPM浄化 + ViT分類器のラッパー
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, purifier, classifier, mean, std):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → DDPM浄化 → 2クラスロジット"""
        x_purified = self.purifier(x)
        x_norm = (x_purified - self.mean) / self.std
        return self.classifier(x_norm)


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み（ViT分類器で正しく分類されたサンプル）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NORMAL', 'PNEUMONIA'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_models(args, device):
    """ViT分類器とDDPMを読み込み"""
    # ViT分類器（2クラス: NORMAL, PNEUMONIA）
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 2)
    )
    
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
            ddpm.load_state_dict(ckpt['ema_state_dict'])
            print("Loaded DDPM from EMA state dict")
        elif 'model_state_dict' in ckpt:
            ddpm.load_state_dict(ckpt['model_state_dict'])
            print("Loaded DDPM from model state dict")
        else:
            ddpm.load_state_dict(ckpt)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    
    print(f"Loaded ViT classifier from {args.clf_ckpt}")
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return classifier, ddpm


# ========== 予測取得と精度計算（統合） ==========
def get_predictions_and_accuracy(model, x, y, bs=32, device=None, desc="Evaluation"):
    """モデルの予測を取得して精度も計算（重複計算を避けるため統合）
    
    Returns:
        predictions: numpy array of predictions
        accuracy: float accuracy value
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=desc, total=n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            batch_preds = outputs.argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    
    predictions = torch.cat(preds).numpy()
    accuracy = correct / len(x)
    
    return predictions, accuracy


# ========== 後方互換性のためのラッパー関数 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    """モデルの精度を計算（後方互換性用）"""
    _, acc = get_predictions_and_accuracy(model, x, y, bs, device)
    return acc


def get_predictions(model, x, bs=32, device=None):
    """モデルの予測を取得（後方互換性用）"""
    preds, _ = get_predictions_and_accuracy(model, x, torch.zeros(len(x)), bs, device)
    return preds


# ========== 混同行列出力 ==========
def print_confusion_matrix(y_true, y_pred, title, classes=None):
    """混同行列をテキスト出力"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        
        print(f"\n{title}:")
        if classes:
            print(f"  Classes: {classes}")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return {}


# ========== AutoAttack実行 ==========
def run_autoattack(model, x_test, y_test, epsilon, version, norm, device, 
                   attacks_to_run=None, n_restarts=1, batch_size=32, log_dir=None):
    """AutoAttackを実行して敵対的サンプルを生成
    
    Args:
        model: 分類器（入力は[0,1]のRGB画像）
        x_test: 入力画像 [N, 3, H, W] in [0, 1]
        y_test: ラベル [N]
        epsilon: 摂動の最大値（ピクセルスケール 0-1）
        version: AutoAttackのバージョン ('standard', 'plus', 'rand', 'custom')
        norm: 攻撃ノルム ('Linf', 'L2')
        device: デバイス
        attacks_to_run: 実行する攻撃のリスト（customバージョン用）
        n_restarts: リスタート回数
        batch_size: バッチサイズ
        log_dir: ログディレクトリ
    
    Returns:
        x_adv: 敵対的画像 [N, 3, H, W] in [0, 1]
    """
    print(f"\nRunning AutoAttack with epsilon={epsilon:.4f}, version={version}, norm={norm}...")
    
    # GPU情報を表示
    if torch.cuda.is_available():
        print(f"AutoAttack will use GPU: {torch.cuda.get_device_name(device.index)}")
        torch.cuda.reset_peak_memory_stats(device)
    
    # AutoAttackの初期化
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, verbose=False, device=device)
    
    # カスタムバージョンの場合、実行する攻撃を指定
    if version == 'custom' and attacks_to_run is not None:
        adversary.attacks_to_run = attacks_to_run
    
    # リスタート回数の設定
    if hasattr(adversary, 'apgd'):
        adversary.apgd.n_restarts = n_restarts
    if hasattr(adversary, 'apgd_targeted'):
        adversary.apgd_targeted.n_restarts = n_restarts
    
    # ログパスの設定
    if log_dir is not None:
        log_path = os.path.join(log_dir, 'autoattack_log.txt')
    else:
        log_path = None
    
    # 攻撃実行
    x_test_gpu = x_test.to(device)
    y_test_gpu = y_test.to(device)
    
    print(f"Input images moved to {device}")
    if torch.cuda.is_available():
        print(f"GPU Memory used: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB / {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB")
    
    # AutoAttackの攻撃実行
    x_adv = adversary.run_standard_evaluation(x_test_gpu, y_test_gpu, bs=batch_size, return_labels=False)
    
    print(f"Generated {len(x_adv)} adversarial samples")
    if torch.cuda.is_available():
        print(f"Peak GPU Memory used: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB")
    
    return x_adv.cpu()


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, 
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # 4枚を並べて保存: Clean, Clean+DDPM, Adv, Adv+DDPM
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Clean+DDPM | Adversarial | Adv+DDPM]")


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
    
    # GPU情報を表示
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(args.gpu)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        torch.cuda.set_device(args.gpu)
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.2f} GB")
    else:
        print(f"CUDA Available: False")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"autoattack_eps{args.epsilon:.4f}_{args.version}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, ddpm = load_models(args, device)
    
    # 浄化器
    purifier = DDPMPurifierGray(
        ddpm, device,
        start_t=args.start_t,
        T_purify=args.T_purify,
        eta=args.eta
    ).to(device)
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = DDPMDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Classes: {classes}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + DDPM Defense Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Version: {args.version}")
    print(f"Norm: {args.norm}")
    if args.version == 'custom':
        print(f"Attacks: {args.attacks_to_run}")
    print(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    pred_clean, clean_acc = get_predictions_and_accuracy(
        classifier_model, x_test, y_test, bs=args.batch_size, device=device, 
        desc="Evaluating clean images (classifier only)"
    )
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with DDPM purification...")
    pred_clean_purified, clean_purified_acc = get_predictions_and_accuracy(
        defense_model, x_test, y_test, bs=args.batch_size, device=device,
        desc="Evaluating clean images (with DDPM)"
    )
    print(f"Clean accuracy (with DDPM): {clean_purified_acc:.4f}")
    results['clean_acc_with_ddpm'] = clean_purified_acc
    
    # ========== 3. AutoAttack & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running AutoAttack and evaluating adversarial images...")
    start_time = time.time()
    
    # AutoAttack実行
    x_adv = run_autoattack(
        classifier_model, x_test, y_test, 
        args.epsilon, args.version, args.norm, device,
        attacks_to_run=args.attacks_to_run if args.version == 'custom' else None,
        n_restarts=args.n_restarts,
        batch_size=args.batch_size,
        log_dir=log_dir
    )
    
    attack_time = time.time() - start_time
    
    pred_adv_no_def, adv_acc_no_defense = get_predictions_and_accuracy(
        classifier_model, x_adv, y_test, bs=args.batch_size, device=device,
        desc="Evaluating adversarial images (no defense)"
    )
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with DDPM purification...")
    pred_adv_defended, adv_defended_acc = get_predictions_and_accuracy(
        defense_model, x_adv, y_test, bs=args.batch_size, device=device,
        desc="Evaluating adversarial images (with DDPM)"
    )
    print(f"Adversarial accuracy (with DDPM): {adv_defended_acc:.4f}")
    results['adv_acc_with_ddpm'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"Note: DDPM is trained on grayscale images")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}")
    print(f"  With DDPM purification:   {results['clean_acc_with_ddpm']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (AutoAttack):")
    print(f"  Without defense:          {results['adv_acc_no_defense']:.4f}")
    print(f"  With DDPM purification:   {results['adv_acc_with_ddpm']:.4f}")
    print(f"  Defense improvement:      {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # 注: 予測は既に上で計算済み（重複計算を避けるため）
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with DDPM)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with DDPM)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== 浄化画像を生成して保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified_clean = []
    x_purified_adv = []
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Purifying clean samples"):
            x_purified_clean.append(purifier(x_test[i:i+1].to(device)).cpu())
        
        for i in tqdm(range(n_samples), desc="Purifying adversarial samples"):
            x_purified_adv.append(purifier(x_adv[i:i+1].to(device)).cpu())
    
    x_purified_clean = torch.cat(x_purified_clean, dim=0)
    x_purified_adv = torch.cat(x_purified_adv, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(), 
        x_adv[:n_samples].cpu(),
        x_purified_clean,
        x_purified_adv,
        y_test[:n_samples].cpu().numpy(), 
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # ==================== 敵対的サンプル保存 ====================
    torch.save({
        'x_clean': x_test.cpu(),
        'x_adv': x_adv.cpu(),
        'y': y_test.cpu(),
        'epsilon': args.epsilon,
        'version': args.version,
        'norm': args.norm,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - AutoAttack + DDPM Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: AutoAttack\n")
        f.write(f"Version: {args.version}\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Norm: {args.norm}\n")
        if args.version == 'custom':
            f.write(f"Attacks: {args.attacks_to_run}\n")
        f.write(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}\n")
        f.write(f"Samples: {len(x_test)}\n")
        f.write(f"Note: DDPM is trained on grayscale images\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['clean_acc_with_ddpm']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (AutoAttack):\n")
        f.write(f"  Without defense:          {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['adv_acc_with_ddpm']:.4f}\n")
        f.write(f"  Defense improvement:      {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (ViT Classifier)", cm_clean), 
                         ("Clean (with DDPM)", cm_clean_purified),
                         ("Adversarial (No Defense)", cm_adv_no_def),
                         ("Adversarial (with DDPM)", cm_adv_defended)]:
            if cm:
                f.write(f"{name}:\n")
                f.write(f"  TN: {cm['tn']:4d}  FP: {cm['fp']:4d}\n")
                f.write(f"  FN: {cm['fn']:4d}  TP: {cm['tp']:4d}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.4f}\n")
                f.write(f"  Precision: {cm['precision']:.4f}, Recall: {cm['recall']:.4f}, F1: {cm['f1']:.4f}\n\n")
    
    # JSON形式でも保存
    results_json = {
        'classifier': 'ViT-B/16',
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'clean_acc_with_ddpm': results['clean_acc_with_ddpm'],
        'adv_acc_no_defense': results['adv_acc_no_defense'],
        'adv_acc_with_ddpm': results['adv_acc_with_ddpm'],
        'defense_improvement': results['defense_improvement'],
        'attack_time': results['attack_time'],
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")
    print(f"✅ Summary: {summary_path}")
    
    return results


if __name__ == '__main__':
    main()
