"""
ChestX-ray Dataset - DDPM浄化パラメータのグリッドサーチ
ViT分類器 + FGSM攻撃に対する最適なstart_t, T_purifyを探索

パラメータ範囲:
- start_t: 0-150 (10間隔)
- T_purify: 0-150 (10間隔)

テストデータ: 正例25枚 + 負例25枚 = 50枚
"""

import os
import sys
import argparse
import random
import time
import json
import gc
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DDPM Parameter Grid Search')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # グリッドサーチ設定
    parser.add_argument('--start_t_min', type=int, default=150)
    parser.add_argument('--start_t_max', type=int, default=300)
    parser.add_argument('--start_t_step', type=int, default=10)
    parser.add_argument('--t_purify_min', type=int, default=150)
    parser.add_argument('--t_purify_max', type=int, default=300)
    parser.add_argument('--t_purify_step', type=int, default=10)
    
    # サンプル数
    parser.add_argument('--n_samples_per_class', type=int, default=10,
                        help='Number of samples per class')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='/mnt/data1/gotou/projects/resnet/chestxray/ddpm/ddpm_out3/best_model.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/ddpm/search')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== DDPMモデル定義 ==========
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
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        
        # メモリ効率を改善：バッチ処理を避けて単一サンプル処理にする
        # グリッドサーチでは小バッチなので1サンプルずつ処理
        if b == 1:
            attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        else:
            # 複数サンプルの場合は最小バッチサイズで分割
            attn_out_list = []
            for i in range(b):
                attn_out_i, _ = self.mha(x_flat[i:i+1], x_flat[i:i+1], x_flat[i:i+1])
                attn_out_list.append(attn_out_i)
            attn_out = torch.cat(attn_out_list, dim=0)
        
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.ln(x_flat))
        return x_flat.transpose(1, 2).view(b, c, h, w)


class SimpleUNet(nn.Module):
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
        
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn_bot = attn(base_ch*8)
        
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
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        e3 = self.attn3(e3)
        e4 = self.enc4(self.down3(e3), t_emb)
        e4 = self.attn4(e4)
        b = self.bot1(self.down4(e4), t_emb)
        b = self.bot2(b, t_emb)
        b = self.attn_bot(b)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)


# ========== DDPM浄化クラス（パラメータ可変）==========
class DDPMPurifierGray(nn.Module):
    def __init__(self, ddpm_model, device, T_steps=1000):
        super().__init__()
        self.ddpm = ddpm_model
        self.device = device
        self.T_steps = T_steps
        
        betas = self._cosine_beta_schedule(T_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        posterior_variance = torch.zeros_like(betas)
        posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod_prev[1:]) / (1.0 - alphas_cumprod[1:])
        posterior_variance[0] = 1e-8
        self.register_buffer('posterior_variance', posterior_variance)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
        f = (t / steps + s) / (1 + s)
        alphas_bar = torch.cos(f * torch.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(min=1e-8, max=0.999)
        return betas.to(torch.float32)
    
    def rgb_to_gray(self, x_rgb):
        weights = torch.tensor([0.299, 0.587, 0.114], device=x_rgb.device).view(1, 3, 1, 1)
        return (x_rgb * weights).sum(dim=1, keepdim=True)
    
    def gray_to_rgb(self, x_gray):
        return x_gray.repeat(1, 3, 1, 1)
    
    def pixel_to_ddpm(self, x_pixel):
        return x_pixel * 2.0 - 1.0
    
    def ddpm_to_pixel(self, x_ddpm):
        return torch.clamp((x_ddpm + 1.0) / 2.0, 0, 1)
    
    def p_sample_v_pred(self, x_t, t_batch):
        t = t_batch[0].item()
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_alphas_t = self.sqrt_alphas[t].view(-1, 1, 1, 1)
        
        v_pred = self.ddpm(x_t, t_batch)
        x0_pred = sqrt_alpha_cumprod_t * x_t - sqrt_one_minus_alpha_cumprod_t * v_pred
        x0_pred = torch.tanh(x0_pred * 0.8) / 0.8
        
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
    
    def purify(self, x_pixel_rgb, start_t, T_purify):
        """パラメータ指定可能な浄化"""
        if start_t == 0 or T_purify == 0:
            return x_pixel_rgb
        
        b = x_pixel_rgb.size(0)
        device = x_pixel_rgb.device
        
        x_gray = self.rgb_to_gray(x_pixel_rgb)
        x_ddpm = self.pixel_to_ddpm(x_gray)
        
        t0 = torch.full((b,), start_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_ddpm)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_ddpm + sqrt_one_minus_alpha_bar * noise
        
        x0_pred = None
        for t_ in range(start_t, max(start_t - T_purify, 0), -1):
            t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
            x_t, x0_pred = self.p_sample_v_pred(x_t, t_batch)
        
        x0_hat = x0_pred if x0_pred is not None else x_t
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        x_purified_gray = self.ddpm_to_pixel(x0_hat)
        x_purified_rgb = self.gray_to_rgb(x_purified_gray)
        
        return x_purified_rgb


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x.requires_grad = True
    
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    
    x_adv = x + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    return x_adv


# ========== データ読み込み ==========
def load_subset_samples(cached_path, n_per_class=25):
    """各クラスからn枚ずつサンプルを取得"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_all = cached['x_test']
    y_all = cached['y_test']
    classes = cached.get('classes', ['NORMAL', 'PNEUMONIA'])
    
    # 各クラスからサンプリング
    x_subset = []
    y_subset = []
    
    for class_idx in range(2):
        mask = (y_all == class_idx)
        x_class = x_all[mask]
        y_class = y_all[mask]
        
        n_available = len(x_class)
        n_take = min(n_per_class, n_available)
        
        x_subset.append(x_class[:n_take])
        y_subset.append(y_class[:n_take])
        print(f"  {classes[class_idx]}: {n_take} samples")
    
    x_test = torch.cat(x_subset, dim=0)
    y_test = torch.cat(y_subset, dim=0)
    
    print(f"Total: {len(x_test)} samples")
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_models(args, device):
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
    
    ddpm = SimpleUNet(in_ch=1, base_ch=64, time_emb_dim=256).to(device)
    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt and isinstance(ckpt['ema_state_dict'], dict):
            ddpm.load_state_dict(ckpt['ema_state_dict'])
        elif 'model_state_dict' in ckpt:
            ddpm.load_state_dict(ckpt['model_state_dict'])
        else:
            ddpm.load_state_dict(ckpt)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    
    return classifier, ddpm


# ========== 評価関数 ==========
def evaluate_with_params(purifier, classifier_model, x_adv, y_test, start_t, T_purify, device):
    """指定パラメータでの防御精度を計算"""
    batch_size = 1  # メモリ節約のため1サンプルずつ処理
    correct = 0
    total = 0
    
    try:
        with torch.no_grad():
            for i in range(0, len(x_adv), batch_size):
                x_batch = x_adv[i:i+batch_size].to(device)
                y_batch = y_test[i:i+batch_size].to(device)
                
                # 浄化
                x_purified = purifier.purify(x_batch, start_t, T_purify)
                
                # 分類
                outputs = classifier_model(x_purified)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
                # 中間結果を削除してメモリ解放
                del x_batch, y_batch, x_purified, outputs, preds
    finally:
        # メモリをクリア
        torch.cuda.empty_cache()
        gc.collect()
    
    return correct / total


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"grid_search_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, ddpm = load_models(args, device)
    purifier = DDPMPurifierGray(ddpm, device).to(device)
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み（サブセット）
    x_test, y_test, classes = load_subset_samples(args.cached_samples, args.n_samples_per_class)
    
    # FGSM攻撃で敵対的サンプルを生成
    print(f"\nGenerating adversarial samples with epsilon={args.epsilon:.4f}...")
    x_adv = fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device)
    
    # 攻撃前後の精度
    with torch.no_grad():
        outputs_clean = classifier_model(x_test.to(device))
        clean_acc = (outputs_clean.argmax(1) == y_test.to(device)).float().mean().item()
        
        outputs_adv = classifier_model(x_adv.to(device))
        adv_acc = (outputs_adv.argmax(1) == y_test.to(device)).float().mean().item()
    
    print(f"Clean accuracy: {clean_acc:.4f}")
    print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    
    # パラメータグリッド生成
    start_t_values = list(range(args.start_t_min, args.start_t_max + 1, args.start_t_step))
    t_purify_values = list(range(args.t_purify_min, args.t_purify_max + 1, args.t_purify_step))
    
    print(f"\n{'='*70}")
    print("Grid Search Parameters")
    print(f"{'='*70}")
    print(f"start_t: {start_t_values}")
    print(f"T_purify: {t_purify_values}")
    print(f"Total combinations: {len(start_t_values) * len(t_purify_values)}")
    print(f"{'='*70}")
    
    # グリッドサーチ実行
    results = []
    param_combinations = list(product(start_t_values, t_purify_values))
    
    print("\nRunning grid search...")
    for idx, (start_t, T_purify) in enumerate(tqdm(param_combinations, desc="Grid Search")):
        # 定期的なメモリクリア（10イテレーションごと）
        if idx > 0 and idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # T_purify > start_t の場合は無効（浄化ステップが開始点を超える）
        effective_T_purify = min(T_purify, start_t)
        
        try:
            acc = evaluate_with_params(purifier, classifier_model, x_adv, y_test, 
                                       start_t, effective_T_purify, device)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n⚠️ CUDA OOM at ({start_t}, {T_purify}): Skipping...")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
        
        results.append({
            'start_t': start_t,
            'T_purify': T_purify,
            'effective_T_purify': effective_T_purify,
            'defended_acc': acc,
            'improvement': acc - adv_acc
        })
    # 結果をDataFrameに変換
    df = pd.DataFrame(results)
    
    # ベスト結果
    best_idx = df['defended_acc'].idxmax()
    best_result = df.loc[best_idx]
    
    print(f"\n{'='*70}")
    print("BEST RESULT")
    print(f"{'='*70}")
    print(f"start_t: {best_result['start_t']}")
    print(f"T_purify: {best_result['T_purify']}")
    print(f"Effective T_purify: {best_result['effective_T_purify']}")
    print(f"Defended accuracy: {best_result['defended_acc']:.4f}")
    print(f"Improvement: {best_result['improvement']:+.4f}")
    print(f"{'='*70}")
    
    # Top 10結果
    print("\nTop 10 Results:")
    top10 = df.nlargest(10, 'defended_acc')
    print(top10.to_string(index=False))
    
    # ヒートマップ用のピボットテーブル作成
    pivot_acc = df.pivot(index='start_t', columns='T_purify', values='defended_acc')
    pivot_imp = df.pivot(index='start_t', columns='T_purify', values='improvement')
    
    # ヒートマップ保存
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 防御精度のヒートマップ
    sns.heatmap(pivot_acc, annot=True, fmt='.2f', cmap='RdYlGn', 
                ax=axes[0], vmin=0, vmax=1, annot_kws={'size': 8})
    axes[0].set_title(f'Defended Accuracy\n(Clean: {clean_acc:.2f}, Adv: {adv_acc:.2f})', fontsize=12)
    axes[0].set_xlabel('T_purify')
    axes[0].set_ylabel('start_t')
    
    # 改善度のヒートマップ
    sns.heatmap(pivot_imp, annot=True, fmt='+.2f', cmap='RdYlGn', center=0,
                ax=axes[1], annot_kws={'size': 8})
    axes[1].set_title('Improvement over No Defense', fontsize=12)
    axes[1].set_xlabel('T_purify')
    axes[1].set_ylabel('start_t')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved heatmap to: {os.path.join(log_dir, 'heatmap.png')}")
    
    # 結果をCSV保存
    df.to_csv(os.path.join(log_dir, 'grid_search_results.csv'), index=False)
    print(f"Saved results to: {os.path.join(log_dir, 'grid_search_results.csv')}")
    
    # サマリー保存
    summary = {
        'timestamp': timestamp,
        'epsilon': args.epsilon,
        'n_samples': len(x_test),
        'clean_acc': clean_acc,
        'adv_acc_no_defense': adv_acc,
        'best_start_t': int(best_result['start_t']),
        'best_T_purify': int(best_result['T_purify']),
        'best_defended_acc': float(best_result['defended_acc']),
        'best_improvement': float(best_result['improvement']),
        'param_grid': {
            'start_t': start_t_values,
            'T_purify': t_purify_values
        }
    }
    
    with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {os.path.join(log_dir, 'summary.json')}")
    
    # テキストサマリー
    with open(os.path.join(log_dir, 'summary.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("DDPM Parameter Grid Search Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: ChestX-ray\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: FGSM (epsilon={args.epsilon:.4f})\n")
        f.write(f"Samples: {len(x_test)} ({args.n_samples_per_class} per class)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Baseline Results\n")
        f.write("-"*70 + "\n")
        f.write(f"Clean accuracy: {clean_acc:.4f}\n")
        f.write(f"Adversarial accuracy (no defense): {adv_acc:.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Best Parameters\n")
        f.write("-"*70 + "\n")
        f.write(f"start_t: {best_result['start_t']}\n")
        f.write(f"T_purify: {best_result['T_purify']}\n")
        f.write(f"Defended accuracy: {best_result['defended_acc']:.4f}\n")
        f.write(f"Improvement: {best_result['improvement']:+.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Top 10 Results\n")
        f.write("-"*70 + "\n")
        f.write(top10.to_string(index=False))
        f.write("\n")
    
    print(f"\n✅ Grid search completed!")
    print(f"✅ Results saved to: {log_dir}")
    
    return df, best_result


if __name__ == '__main__':
    main()
