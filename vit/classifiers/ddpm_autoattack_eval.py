"""
ChestX-ray Dataset - AutoAttack + DDPM Purification Defense (ViT Classifier)
AutoAttackによる強力な敵対的攻撃に対するDDPM防御の検証

AutoAttack:
- APGD-CE: Auto-PGD with cross-entropy loss
- APGD-DLR: Auto-PGD with difference of logits ratio loss  
- FAB: Fast Adaptive Boundary attack
- Square: Square attack (query-based)

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. AutoAttack敵対的画像の分類精度（防御なし）
4. AutoAttack敵対的画像を浄化した後の分類精度（防御あり）

実行例:
python ddpm_autoattack_eval.py --epsilon 0.031 --start_t 140 --T_purify 140 --gpu 0
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
    print("AutoAttack not found. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray AutoAttack + DDPM Defense (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon (pixel scale 0-1)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Attack norm')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # DDPM浄化設定
    parser.add_argument('--start_t', type=int, default=140,
                        help='Diffusion start timestep')
    parser.add_argument('--T_purify', type=int, default=140,
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
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.ln(x_flat))
        return x_flat.transpose(1, 2).view(b, c, h, w)


class SimpleUNet(nn.Module):
    """Simple U-Net for grayscale images (in_ch=1)"""
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResidualBlock(base_ch, base_ch*2, time_emb_dim)
        self.enc3 = ResidualBlock(base_ch*2, base_ch*4, time_emb_dim)
        self.enc4 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Attention
        self.attn1 = SelfAttention2d(base_ch*2, num_heads=4)
        self.attn2 = SelfAttention2d(base_ch*4, num_heads=4)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.bottleneck_attn = SelfAttention2d(base_ch*4, num_heads=4)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec4 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 2, stride=2)
        self.dec3 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.attn_dec3 = SelfAttention2d(base_ch*2, num_heads=4)
        
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 2, stride=2)
        self.dec2 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        
        self.out = nn.Conv2d(base_ch, in_ch, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.pool(e1), t_emb)
        e2 = self.attn1(e2)
        e3 = self.enc3(self.pool(e2), t_emb)
        e3 = self.attn2(e3)
        e4 = self.enc4(self.pool(e3), t_emb)
        
        b = self.bottleneck(self.pool(e4), t_emb)
        b = self.bottleneck_attn(b)
        
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1), t_emb)
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), t_emb)
        d3 = self.attn_dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)
        
        return self.out(d1)


# ========== DDPM浄化クラス ==========
class DDPMPurifierGray(nn.Module):
    """DDPM浄化器（グレースケール用）
    RGB→グレースケール→DDPM浄化→RGB変換
    """
    def __init__(self, model, device, T=1000, beta_start=1e-4, beta_end=0.02, prediction_type='v'):
        super().__init__()
        self.model = model
        self.device = device
        self.T = T
        self.prediction_type = prediction_type
        
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        self.to(device)
    
    def _rgb_to_gray(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray
    
    def _gray_to_rgb(self, x):
        return x.repeat(1, 3, 1, 1)
    
    def _to_model_space(self, x):
        return x * 2 - 1
    
    def _from_model_space(self, x):
        return (x + 1) / 2
    
    def add_noise(self, x, t):
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise, noise
    
    @torch.no_grad()
    def purify(self, x, start_t, T_purify, eta=0.0):
        x_gray = self._rgb_to_gray(x)
        x_gray = self._to_model_space(x_gray)
        
        t_tensor = torch.full((x_gray.size(0),), start_t, device=self.device, dtype=torch.long)
        x_noisy, _ = self.add_noise(x_gray, t_tensor)
        
        timesteps = list(range(start_t, start_t - T_purify, -1))
        
        for t in timesteps:
            if t <= 0:
                break
            
            t_batch = torch.full((x_noisy.size(0),), t, device=self.device, dtype=torch.long)
            model_output = self.model(x_noisy, t_batch)
            
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            if self.prediction_type == 'v':
                x0_pred = sqrt_alpha_t * x_noisy - sqrt_one_minus_alpha_t * model_output
                eps_pred = sqrt_alpha_t * model_output + sqrt_one_minus_alpha_t * x_noisy
            else:
                eps_pred = model_output
                x0_pred = (x_noisy - sqrt_one_minus_alpha_t * eps_pred) / sqrt_alpha_t
            
            x0_pred = x0_pred.clamp(-1, 1)
            
            sqrt_alpha_prev = torch.sqrt(alpha_prev)
            sqrt_one_minus_alpha_prev = torch.sqrt(1 - alpha_prev)
            
            x_noisy = sqrt_alpha_prev * x0_pred + sqrt_one_minus_alpha_prev * eps_pred
        
        x_purified = self._from_model_space(x_noisy)
        x_purified = x_purified.clamp(0, 1)
        x_purified = self._gray_to_rgb(x_purified)
        
        return x_purified


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


# ========== DDPM防御付き分類器ラッパー ==========
class DDPMDefendedClassifier(nn.Module):
    """DDPM防御 + ViT分類器のラッパー（AutoAttack用）"""
    def __init__(self, classifier_wrapper, purifier, start_t, T_purify, eta=0.0):
        super().__init__()
        self.classifier = classifier_wrapper
        self.purifier = purifier
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
    
    def forward(self, x):
        x_purified = self.purifier.purify(x, self.start_t, self.T_purify, self.eta)
        return self.classifier(x_purified)


# ========== モデル読み込み ==========
def load_models(args, device):
    """モデルを読み込み"""
    # ViT分類器
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 2)
    )
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier = classifier.to(device).eval()
    print(f"Loaded ViT classifier from {args.clf_ckpt}")
    
    # DDPMモデル
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
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return classifier, ddpm


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NORMAL', 'PNEUMONIA'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    return x_test, y_test, classes


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    
    return correct / len(x)


def get_predictions(model, x, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds.append(outputs.argmax(dim=1).cpu())
    
    return torch.cat(preds).numpy()


# ========== 混同行列出力 ==========
def print_confusion_matrix(y_true, y_pred, title, classes=None):
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


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified, y_true, preds_clean, preds_adv, preds_defended,
                       classes, save_dir, max_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        pred_clean = classes[preds_clean[i]] if classes else str(preds_clean[i])
        pred_adv = classes[preds_adv[i]] if classes else str(preds_adv[i])
        pred_def = classes[preds_defended[i]] if classes else str(preds_defended[i])
        
        quad = torch.cat([x_clean[i:i+1], x_adv[i:i+1], x_purified[i:i+1]], dim=0)
        grid = make_grid(quad, nrow=3, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}_clean{pred_clean}_adv{pred_adv}_def{pred_def}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")


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
    log_dir = os.path.join(args.output_dir, f"autoattack_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, ddpm = load_models(args, device)
    purifier = DDPMPurifierGray(ddpm, device).to(device)
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # DDPM防御付き分類器
    defended_model = DDPMDefendedClassifier(
        classifier_model, purifier, args.start_t, args.T_purify, args.eta
    ).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + DDPM Defense Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"Version: {args.version}")
    print(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}, eta={args.eta}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with DDPM purification...")
    clean_purified_acc = get_accuracy(defended_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with DDPM): {clean_purified_acc:.4f}")
    results['clean_purified_acc'] = clean_purified_acc
    
    # ========== 3. AutoAttack ==========
    print("\n[3/4] Running AutoAttack...")
    start_time = time.time()
    
    # AutoAttack on undefended model
    adversary = AutoAttack(classifier_model, norm=args.norm, eps=args.epsilon, version=args.version, device=device)
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
    
    attack_time = time.time() - start_time
    print(f"AutoAttack completed in {attack_time:.2f}s")
    
    # 敵対的画像の精度（防御なし）
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度 ==========
    print("\n[4/4] Evaluating adversarial images with DDPM purification...")
    adv_defended_acc = get_accuracy(defended_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with DDPM): {adv_defended_acc:.4f}")
    results['adv_defended_acc'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f}, Norm: {args.norm}")
    print(f"Defense: DDPM (start_t={args.start_t}, T_purify={args.T_purify})")
    print(f"-"*70)
    print(f"Clean accuracy:              {results['clean_acc']:.4f}")
    print(f"Clean accuracy (with DDPM):  {results['clean_purified_acc']:.4f}")
    print(f"Adversarial (no defense):    {results['adv_acc_no_defense']:.4f}")
    print(f"Adversarial (with DDPM):     {results['adv_defended_acc']:.4f}")
    print(f"Defense improvement:         {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions(defended_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "2. AutoAttack Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "3. AutoAttack Images (with DDPM)", classes)
    
    # ==================== サンプル画像保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified = []
    for i in range(n_samples):
        x_pur = purifier.purify(x_adv[i:i+1].to(device), args.start_t, args.T_purify, args.eta)
        x_purified.append(x_pur.cpu())
    x_purified = torch.cat(x_purified, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(),
        x_adv[:n_samples].cpu(),
        x_purified,
        y_test[:n_samples].cpu().numpy(),
        pred_clean[:n_samples],
        pred_adv_no_def[:n_samples],
        pred_adv_defended[:n_samples],
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # ==================== 結果保存 ====================
    # 敵対的サンプル保存
    torch.save({
        'x_clean': x_test.cpu(),
        'x_adv': x_adv.cpu(),
        'y': y_test.cpu(),
        'epsilon': args.epsilon,
        'attack': 'autoattack',
        'version': args.version,
        'norm': args.norm,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    
    # サマリー保存
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - AutoAttack + DDPM Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: AutoAttack ({args.version})\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Norm: {args.norm}\n")
        f.write(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}, eta={args.eta}\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        f.write(f"Clean accuracy:              {results['clean_acc']:.4f}\n")
        f.write(f"Clean accuracy (with DDPM):  {results['clean_purified_acc']:.4f}\n")
        f.write(f"Adversarial (no defense):    {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"Adversarial (with DDPM):     {results['adv_defended_acc']:.4f}\n")
        f.write(f"Defense improvement:         {results['defense_improvement']:+.4f}\n\n")
        f.write(f"Attack time: {results['attack_time']:.2f}s\n")
    
    # JSON保存
    results_json = {
        'classifier': 'ViT-B/16',
        'attack': 'autoattack',
        'version': args.version,
        'norm': args.norm,
        'epsilon': args.epsilon,
        'ddpm_start_t': args.start_t,
        'ddpm_T_purify': args.T_purify,
        **results
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")
    
    return results


if __name__ == '__main__':
    main()
