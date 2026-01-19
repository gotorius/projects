"""
DermMel - DDPM Defense Parameter Grid Search

start_tとt_purifyの最適パラメータを探索

実行例:
python grid_search_params.py --gpu 0
"""

import os
import sys
import math
import argparse
import time
import json
from datetime import datetime
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from tqdm.auto import tqdm


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel DDPM Parameter Grid Search')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # グリッドサーチ設定
    parser.add_argument('--start_t_min', type=int, default=50,
                        help='Minimum start_t')
    parser.add_argument('--start_t_max', type=int, default=150,
                        help='Maximum start_t')
    parser.add_argument('--start_t_step', type=int, default=10,
                        help='Step size for start_t')
    parser.add_argument('--t_purify_min', type=int, default=10,
                        help='Minimum t_purify')
    parser.add_argument('--t_purify_max', type=int, default=100,
                        help='Maximum t_purify')
    parser.add_argument('--t_purify_step', type=int, default=10,
                        help='Step size for t_purify')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='Stochasticity parameter for DDIM')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/ddpm_out2/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/grid_search_results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for grid search (balanced)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== モデル定義 (from ddpm_train_v2.py) ==========
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
    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=256, attn_heads=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        
        def attn(ch):
            return SelfAttention2d(ch, num_heads=attn_heads)

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
        e3 = self.attn3(e3)
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        e4 = self.attn4(e4)
        d4 = self.down4(e4)

        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)
        b = self.attn_bot(b)

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


# ========== Gaussian Diffusion (Cosine Schedule) ==========
class GaussianDiffusion:
    def __init__(self, timesteps=1000, device='cuda', schedule='cosine'):
        self.timesteps = timesteps
        self.device = device
        
        if schedule == 'cosine':
            betas = self._make_cosine_schedule(timesteps)
        else:
            betas = torch.linspace(1e-4, 0.02, timesteps)
        
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = torch.zeros_like(self.betas)
        self.posterior_variance[1:] = self.betas[1:] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        self.posterior_variance[0] = 1e-8
    
    def _make_cosine_schedule(self, timesteps):
        s = 0.008
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = (t / timesteps + s) / (1 + s)
        alphas_bar = torch.cos(f * math.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = betas.clamp(min=1e-8, max=0.999)
        return betas.to(torch.float32)


# ========== DDPM Purifier ==========
class DDPMPurifier(nn.Module):
    def __init__(self, unet, diffusion, device, t_purify=50, start_t=80, eta=0.0):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.device = device
        self.t_purify = t_purify
        self.start_t = start_t
        self.eta = eta
    
    def set_params(self, start_t, t_purify):
        """パラメータを動的に設定"""
        self.start_t = start_t
        self.t_purify = t_purify
    
    @torch.no_grad()
    def forward(self, x):
        x_minus1to1 = x * 2.0 - 1.0
        
        batch_size = x.size(0)
        t0 = torch.full((batch_size,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_minus1to1)
        
        sqrt_alpha_bar_t0 = self.diffusion.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_1m_alpha_bar_t0 = self.diffusion.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t0 * x_minus1to1 + sqrt_1m_alpha_bar_t0 * noise
        
        eps_pred_final = None
        t_final = self.start_t
        
        for i in range(self.t_purify):
            curr_t = self.start_t - i
            if curr_t < 0:
                break
            
            t_batch = torch.full((batch_size,), curr_t, device=self.device, dtype=torch.long)
            eps_pred = self.unet(x_t, t_batch)
            
            alpha_t = self.diffusion.alphas[curr_t]
            alpha_bar_t = self.diffusion.alphas_cumprod[curr_t]
            
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
            )
            
            if curr_t > 0:
                z = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt(self.diffusion.posterior_variance[curr_t])
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            x_t = torch.clamp(x_t, -1.0, 1.0)
            
            eps_pred_final = eps_pred
            t_final = curr_t
        
        # x0再構成
        alpha_bar_tf = self.diffusion.alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_pred_final) / torch.sqrt(alpha_bar_tf + 1e-12)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        x_purified = (x0_hat + 1.0) / 2.0
        x_purified = torch.clamp(x_purified, 0, 1)
        
        return x_purified


# ========== モデル読み込み ==========
def load_classifier(args, device):
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    return classifier


def load_ddpm(args, device):
    unet = SimpleUNet(in_ch=3, base_ch=128, time_emb_dim=256, attn_heads=4).to(device)
    
    ddpm_ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if 'model_state_dict' in ddpm_ckpt:
        unet.load_state_dict(ddpm_ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ddpm_ckpt)
    
    unet.eval()
    
    diffusion = GaussianDiffusion(timesteps=1000, device=device, schedule='cosine')
    
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return unet, diffusion


def load_and_subsample(args, device):
    """キャッシュサンプルからバランスの取れたサブセットを取得"""
    data = torch.load(args.cached_samples, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    
    # クラスごとにインデックスを取得
    class_indices = {}
    for i, label in enumerate(y_test):
        label_int = label.item()
        if label_int not in class_indices:
            class_indices[label_int] = []
        class_indices[label_int].append(i)
    
    # 各クラスから均等にサンプリング
    samples_per_class = args.num_samples // len(classes)
    selected_indices = []
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    for label, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected = indices
        selected_indices.extend(selected)
    
    selected_indices = sorted(selected_indices)
    
    x_subset = x_test[selected_indices]
    y_subset = y_test[selected_indices]
    
    print(f"Selected {len(x_subset)} samples ({samples_per_class} per class)")
    for i, cls in enumerate(classes):
        count = (y_subset == i).sum().item()
        print(f"  {cls}: {count}")
    
    return x_subset, y_subset, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().to(device)
    x.requires_grad = True
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# ========== 評価関数 ==========
def evaluate(model, x_test, y_test, device, batch_size=16):
    model.eval()
    correct = 0
    total = 0
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            x_norm = (x_batch - mean) / std
            outputs = model(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


def evaluate_with_purification(purifier, classifier, x_test, y_test, device, batch_size=8):
    classifier.eval()
    
    correct = 0
    total = 0
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            x_purified = purifier(x_batch)
            
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


# ========== グリッドサーチ ==========
def grid_search(args, classifier, purifier, x_clean, x_adv, y_test, device):
    """パラメータのグリッドサーチを実行"""
    
    # パラメータ範囲
    start_t_values = list(range(args.start_t_min, args.start_t_max + 1, args.start_t_step))
    t_purify_values = list(range(args.t_purify_min, args.t_purify_max + 1, args.t_purify_step))
    
    print(f"\nGrid Search Parameters:")
    print(f"  start_t: {start_t_values}")
    print(f"  t_purify: {t_purify_values}")
    print(f"  Total combinations: {len(start_t_values) * len(t_purify_values)}")
    
    results = []
    best_result = None
    best_score = -1
    
    total_combos = len(start_t_values) * len(t_purify_values)
    pbar = tqdm(total=total_combos, desc="Grid Search")
    
    for start_t, t_purify in product(start_t_values, t_purify_values):
        # t_purifyがstart_tを超えないようにする
        if t_purify > start_t:
            pbar.update(1)
            continue
        
        # パラメータ設定
        purifier.set_params(start_t, t_purify)
        
        # クリーン画像の浄化後精度
        clean_purified_acc = evaluate_with_purification(
            purifier, classifier, x_clean, y_test, device, args.batch_size
        )
        
        # 敵対的画像の浄化後精度
        adv_purified_acc = evaluate_with_purification(
            purifier, classifier, x_adv, y_test, device, args.batch_size
        )
        
        # 防御改善量
        defense_improvement = adv_purified_acc  # 元々0%なので
        
        # 総合スコア: クリーン精度と敵対的精度の調和平均
        if clean_purified_acc > 0 and adv_purified_acc > 0:
            harmonic_mean = 2 * clean_purified_acc * adv_purified_acc / (clean_purified_acc + adv_purified_acc)
        else:
            harmonic_mean = 0
        
        result = {
            'start_t': start_t,
            't_purify': t_purify,
            'clean_purified_acc': clean_purified_acc,
            'adv_purified_acc': adv_purified_acc,
            'harmonic_mean': harmonic_mean
        }
        results.append(result)
        
        # ベスト更新チェック
        if harmonic_mean > best_score:
            best_score = harmonic_mean
            best_result = result
        
        pbar.set_postfix({
            'start_t': start_t,
            't_purify': t_purify,
            'clean': f'{clean_purified_acc:.2%}',
            'adv': f'{adv_purified_acc:.2%}',
            'best': f'{best_score:.2%}'
        })
        pbar.update(1)
    
    pbar.close()
    
    return results, best_result


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    unet, diffusion = load_ddpm(args, device)
    
    # 浄化器作成（初期パラメータは後で変更）
    purifier = DDPMPurifier(unet, diffusion, device, t_purify=50, start_t=80, eta=args.eta)
    
    # データ読み込み（バランスの取れたサブセット）
    x_clean, y_test, classes = load_and_subsample(args, device)
    
    print(f"\n{'='*70}")
    print("DDPM Parameter Grid Search (DermMel)")
    print(f"{'='*70}")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Samples: {len(x_clean)} (balanced)")
    print(f"Classes: {classes}")
    print(f"{'='*70}")
    
    # クリーン精度を確認
    clean_acc = evaluate(classifier, x_clean, y_test, device, args.batch_size)
    print(f"\nClean accuracy (no defense): {clean_acc:.4f}")
    
    # FGSM攻撃
    print("\nGenerating adversarial examples (FGSM)...")
    x_adv_list = []
    for i in tqdm(range(0, len(x_clean), args.batch_size), desc="FGSM Attack"):
        x_batch = x_clean[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    
    adv_acc = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    
    # グリッドサーチ実行
    print("\nStarting grid search...")
    start_time = time.time()
    results, best_result = grid_search(args, classifier, purifier, x_clean, x_adv, y_test, device)
    elapsed_time = time.time() - start_time
    
    # 結果表示
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Total combinations tested: {len(results)}")
    print(f"\nBest Parameters:")
    print(f"  start_t: {best_result['start_t']}")
    print(f"  t_purify: {best_result['t_purify']}")
    print(f"  Clean accuracy (with DDPM): {best_result['clean_purified_acc']:.4f}")
    print(f"  Adversarial accuracy (with DDPM): {best_result['adv_purified_acc']:.4f}")
    print(f"  Harmonic mean: {best_result['harmonic_mean']:.4f}")
    print(f"{'='*70}")
    
    # 上位10結果を表示
    sorted_results = sorted(results, key=lambda x: x['harmonic_mean'], reverse=True)
    print("\nTop 10 Results (by harmonic mean):")
    print("-" * 70)
    print(f"{'Rank':>4} {'start_t':>8} {'t_purify':>8} {'Clean':>8} {'Adv':>8} {'HMean':>8}")
    print("-" * 70)
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:>4} {r['start_t']:>8} {r['t_purify']:>8} {r['clean_purified_acc']:>8.4f} {r['adv_purified_acc']:>8.4f} {r['harmonic_mean']:>8.4f}")
    
    # 結果保存
    output_data = {
        'config': vars(args),
        'clean_acc_no_defense': clean_acc,
        'adv_acc_no_defense': adv_acc,
        'best_result': best_result,
        'all_results': results,
        'elapsed_time': elapsed_time
    }
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # テキストレポート保存
    with open(os.path.join(log_dir, 'results.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("DDPM Parameter Grid Search Results (DermMel)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Samples: {len(x_clean)} (balanced)\n")
        f.write(f"Total time: {elapsed_time:.2f}s\n\n")
        f.write(f"Clean accuracy (no defense): {clean_acc:.4f}\n")
        f.write(f"Adversarial accuracy (no defense): {adv_acc:.4f}\n\n")
        f.write("Best Parameters:\n")
        f.write(f"  start_t: {best_result['start_t']}\n")
        f.write(f"  t_purify: {best_result['t_purify']}\n")
        f.write(f"  Clean accuracy (with DDPM): {best_result['clean_purified_acc']:.4f}\n")
        f.write(f"  Adversarial accuracy (with DDPM): {best_result['adv_purified_acc']:.4f}\n")
        f.write(f"  Harmonic mean: {best_result['harmonic_mean']:.4f}\n\n")
        f.write("All Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'start_t':>8} {'t_purify':>8} {'Clean':>8} {'Adv':>8} {'HMean':>8}\n")
        f.write("-" * 70 + "\n")
        for r in sorted_results:
            f.write(f"{r['start_t']:>8} {r['t_purify']:>8} {r['clean_purified_acc']:>8.4f} {r['adv_purified_acc']:>8.4f} {r['harmonic_mean']:>8.4f}\n")
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
