"""
DermMel - PGD Attack + DDPM Defense Evaluation (Improved)

改善点:
1. x0再構成を使用してノイズを抑制
2. 適切な正規化変換(ImageNet ⇔ DDPM)
3. cosineスケジュール対応

実行例:
python ddpm_pgd_eval_improved.py --epsilon 0.031 --t_purify 50 --start_t 80
"""

import os
import sys
import math
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel DDPM Defense Evaluation - PGD Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD perturbation epsilon')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD steps')
    parser.add_argument('--random_start', type=bool, default=True,
                        help='Random start for PGD')
    
    # DDPM浄化設定
    parser.add_argument('--t_purify', type=int, default=50,
                        help='Number of diffusion steps for purification')
    parser.add_argument('--start_t', type=int, default=80,
                        help='Starting timestep for reverse diffusion')
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
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/pgd/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
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


# ========== DDPM Purifier (Improved) ==========
class DDPMPurifierImproved(nn.Module):
    def __init__(self, unet, diffusion, device, t_purify=50, start_t=80, eta=0.0):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.device = device
        self.t_purify = t_purify
        self.start_t = start_t
        self.eta = eta
    
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


def load_cached_samples(path):
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f"Loaded {len(x_test)} samples from {path}")
    print(f"Classes: {classes}")
    return x_test, y_test, classes


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, alpha, steps, device, random_start=True):
    """PGD攻撃"""
    x = x.clone().to(device)
    y = y.to(device)
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    if random_start:
        x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x.clone()
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        x_norm = (x_adv - mean) / std
        outputs = model(x_norm)
        loss = F.cross_entropy(outputs, y)
        
        model.zero_grad()
        loss.backward()
        
        grad = x_adv.grad.detach()
        x_adv = x_adv.detach() + alpha * grad.sign()
        
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv.detach()


# ========== 評価関数 ==========
def evaluate(model, x_test, y_test, device, batch_size=16):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
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
            predictions.extend(predicted.cpu().numpy())
    
    return correct / total, np.array(predictions)


def evaluate_with_purification(purifier, classifier, x_test, y_test, device, batch_size=8, desc="Purifying"):
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_purified_all = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            x_purified = purifier(x_batch)
            x_purified_all.append(x_purified.cpu())
            
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    x_purified_all = torch.cat(x_purified_all, dim=0)
    return correct / total, np.array(predictions), x_purified_all


def compute_l2_norm(x1, x2):
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def print_confusion_matrix(y_true, y_pred, title, classes, file=None):
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def write_and_print(text):
        print(text)
        if file:
            file.write(text + '\n')
    
    write_and_print(f"\n{title}")
    write_and_print("-" * 60)
    
    header = f"{'':>15}" + "".join([f"Pred {c:>8}" for c in classes])
    write_and_print(header)
    
    for i, true_class in enumerate(classes):
        row = f"{'True ' + true_class:>15}" + "".join([f"{cm[i, j]:>12}" for j in range(len(classes))])
        write_and_print(row)
    
    write_and_print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {'cm': cm, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, labels, classes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), 10)
    
    for i in range(n):
        label = classes[labels[i]]
        images = [x_clean[i], x_adv[i], x_purified_clean[i], x_purified_adv[i]]
        grid = make_grid(images, nrow=4, padding=2, normalize=False)
        save_image(grid, os.path.join(save_dir, f'sample_{i}_{label}.png'))
    
    print(f"Saved {n} sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    classifier = load_classifier(args, device)
    unet, diffusion = load_ddpm(args, device)
    
    purifier = DDPMPurifierImproved(
        unet, diffusion, device,
        t_purify=args.t_purify, start_t=args.start_t, eta=args.eta
    )
    
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("PGD Attack + DDPM Defense Evaluation (DermMel)")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"       Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255), Steps: {args.pgd_steps}")
    write_and_print(f"DDPM: start_t={args.start_t}, t_purify={args.t_purify}, eta={args.eta}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    write_and_print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    write_and_print("\n[2/4] Evaluating clean images with DDPM purification...")
    clean_purified_acc, pred_clean_purified, x_purified_clean = evaluate_with_purification(
        purifier, classifier, x_test, y_test, device, args.batch_size, "Purifying clean images"
    )
    l2_clean_purified = compute_l2_norm(x_test, x_purified_clean)
    write_and_print(f"Clean accuracy (with DDPM): {clean_purified_acc:.4f}")
    write_and_print(f"L2 norm (clean vs purified): {l2_clean_purified:.4f}")
    results['clean_acc_with_ddpm'] = clean_purified_acc
    results['l2_clean_vs_purified'] = l2_clean_purified
    
    write_and_print("\n[3/4] Running PGD attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="PGD Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = pgd_attack(
            classifier, x_batch, y_batch,
            args.epsilon, args.alpha, args.pgd_steps, device, args.random_start
        )
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start_time
    
    l2_clean_adv = compute_l2_norm(x_test, x_adv)
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    write_and_print(f"L2 norm (clean vs adversarial): {l2_clean_adv:.4f}")
    write_and_print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_clean_adv
    results['attack_time'] = attack_time
    
    write_and_print("\n[4/4] Evaluating adversarial images with DDPM purification...")
    adv_purified_acc, pred_adv_purified, x_purified_adv = evaluate_with_purification(
        purifier, classifier, x_adv, y_test, device, args.batch_size, "Purifying adversarial images"
    )
    l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
    write_and_print(f"Adversarial accuracy (with DDPM): {adv_purified_acc:.4f}")
    write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
    results['adv_acc_with_ddpm'] = adv_purified_acc
    results['l2_adv_vs_purified'] = l2_adv_purified
    results['defense_improvement'] = adv_purified_acc - adv_acc
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"       Alpha: {args.alpha:.4f}, Steps: {args.pgd_steps}")
    write_and_print(f"Defense: DDPM, start_t={args.start_t}, t_purify={args.t_purify}")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With DDPM:                   {results['clean_acc_with_ddpm']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (PGD):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With DDPM:                   {results['adv_acc_with_ddpm']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    write_and_print(f"  Clean vs Purified:           {results['l2_clean_vs_purified']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    write_and_print(f"  Adversarial vs Purified:     {results['l2_adv_vs_purified']:.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Attack time: {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes, results_file)
    cm_results['clean_purified'] = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with DDPM)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_purified'] = print_confusion_matrix(y_true, pred_adv_purified, "4. Adversarial Images (with DDPM)", classes, results_file)
    
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(x_test[:10], x_adv[:10], x_purified_clean[:10], x_purified_adv[:10],
                       y_test[:10], classes, samples_dir)
    
    results_file.close()
    
    results_save = {
        'config': vars(args),
        'results': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results.items()},
        'confusion_matrices': {
            k: {
                'cm': v['cm'].tolist(),
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1'])
            } for k, v in cm_results.items()
        }
    }
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print(f"\nResults saved to {log_dir}")
    print(f"Text results: {os.path.join(log_dir, 'results.txt')}")


if __name__ == '__main__':
    main()
