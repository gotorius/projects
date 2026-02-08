"""
PCam Dataset - AutoAttack + DDPM Purification Defense (ViT Classifier)
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. AutoAttack敵対的画像の分類精度（防御なし）
4. AutoAttack敵対的画像を浄化した後の分類精度（防御あり）
"""

"""
# 基本実行
python ddpm_autoattack_eval.py

# パラメータ指定
python ddpm_autoattack_eval.py \
    --epsilon 0.03137 \
    --start_t 280 \
    --T_purify 300 \
    --gpu 0
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
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not found. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='PCam AutoAttack + DDPM Defense (ViT)')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='Perturbation epsilon')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--version', type=str, default='standard', choices=['standard', 'plus', 'rand'])
    
    parser.add_argument('--start_t', type=int, default=280)
    parser.add_argument('--T_purify', type=int, default=300)
    parser.add_argument('--eta', type=float, default=1.0)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='/mnt/data1/gotou/projects/resnet/pcam/ddpm/checkpoints/best_model.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/ddpm/autoattack/results')
    
    parser.add_argument('--gpu', type=int, default=0)
    
    return parser.parse_args()


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
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.mha(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ff(self.ln(x_flat))
        return x_flat.transpose(1, 2).view(b, c, h, w)


class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256, attn_heads=4):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2), nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        attn = lambda ch: SelfAttention2d(ch, attn_heads)
        
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
        self.out_conv = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_ch, 3, padding=1))
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.attn3(self.enc3(self.down2(e2), t_emb))
        e4 = self.attn4(self.enc4(self.down3(e3), t_emb))
        b = self.attn_bot(self.bot2(self.bot1(self.down4(e4), t_emb), t_emb))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)


class DDPMPurifierRGB(nn.Module):
    def __init__(self, ddpm_model, device, T_steps=1000, start_t=80, T_purify=50, eta=0.0):
        super().__init__()
        self.ddpm = ddpm_model
        self.device = device
        self.T_steps = T_steps
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
        
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
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
        f = (t / timesteps + s) / (1 + s)
        alphas_bar = torch.cos(f * torch.pi / 2) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return betas.clamp(min=1e-8, max=0.999).to(torch.float32)
    
    def p_sample(self, x_t, t_batch):
        t = t_batch[0].item()
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].view(-1, 1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_alphas_t = self.sqrt_alphas[t].view(-1, 1, 1, 1)
        
        eps_pred = self.ddpm(x_t, t_batch)
        x0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * eps_pred) / sqrt_alpha_cumprod_t
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        posterior_mean_coef1 = (betas_t * torch.sqrt(alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t + 1e-8)
        posterior_mean_coef2 = ((1.0 - alpha_cumprod_prev_t) * sqrt_alphas_t) / (1.0 - alpha_cumprod_t + 1e-8)
        model_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * x_t
        
        posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1).clamp(min=1e-20)
        
        if t == 0:
            return model_mean, x0_pred
        else:
            noise = torch.randn_like(x_t)
            return model_mean + self.eta * torch.sqrt(posterior_var_t) * noise, x0_pred
    
    def purify(self, x_pixel_rgb):
        b = x_pixel_rgb.size(0)
        device = x_pixel_rgb.device
        x_ddpm = x_pixel_rgb * 2.0 - 1.0
        
        t0 = torch.full((b,), self.start_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_ddpm)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_ddpm + sqrt_one_minus_alpha_bar * noise
        
        x0_pred = None
        for t_ in range(self.start_t, max(self.start_t - self.T_purify, 0), -1):
            t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
            x_t, x0_pred = self.p_sample(x_t, t_batch)
        
        x0_hat = x0_pred if x0_pred is not None else x_t
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        return torch.clamp((x0_hat + 1.0) / 2.0, 0, 1)
    
    def forward(self, x):
        return self.purify(x)


class ViTClassifierWrapper(nn.Module):
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        return self.classifier((x - self.mean) / self.std)


class DDPMDefenseWrapper(nn.Module):
    def __init__(self, purifier, classifier, mean, std):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_purified = self.purifier(x)
        return self.classifier((x_purified - self.mean) / self.std)


def load_cached_samples(cached_path):
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['normal', 'tumor'])
    print(f"Loaded {len(x_test)} samples, Classes: {classes}")
    return x_test, y_test, classes


def load_models(args, device):
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_features, 2))
    
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    
    ddpm = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(device)
    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if isinstance(ckpt, dict):
        if 'ema_state_dict' in ckpt:
            ddpm.load_state_dict(ckpt['ema_state_dict'])
        elif 'model_state_dict' in ckpt:
            ddpm.load_state_dict(ckpt['model_state_dict'])
        else:
            ddpm.load_state_dict(ckpt)
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    
    print(f"Loaded ViT classifier and DDPM")
    return classifier, ddpm


def get_predictions_and_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds, correct = [], 0
    
    with torch.no_grad():
        for i in range(n_batches):
            x_batch = x[i*bs:(i+1)*bs].to(device)
            y_batch = y[i*bs:(i+1)*bs].to(device)
            outputs = model(x_batch)
            batch_preds = outputs.argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    
    return torch.cat(preds).numpy(), correct / len(x)


def print_confusion_matrix(y_true, y_pred, title, classes=None):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        print(f"\n{title}:")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"  Acc: {accuracy:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return {}


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    classifier, ddpm = load_models(args, device)
    
    purifier = DDPMPurifierRGB(ddpm, device, start_t=args.start_t, T_purify=args.T_purify, eta=args.eta).to(device)
    
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = DDPMDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    print(f"\n{'='*70}")
    print("PCam - AutoAttack + DDPM Defense (ViT)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f}, Norm: {args.norm}, Version: {args.version}")
    
    results = {}
    
    print("\n[1/4] Clean accuracy...")
    _, clean_acc = get_predictions_and_accuracy(classifier_model, x_test, y_test, args.batch_size, device)
    results['clean_acc_classifier'] = clean_acc
    print(f"Clean accuracy: {clean_acc:.4f}")
    
    print("\n[2/4] Clean + DDPM accuracy...")
    _, clean_purified_acc = get_predictions_and_accuracy(defense_model, x_test, y_test, args.batch_size, device)
    results['clean_acc_with_ddpm'] = clean_purified_acc
    print(f"Clean + DDPM: {clean_purified_acc:.4f}")
    
    print("\n[3/4] Running AutoAttack...")
    start_time = time.time()
    adversary = AutoAttack(classifier_model, norm=args.norm, eps=args.epsilon, version=args.version, verbose=True)
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
    attack_time = time.time() - start_time
    
    _, adv_acc_no_defense = get_predictions_and_accuracy(classifier_model, x_adv.cpu(), y_test, args.batch_size, device)
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    
    print("\n[4/4] Adversarial + DDPM accuracy...")
    _, adv_defended_acc = get_predictions_and_accuracy(defense_model, x_adv.cpu(), y_test, args.batch_size, device)
    results['adv_acc_with_ddpm'] = adv_defended_acc
    results['defense_improvement'] = adv_defended_acc - adv_acc_no_defense
    print(f"Adversarial + DDPM: {adv_defended_acc:.4f}")
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Clean: {clean_acc:.4f} | Clean+DDPM: {clean_purified_acc:.4f}")
    print(f"Adv: {adv_acc_no_defense:.4f} | Adv+DDPM: {adv_defended_acc:.4f}")
    print(f"Defense improvement: {results['defense_improvement']:+.4f}")
    print(f"Attack time: {attack_time:.2f}s")
    
    torch.save({
        'x_clean': x_test.cpu(), 'x_adv': x_adv.cpu(), 'y': y_test.cpu(),
        'epsilon': args.epsilon
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump({'dataset': 'PCam', 'classifier': 'ViT-B/16', 'args': vars(args), **results}, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")


if __name__ == '__main__':
    main()
