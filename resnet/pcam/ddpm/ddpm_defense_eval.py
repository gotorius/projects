"""
DiffPure (DDPM-based Defense) Evaluation Script for PCam Dataset
Supports: FGSM, PGD, AutoAttack

Usage:
    python ddpm_defense_eval.py --attack all --num_samples 100
    python ddpm_defense_eval.py --attack fgsm --num_samples 50
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from tqdm.auto import tqdm
import math


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DiffPure Defense Evaluation for PCam')
    
    # 攻撃設定
    parser.add_argument('--attack', type=str, default='all',
                        choices=['fgsm', 'pgd', 'autoattack', 'all'],
                        help='Attack type to evaluate')
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Perturbation budget')
    
    # PGD設定
    parser.add_argument('--pgd_alpha', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=20,
                        help='PGD iterations')
    
    # DiffPure設定
    parser.add_argument('--t_purify', type=int, default=250,
                        help='Purification timestep (0 = no purification)')
    parser.add_argument('--use_purification', action='store_true',
                        help='Enable DiffPure purification')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/checkpoints/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/kaggle/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/test',
                        help='Test data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/eval_results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== U-Net Components ==========
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
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

        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        
        self.enc2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        
        self.enc3 = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)
        
        self.enc4 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)

        # Bottleneck
        self.bot1 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch * 16, base_ch * 8, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch * 8, base_ch * 4, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim)

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
        d3 = self.down3(e3)
        
        e4 = self.enc4(d3, t_emb)
        d4 = self.down4(e4)

        # Bottleneck
        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)

        # Decode
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

        return self.out_conv(u1)


# ========== DDPM Diffusion ==========
class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For sampling
        self.posterior_variance = torch.zeros_like(self.betas)
        self.posterior_variance[1:] = (
            self.betas[1:] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        )
        self.posterior_variance[0] = 1e-8

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def p_sample(self, model, x_t, t):
        """Reverse diffusion step"""
        t_scalar = t[0].item()
        
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)
        
        # Predict noise
        eps_pred = model(x_t, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alpha_t * eps_pred)
        
        if t_scalar == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def purify(self, model, x, t_purify):
        """
        DiffPure purification
        1. Add noise to timestep t_purify
        2. Denoise back to t=0
        """
        if t_purify == 0:
            return x
        
        batch_size = x.shape[0]
        
        # Normalize to [-1, 1] for diffusion
        x = x * 2.0 - 1.0
        
        # Forward: add noise to t_purify
        t_batch = torch.full((batch_size,), t_purify - 1, device=self.device, dtype=torch.long)
        x_noisy = self.q_sample(x, t_batch)
        
        # Reverse: denoise from t_purify to 0
        x_denoised = x_noisy
        for t in reversed(range(t_purify)):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x_denoised = self.p_sample(model, x_denoised, t_batch)
        
        # Back to [0, 1]
        x_denoised = torch.clamp((x_denoised + 1.0) / 2.0, 0, 1)
        
        return x_denoised


# ========== Attacks ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM attack"""
    x = x.clone().to(device)
    x.requires_grad = True
    
    outputs = model(x)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


def pgd_attack(model, x, y, epsilon, alpha, steps, device):
    """PGD attack"""
    x_adv = x.clone().to(device)
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1).detach()
    
    for _ in range(steps):
        x_adv.requires_grad = True
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y.to(device))
        loss.backward()
        
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    
    return x_adv


def autoattack_eval(model, x, y, epsilon, device):
    """AutoAttack (requires autoattack library)"""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("AutoAttack not installed. Install with: pip install git+https://github.com/fra31/auto-attack")
        return x
    
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard', device=device)
    x_adv = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    
    return x_adv


# ========== Evaluation ==========
def evaluate(model, dataloader, device, attack_fn=None, purifier=None, desc="Evaluating"):
    """Evaluate model with optional attack and purification"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc=desc):
        images, labels = images.to(device), labels.to(device)
        
        # Attack
        if attack_fn is not None:
            images = attack_fn(images, labels)
        
        # Purification
        if purifier is not None:
            images = purifier(images)
        
        # Prediction
        with torch.no_grad():
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def print_confusion_matrix(y_true, y_pred, title, classes):
    """Print confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{title}")
    print("-" * 50)
    print(f"{'':>15} {'Pred ' + classes[0]:>15} {'Pred ' + classes[1]:>15}")
    print(f"{'True ' + classes[0]:>15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'True ' + classes[1]:>15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# ========== Main Evaluation ==========
def run_attack_evaluation(args, attack_name, attack_fn, classifier, purifier, test_loader, device, classes):
    """Run evaluation for a single attack"""
    print(f"\n{'='*70}")
    print(f"Evaluating {attack_name.upper()}")
    print(f"{'='*70}")
    
    results = {}
    
    # 1. Clean accuracy
    print("\n[1/4] Clean images (no attack, no defense)...")
    clean_acc, clean_preds, clean_labels = evaluate(
        classifier, test_loader, device, desc="Clean"
    )
    print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # 2. Clean + Purification
    if args.use_purification:
        print(f"\n[2/4] Clean images with DiffPure (t={args.t_purify})...")
        clean_purified_acc, clean_purified_preds, _ = evaluate(
            classifier, test_loader, device, purifier=purifier, desc="Clean + DiffPure"
        )
        print(f"Clean + DiffPure accuracy: {clean_purified_acc:.4f}")
        results['clean_purified_acc'] = clean_purified_acc
    
    # 3. Attack (no defense)
    print(f"\n[3/4] {attack_name.upper()} attack (no defense)...")
    
    def attack_wrapper(images, labels):
        return attack_fn(classifier, images, labels, args.epsilon, device)
    
    adv_acc, adv_preds, adv_labels = evaluate(
        classifier, test_loader, device, attack_fn=attack_wrapper, desc=f"{attack_name.upper()} attack"
    )
    print(f"{attack_name.upper()} attack accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    
    # 4. Attack + Defense
    if args.use_purification:
        print(f"\n[4/4] {attack_name.upper()} attack with DiffPure defense...")
        adv_defended_acc, adv_defended_preds, _ = evaluate(
            classifier, test_loader, device, 
            attack_fn=attack_wrapper, 
            purifier=purifier,
            desc=f"{attack_name.upper()} + DiffPure"
        )
        print(f"{attack_name.upper()} + DiffPure accuracy: {adv_defended_acc:.4f}")
        results['adv_defended_acc'] = adv_defended_acc
        results['defense_improvement'] = adv_defended_acc - adv_acc
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{attack_name.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"Clean accuracy:                    {results['clean_acc']:.4f}")
    if args.use_purification:
        print(f"Clean + DiffPure:                  {results['clean_purified_acc']:.4f}")
    print(f"{attack_name.upper()} attack (no defense):      {results['adv_acc_no_defense']:.4f}")
    if args.use_purification:
        print(f"{attack_name.upper()} + DiffPure:                {results['adv_defended_acc']:.4f}")
        print(f"Defense improvement:               {results['defense_improvement']:+.4f}")
    print(f"{'='*70}")
    
    # Confusion matrices
    print(f"\nConfusion Matrices for {attack_name.upper()}")
    cm_clean = print_confusion_matrix(clean_labels, clean_preds, "Clean", classes)
    cm_adv = print_confusion_matrix(adv_labels, adv_preds, f"{attack_name.upper()} (no defense)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'adv_no_defense': cm_adv
    }
    
    if args.use_purification:
        cm_clean_purified = print_confusion_matrix(clean_labels, clean_purified_preds, "Clean + DiffPure", classes)
        cm_adv_defended = print_confusion_matrix(adv_labels, adv_defended_preds, f"{attack_name.upper()} + DiffPure", classes)
        results['confusion_matrices']['clean_purified'] = cm_clean_purified
        results['confusion_matrices']['adv_defended'] = cm_adv_defended
    
    return results


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.attack}_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load classifier
    print(f"\nLoading classifier from {args.clf_ckpt}")
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    
    # Wrap classifier with normalization
    class NormalizedClassifier(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        
        def forward(self, x):
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)
            return self.model(x)
    
    classifier = NormalizedClassifier(classifier).to(device).eval()
    
    # Load DDPM
    purifier = None
    if args.use_purification:
        print(f"\nLoading DDPM from {args.ddpm_ckpt}")
        ddpm_ckpt = torch.load(args.ddpm_ckpt, map_location=device)
        
        # Get model parameters from checkpoint
        model_args = ddpm_ckpt.get('args', {})
        base_ch = model_args.get('base_channels', 64)
        
        unet = SimpleUNet(in_ch=3, base_ch=base_ch, time_emb_dim=256).to(device)
        unet.load_state_dict(ddpm_ckpt['model_state_dict'])
        unet.eval()
        
        diffusion = GaussianDiffusion(timesteps=1000, device=device)
        
        def purify_fn(x):
            return diffusion.purify(unet, x, args.t_purify)
        
        purifier = purify_fn
        print(f"DiffPure enabled with t_purify={args.t_purify}")
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    from torchvision import datasets
    full_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # Limit samples
    if args.num_samples < len(full_dataset):
        indices = torch.randperm(len(full_dataset))[:args.num_samples].tolist()
        test_dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        test_dataset = full_dataset
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    classes = full_dataset.classes
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Run evaluations
    all_results = {}
    
    if args.attack == 'all':
        attacks_to_run = ['fgsm', 'pgd', 'autoattack']
    else:
        attacks_to_run = [args.attack]
    
    for attack_name in attacks_to_run:
        print(f"\n\n{'#'*70}")
        print(f"# Running {attack_name.upper()} evaluation")
        print(f"{'#'*70}\n")
        
        # Select attack function
        if attack_name == 'fgsm':
            attack_fn = fgsm_attack
        elif attack_name == 'pgd':
            def pgd_fn(model, x, y, eps, dev):
                return pgd_attack(model, x, y, eps, args.pgd_alpha, args.pgd_steps, dev)
            attack_fn = pgd_fn
        elif attack_name == 'autoattack':
            attack_fn = autoattack_eval
        
        # Run evaluation
        start_time = time.time()
        results = run_attack_evaluation(
            args, attack_name, attack_fn, classifier, purifier, test_loader, device, classes
        )
        results['time'] = time.time() - start_time
        
        all_results[attack_name] = results
        
        # Save individual result
        result_file = os.path.join(output_dir, f'{attack_name}_results.json')
        with open(result_file, 'w') as f:
            save_results = {k: v for k, v in results.items() if k != 'confusion_matrices'}
            save_results['confusion_matrices'] = {
                k: {kk: int(vv) if isinstance(vv, (int, np.integer)) else float(vv)
                    for kk, vv in v.items()}
                for k, v in results['confusion_matrices'].items()
            }
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {result_file}")
    
    # Save config
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY OF ALL ATTACKS")
    print(f"{'='*70}")
    for attack_name, results in all_results.items():
        print(f"\n{attack_name.upper()}:")
        print(f"  Clean: {results['clean_acc']:.4f}")
        if 'clean_purified_acc' in results:
            print(f"  Clean + DiffPure: {results['clean_purified_acc']:.4f}")
        print(f"  Attack (no defense): {results['adv_acc_no_defense']:.4f}")
        if 'adv_defended_acc' in results:
            print(f"  Attack + DiffPure: {results['adv_defended_acc']:.4f}")
            print(f"  Improvement: {results['defense_improvement']:+.4f}")
        print(f"  Time: {results['time']:.2f}s")
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
