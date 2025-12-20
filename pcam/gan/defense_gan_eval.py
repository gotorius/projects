"""
Defense-GAN: PCam データセット用 敵対的防御評価コード

Reference:
    "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
    Pouya Samangouei, Maya Kabkab, Rama Chellappa
    ICLR 2018

Defense-GANは、GANの生成器を使って入力画像を「浄化」し、
敵対的摂動を除去する防御手法です。

浄化プロセス:
1. 入力画像 x に対して、最適な潜在変数 z* を勾配降下法で探索
2. z* から生成された画像 G(z*) を分類器への入力として使用

Supports: FGSM, PGD, AutoAttack

Usage:
    python defense_gan_eval.py --attack all --num_samples 100
    python defense_gan_eval.py --attack fgsm --use_defense
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
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from tqdm.auto import tqdm
import math


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation for PCam')
    
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
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true',
                        help='Enable Defense-GAN purification')
    parser.add_argument('--rec_iters', type=int, default=200,
                        help='Number of gradient descent iterations for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=10,
                        help='Number of random restarts for reconstruction')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints/best_model.pth',
                        help='Defense-GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/kaggle/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/test',
                        help='Test data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Generator (DCGAN-based) ==========
class Generator(nn.Module):
    """
    DCGAN-based Generator for 224x224 images
    """
    def __init__(self, latent_dim=128, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial projection: latent -> (ngf*16) x 7 x 7
        self.init_size = 7
        self.fc = nn.Linear(latent_dim, ngf * 16 * self.init_size * self.init_size)
        
        self.main = nn.Sequential(
            # 7x7 -> 14x14
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, self.init_size, self.init_size)
        return self.main(x)


# ========== Defense-GAN Purification ==========
class DefenseGAN:
    """
    Defense-GAN purification module.
    
    Given an input image x, we search for the latent vector z* that
    minimizes the reconstruction error ||G(z) - x||_2.
    
    The purified image is G(z*).
    """
    def __init__(self, generator, device, rec_iters=200, rec_rr=10, rec_lr=0.01):
        """
        Args:
            generator: Pre-trained GAN generator
            device: torch device
            rec_iters: Number of gradient descent iterations
            rec_rr: Number of random restarts
            rec_lr: Learning rate for gradient descent
        """
        self.generator = generator
        self.device = device
        self.rec_iters = rec_iters
        self.rec_rr = rec_rr
        self.rec_lr = rec_lr
        self.latent_dim = generator.latent_dim
        
        # Put generator in eval mode
        self.generator.eval()
    
    def purify(self, x):
        """
        Purify input images using Defense-GAN.
        
        Args:
            x: Input images in [0, 1] range, shape (batch, 3, H, W)
        
        Returns:
            Purified images in [0, 1] range
        """
        batch_size = x.shape[0]
        
        # Convert to [-1, 1] for generator
        x_target = x * 2.0 - 1.0
        
        # Initialize best reconstructions
        best_z = torch.zeros(batch_size, self.latent_dim, device=self.device)
        best_loss = torch.full((batch_size,), float('inf'), device=self.device)
        best_recon = torch.zeros_like(x_target)
        
        # Multiple random restarts
        for rr in range(self.rec_rr):
            # Initialize z randomly
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            
            # Optimizer for z
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            # Gradient descent to find optimal z
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                
                # Generate image from z
                recon = self.generator(z)
                
                # Reconstruction loss (per sample)
                loss_per_sample = ((recon - x_target) ** 2).view(batch_size, -1).mean(dim=1)
                loss = loss_per_sample.sum()
                
                loss.backward()
                optimizer.step()
            
            # Check if this restart is better
            with torch.no_grad():
                recon = self.generator(z)
                loss_per_sample = ((recon - x_target) ** 2).view(batch_size, -1).mean(dim=1)
                
                # Update best for each sample
                improved = loss_per_sample < best_loss
                best_loss[improved] = loss_per_sample[improved]
                best_z[improved] = z[improved]
                best_recon[improved] = recon[improved]
        
        # Generate final purified images
        with torch.no_grad():
            purified = self.generator(best_z)
            # Convert back to [0, 1]
            purified = (purified + 1.0) / 2.0
            purified = torch.clamp(purified, 0, 1)
        
        return purified
    
    def purify_batch_efficient(self, x):
        """
        More memory-efficient batch purification.
        Process one image at a time but with multiple random restarts.
        """
        batch_size = x.shape[0]
        purified = []
        
        for i in range(batch_size):
            xi = x[i:i+1]
            xi_purified = self._purify_single(xi)
            purified.append(xi_purified)
        
        return torch.cat(purified, dim=0)
    
    def _purify_single(self, x):
        """Purify a single image with multiple random restarts."""
        x_target = x * 2.0 - 1.0
        
        best_z = None
        best_loss = float('inf')
        
        for rr in range(self.rec_rr):
            z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                recon = self.generator(z)
                loss = F.mse_loss(recon, x_target)
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                recon = self.generator(z)
                final_loss = F.mse_loss(recon, x_target).item()
                
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_z = z.clone()
        
        with torch.no_grad():
            purified = self.generator(best_z)
            purified = (purified + 1.0) / 2.0
            purified = torch.clamp(purified, 0, 1)
        
        return purified


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
def run_attack_evaluation(args, attack_name, attack_fn, classifier, defense_gan, test_loader, device, classes, output_dir):
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
    
    # 2. Clean + Defense-GAN
    if args.use_defense:
        print(f"\n[2/4] Clean images with Defense-GAN...")
        clean_defended_acc, clean_defended_preds, _ = evaluate(
            classifier, test_loader, device, purifier=defense_gan.purify, desc="Clean + Defense-GAN"
        )
        print(f"Clean + Defense-GAN accuracy: {clean_defended_acc:.4f}")
        results['clean_defended_acc'] = clean_defended_acc
    
    # 3. Attack (no defense)
    print(f"\n[3/4] {attack_name.upper()} attack (no defense)...")
    
    def attack_wrapper(images, labels):
        return attack_fn(classifier, images, labels, args.epsilon, device)
    
    adv_acc, adv_preds, adv_labels = evaluate(
        classifier, test_loader, device, attack_fn=attack_wrapper, desc=f"{attack_name.upper()} attack"
    )
    print(f"{attack_name.upper()} attack accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    
    # 4. Attack + Defense-GAN
    if args.use_defense:
        print(f"\n[4/4] {attack_name.upper()} attack with Defense-GAN...")
        adv_defended_acc, adv_defended_preds, _ = evaluate(
            classifier, test_loader, device, 
            attack_fn=attack_wrapper, 
            purifier=defense_gan.purify,
            desc=f"{attack_name.upper()} + Defense-GAN"
        )
        print(f"{attack_name.upper()} + Defense-GAN accuracy: {adv_defended_acc:.4f}")
        results['adv_defended_acc'] = adv_defended_acc
        results['defense_improvement'] = adv_defended_acc - adv_acc
    
    # Print results
    print(f"\n{'='*70}")
    print(f"{attack_name.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"Clean accuracy:                    {results['clean_acc']:.4f}")
    if args.use_defense:
        print(f"Clean + Defense-GAN:               {results['clean_defended_acc']:.4f}")
    print(f"{attack_name.upper()} attack (no defense):      {results['adv_acc_no_defense']:.4f}")
    if args.use_defense:
        print(f"{attack_name.upper()} + Defense-GAN:            {results['adv_defended_acc']:.4f}")
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
    
    if args.use_defense:
        cm_clean_defended = print_confusion_matrix(clean_labels, clean_defended_preds, "Clean + Defense-GAN", classes)
        cm_adv_defended = print_confusion_matrix(adv_labels, adv_defended_preds, f"{attack_name.upper()} + Defense-GAN", classes)
        results['confusion_matrices']['clean_defended'] = cm_clean_defended
        results['confusion_matrices']['adv_defended'] = cm_adv_defended
    
    return results


def save_sample_images(images_dict, output_dir, filename):
    """Save sample images for visualization"""
    n_samples = min(8, next(iter(images_dict.values())).size(0))
    
    rows = []
    for name, imgs in images_dict.items():
        rows.append(imgs[:n_samples])
    
    grid = make_grid(torch.cat(rows, dim=0), nrow=n_samples, normalize=True)
    save_image(grid, os.path.join(output_dir, filename))


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory based on attack type
    if args.use_defense:
        output_subdir = f"{args.attack}"
    else:
        output_subdir = f"{args.attack}_no_defense"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, output_subdir, f"eps{args.epsilon:.4f}_{timestamp}")
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
    
    # Load Defense-GAN
    defense_gan = None
    if args.use_defense:
        print(f"\nLoading Defense-GAN from {args.gan_ckpt}")
        gan_ckpt = torch.load(args.gan_ckpt, map_location=device)
        
        # Get model parameters from checkpoint
        model_args = gan_ckpt.get('args', {})
        latent_dim = model_args.get('latent_dim', 128)
        ngf = model_args.get('ngf', 64)
        
        generator = Generator(latent_dim=latent_dim, ngf=ngf).to(device)
        generator.load_state_dict(gan_ckpt['generator_state_dict'])
        generator.eval()
        
        defense_gan = DefenseGAN(
            generator=generator,
            device=device,
            rec_iters=args.rec_iters,
            rec_rr=args.rec_rr,
            rec_lr=args.rec_lr
        )
        print(f"Defense-GAN enabled:")
        print(f"  - Reconstruction iterations: {args.rec_iters}")
        print(f"  - Random restarts: {args.rec_rr}")
        print(f"  - Learning rate: {args.rec_lr}")
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
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
            args, attack_name, attack_fn, classifier, defense_gan, test_loader, device, classes, output_dir
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
        if 'clean_defended_acc' in results:
            print(f"  Clean + Defense-GAN: {results['clean_defended_acc']:.4f}")
        print(f"  Attack (no defense): {results['adv_acc_no_defense']:.4f}")
        if 'adv_defended_acc' in results:
            print(f"  Attack + Defense-GAN: {results['adv_defended_acc']:.4f}")
            print(f"  Improvement: {results['defense_improvement']:+.4f}")
        print(f"  Time: {results['time']:.2f}s")
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == '__main__':
    main()
