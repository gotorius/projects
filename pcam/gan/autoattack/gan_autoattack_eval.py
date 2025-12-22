"""
Defense-GAN Adversarial Defense Evaluation for PCam Dataset - AutoAttack

Defense-GANは、GANの生成器を使って入力画像を「浄化」し、
敵対的摂動を除去する防御手法です。

浄化プロセス:
1. 入力画像 x に対して、最適な潜在変数 z* を勾配降下法で探索
2. z* から生成された画像 G(z*) を分類器への入力として使用

AutoAttack: 複数の強力な攻撃手法を組み合わせた信頼性の高い評価

実行例:
python gan_autoattack_eval.py --epsilon 0.031 --use_defense
python gan_autoattack_eval.py --epsilon 0.031 --aa_version standard
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
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm
import torchvision.transforms as transforms


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation - AutoAttack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon')
    parser.add_argument('--aa_version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    parser.add_argument('--aa_norm', type=str, default='Linf',
                        choices=['Linf', 'L2'],
                        help='AutoAttack norm')
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true',
                        help='Enable Defense-GAN purification')
    parser.add_argument('--rec_iters', type=int, default=500,
                        help='Number of reconstruction iterations')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=10,
                        help='Number of random restarts')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints/best_model.pth',
                        help='GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/autoattack/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Generator (DCGAN-based) ==========
class Generator(nn.Module):
    """
    DCGAN-based Generator for 224x224 images
    latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
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
    Defense-GAN: 敵対的画像をGANの生成器を使って浄化
    
    浄化は、入力画像に最も近い画像を生成する潜在変数z*を探索することで行う。
    z* = argmin_z ||G(z) - x||^2
    """
    def __init__(self, generator, latent_dim=128, rec_iters=500, rec_lr=0.01, 
                 rec_rr=10, device='cuda'):
        self.generator = generator
        self.generator.eval()
        self.latent_dim = latent_dim
        self.rec_iters = rec_iters
        self.rec_lr = rec_lr
        self.rec_rr = rec_rr  # random restarts
        self.device = device
        
    def reconstruct(self, x):
        """
        入力画像xを生成器で再構成
        x: (B, 3, H, W), [0, 1]の範囲
        return: 再構成画像 (B, 3, H, W), [0, 1]の範囲
        """
        batch_size = x.size(0)
        
        # 入力を[-1, 1]に変換（GANの出力範囲）
        x_target = x * 2.0 - 1.0
        
        best_z = None
        best_loss = float('inf') * torch.ones(batch_size, device=self.device)
        
        # Multiple random restarts
        for r in range(self.rec_rr):
            # Initialize z randomly
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                
                # Generate image
                x_gen = self.generator(z)
                
                # Reconstruction loss (per sample)
                loss = F.mse_loss(x_gen, x_target, reduction='none')
                loss = loss.view(batch_size, -1).mean(dim=1)
                
                # Backward
                total_loss = loss.sum()
                total_loss.backward()
                optimizer.step()
            
            # Update best z for each sample
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                # Update if better
                better_mask = final_loss < best_loss
                if better_mask.any():
                    if best_z is None:
                        best_z = z.clone()
                    else:
                        best_z[better_mask] = z[better_mask]
                    best_loss[better_mask] = final_loss[better_mask]
        
        # Generate final reconstruction
        with torch.no_grad():
            x_rec = self.generator(best_z)
            # Convert back to [0, 1]
            x_rec = (x_rec + 1.0) / 2.0
            x_rec = torch.clamp(x_rec, 0, 1)
        
        return x_rec


# ========== Normalized Classifier Wrapper ==========
class NormalizedClassifier(nn.Module):
    """AutoAttack用に正規化を内包した分類器ラッパー"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """分類器を読み込み"""
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Fix state_dict key mismatch: fc.1 -> fc (if needed)
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('fc.1.'):
            new_k = k.replace('fc.1.', 'fc.')
            fixed_state_dict[new_k] = v
        elif k.startswith('fc.0.') or k == 'fc.weight' or k == 'fc.bias':
            fixed_state_dict[k] = v
        else:
            fixed_state_dict[k] = v
    
    # Try loading with Sequential fc structure first
    try:
        classifier.load_state_dict(fixed_state_dict)
    except RuntimeError:
        # If failed, try with simple Linear fc
        classifier.fc = nn.Linear(num_features, num_classes)
        # Need to remap fc.1 keys
        simple_state_dict = {}
        for k, v in state_dict.items():
            if k == 'fc.1.weight':
                simple_state_dict['fc.weight'] = v
            elif k == 'fc.1.bias':
                simple_state_dict['fc.bias'] = v
            elif not k.startswith('fc.'):
                simple_state_dict[k] = v
            else:
                simple_state_dict[k] = v
        classifier.load_state_dict(simple_state_dict)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    return classifier


def load_generator(args, device):
    """GAN生成器を読み込み"""
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    # Get model parameters from checkpoint
    if 'args' in checkpoint:
        model_args = checkpoint['args']
        latent_dim = model_args.get('latent_dim', 128)
        ngf = model_args.get('ngf', 64)
    else:
        latent_dim = 128
        ngf = 64
    
    generator = Generator(latent_dim=latent_dim, ngf=ngf).to(device)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    elif 'model_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    print(f"Loaded generator from {args.gan_ckpt}")
    print(f"Generator config: latent_dim={latent_dim}, ngf={ngf}")
    
    return generator, latent_dim


# ========== データ読み込み ==========
def load_cached_samples(path):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f"Loaded {len(x_test)} samples from {path}")
    print(f"Classes: {classes}")
    return x_test, y_test, classes


# ========== AutoAttack ==========
def autoattack_eval(model, x, y, epsilon, norm='Linf', version='standard', device='cuda'):
    """AutoAttack評価"""
    try:
        from autoattack import AutoAttack
    except ImportError:
        print("AutoAttack not installed. Install with: pip install git+https://github.com/fra31/auto-attack")
        return x
    
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, device=device)
    adversary.seed = 42  # 再現性のため
    
    x_adv = adversary.run_standard_evaluation(x, y, bs=min(x.shape[0], 64))
    
    return x_adv


# ========== 評価関数 ==========
def evaluate(model, x_test, y_test, device, batch_size=16):
    """精度を計算"""
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


def evaluate_normalized(model, x_test, y_test, device, batch_size=16):
    """正規化が内包されたモデルで精度を計算"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            outputs = model(x_batch)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    return correct / total, np.array(predictions)


def evaluate_with_defense(defense_gan, classifier, x_test, y_test, device, batch_size=4, desc="Defense-GAN"):
    """Defense-GAN適用後の精度を計算"""
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_purified_all = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    # Defense-GANは計算コストが高いため、小さいバッチで処理
    for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
        x_batch = x_test[i:i+batch_size].to(device)
        y_batch = y_test[i:i+batch_size].to(device)
        
        # Defense-GAN purification
        x_purified = defense_gan.reconstruct(x_batch)
        x_purified_all.append(x_purified.cpu())
        
        # 分類
        with torch.no_grad():
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
        
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        predictions.extend(predicted.cpu().numpy())
    
    x_purified_all = torch.cat(x_purified_all, dim=0)
    return correct / total, np.array(predictions), x_purified_all


# ========== ユーティリティ ==========
def compute_l2_norm(x1, x2):
    """L2ノルムを計算"""
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def print_confusion_matrix(y_true, y_pred, title, classes, file=None):
    """混同行列を出力"""
    cm = confusion_matrix(y_true, y_pred)
    
    # メトリクス計算
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
    
    return {
        'cm': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, labels, classes, save_dir):
    """サンプル画像を保存"""
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
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # デバイス
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ (MMDDHHMM形式)
    timestamp = datetime.now().strftime("%m%d%H%M")
    defense_str = "defense" if args.use_defense else "no_defense"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # 結果ファイル
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    
    # GAN生成器読み込み
    generator, latent_dim = load_generator(args, device)
    
    # Defense-GAN
    defense_gan = None
    if args.use_defense:
        defense_gan = DefenseGAN(
            generator=generator,
            latent_dim=latent_dim,
            rec_iters=args.rec_iters,
            rec_lr=args.rec_lr,
            rec_rr=args.rec_rr,
            device=device
        )
        print(f"Defense-GAN enabled: iters={args.rec_iters}, lr={args.rec_lr}, rr={args.rec_rr}")
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("AutoAttack + Defense-GAN Evaluation")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: AutoAttack")
    write_and_print(f"  - Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"  - Norm: {args.aa_norm}")
    write_and_print(f"  - Version: {args.aa_version}")
    write_and_print(f"Defense: Defense-GAN (enabled={args.use_defense})")
    if args.use_defense:
        write_and_print(f"  - Reconstruction iters: {args.rec_iters}")
        write_and_print(f"  - Reconstruction lr: {args.rec_lr}")
        write_and_print(f"  - Random restarts: {args.rec_rr}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    # 1. クリーン画像の評価
    write_and_print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # 2. クリーン画像 + Defense-GAN
    x_purified_clean = None
    if args.use_defense:
        write_and_print("\n[2/4] Evaluating clean images with Defense-GAN...")
        clean_defense_acc, pred_clean_defense, x_purified_clean = evaluate_with_defense(
            defense_gan, classifier, x_test, y_test, device, batch_size=args.batch_size, desc="Purifying clean images"
        )
        l2_clean_purified = compute_l2_norm(x_test, x_purified_clean)
        write_and_print(f"Clean accuracy (with Defense-GAN): {clean_defense_acc:.4f}")
        write_and_print(f"L2 norm (clean vs purified): {l2_clean_purified:.4f}")
        results['clean_acc_with_defense'] = clean_defense_acc
        results['l2_clean_vs_purified'] = l2_clean_purified
    else:
        write_and_print("\n[2/4] Skipping Defense-GAN evaluation (defense disabled)")
    
    # 3. AutoAttack
    write_and_print(f"\n[3/4] Running AutoAttack (eps={args.epsilon:.4f}, norm={args.aa_norm}, version={args.aa_version})...")
    
    # AutoAttack用に正規化を内包した分類器を作成
    normalized_classifier = NormalizedClassifier(classifier).to(device)
    normalized_classifier.eval()
    
    start_time = time.time()
    x_adv = autoattack_eval(
        normalized_classifier, 
        x_test.to(device), 
        y_test.to(device),
        args.epsilon, 
        norm=args.aa_norm, 
        version=args.aa_version,
        device=device
    )
    x_adv = x_adv.cpu()
    attack_time = time.time() - start_time
    
    l2_clean_adv = compute_l2_norm(x_test, x_adv)
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    write_and_print(f"L2 norm (clean vs adversarial): {l2_clean_adv:.4f}")
    write_and_print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_clean_adv
    results['attack_time'] = attack_time
    
    # 4. 敵対的画像 + Defense-GAN
    x_purified_adv = None
    if args.use_defense:
        write_and_print("\n[4/4] Evaluating adversarial images with Defense-GAN...")
        adv_defense_acc, pred_adv_defense, x_purified_adv = evaluate_with_defense(
            defense_gan, classifier, x_adv, y_test, device, batch_size=args.batch_size, desc="Purifying adversarial images"
        )
        l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
        write_and_print(f"Adversarial accuracy (with Defense-GAN): {adv_defense_acc:.4f}")
        write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
        results['adv_acc_with_defense'] = adv_defense_acc
        results['l2_adv_vs_purified'] = l2_adv_purified
        results['defense_improvement'] = adv_defense_acc - adv_acc
    else:
        write_and_print("\n[4/4] Skipping Defense-GAN evaluation (defense disabled)")
    
    # 最終結果
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: AutoAttack (eps={args.epsilon:.4f}, norm={args.aa_norm}, version={args.aa_version})")
    write_and_print(f"Defense: Defense-GAN (enabled={args.use_defense})")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    if args.use_defense:
        write_and_print(f"  With Defense-GAN:            {results['clean_acc_with_defense']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (AutoAttack):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    if args.use_defense:
        write_and_print(f"  With Defense-GAN:            {results['adv_acc_with_defense']:.4f}")
        write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    if args.use_defense:
        write_and_print(f"  Clean vs Purified:           {results['l2_clean_vs_purified']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    if args.use_defense:
        write_and_print(f"  Adversarial vs Purified:     {results['l2_adv_vs_purified']:.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Attack time: {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    # 混同行列
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes, results_file)
    if args.use_defense:
        cm_results['clean_defense'] = print_confusion_matrix(y_true, pred_clean_defense, "2. Clean Images (with Defense-GAN)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    if args.use_defense:
        cm_results['adv_defense'] = print_confusion_matrix(y_true, pred_adv_defense, "4. Adversarial Images (with Defense-GAN)", classes, results_file)
    
    # サンプル画像保存
    if args.use_defense and x_purified_clean is not None and x_purified_adv is not None:
        write_and_print("\nSaving sample images...")
        samples_dir = os.path.join(log_dir, 'samples')
        save_sample_images(x_test[:10], x_adv[:10], x_purified_clean[:10], x_purified_adv[:10],
                           y_test[:10], classes, samples_dir)
    
    results_file.close()
    
    # JSON形式でも保存
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
