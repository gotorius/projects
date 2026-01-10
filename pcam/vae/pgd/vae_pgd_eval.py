"""
VAE (MagNet-style) PGD Evaluation Script for PCam (PatchCamelyon) Dataset

Reference:
"MagNet: a Two-Pronged Defense against Adversarial Examples"
Meng & Chen, ACM CCS 2017

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. PGD敵対的画像の分類精度（防御なし）
4. PGD敵対的画像を浄化した後の分類精度（防御あり）

実行例:
python vae_pgd_eval.py --vae_ckpt ../checkpoints_v2/20260108_165711/best_model.pth --gpu 0
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='VAE (MagNet) PGD Evaluation for PCam')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD perturbation epsilon')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD steps')
    
    # VAE設定
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension (v2: 512)')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base channels (v2: 64)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached samples')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/vae/checkpoints_v2/20260108_165711/best_model.pth',
                        help='VAE checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/vae/pgd/results',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder',
                        help='Data directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Residual Block (v2アーキテクチャ用) ==========
class ResBlockEncoder(nn.Module):
    """Residual Block for Encoder"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        h = F.leaky_relu(self.bn1(x), 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(self.bn2(h), 0.2)
        h = self.conv2(h)
        
        x = self.skip(x)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        
        return h + x


class ResBlockDecoder(nn.Module):
    """Residual Block for Decoder"""
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        h = F.relu(self.bn1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.skip(x)
        
        return h + x


# ========== Encoder (RGB入力) ==========
class Encoder(nn.Module):
    """Encoder for RGB images: 224 -> 7 -> latent"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        
        self.conv_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.block1 = ResBlockEncoder(base_ch, base_ch, downsample=True)
        self.block2 = ResBlockEncoder(base_ch, base_ch * 2, downsample=True)
        self.block3 = ResBlockEncoder(base_ch * 2, base_ch * 4, downsample=True)
        self.block4 = ResBlockEncoder(base_ch * 4, base_ch * 8, downsample=True)
        self.block5 = ResBlockEncoder(base_ch * 8, base_ch * 8, downsample=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch * 8)
        
        # Latent projections
        self.fc_mu = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.leaky_relu(self.bn_out(h), 0.2)
        
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


# ========== Decoder (RGB出力) ==========
class Decoder(nn.Module):
    """Decoder for RGB images: latent -> 7 -> 224"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.base_ch = base_ch
        
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 7 * 7)
        
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.block1 = ResBlockDecoder(base_ch * 8, base_ch * 8, upsample=True)
        self.block2 = ResBlockDecoder(base_ch * 8, base_ch * 4, upsample=True)
        self.block3 = ResBlockDecoder(base_ch * 4, base_ch * 2, upsample=True)
        self.block4 = ResBlockDecoder(base_ch * 2, base_ch, upsample=True)
        self.block5 = ResBlockDecoder(base_ch, base_ch, upsample=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, 7, 7)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        
        h = F.relu(self.bn_out(h))
        h = self.conv_out(h)
        
        return torch.sigmoid(h)


# ========== VAE ==========
class VAE(nn.Module):
    """VAE for RGB images"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(img_channels, base_ch, latent_dim)
        self.decoder = Decoder(img_channels, base_ch, latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def reconstruct(self, x):
        """再構成のみ（MagNet防御用）- 決定論的"""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ========== VAE Purifier (MagNet-style) ==========
class VAEPurifier(nn.Module):
    """VAEで画像を再構成して敵対的摂動を除去（RGB直接処理）"""
    def __init__(self, vae, device):
        super().__init__()
        self.vae = vae
        self.device = device
    
    def forward(self, x_rgb):
        """
        RGB画像 [0,1] を浄化
        x_rgb: (B, 3, H, W), [0, 1]
        return: 浄化されたRGB画像 (B, 3, H, W), [0, 1]
        """
        self.vae.eval()
        with torch.no_grad():
            x_recon = self.vae.reconstruct(x_rgb)
        return x_recon


# ========== 分類器ラッパー ==========
class ClassifierWrapper(nn.Module):
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        return self.classifier(x_norm)


# ========== VAE Defense Wrapper ==========
class VAEDefenseWrapper(nn.Module):
    """VAE浄化 + 分類器のラッパー"""
    def __init__(self, purifier, classifier, mean, std):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_purified = self.purifier(x)
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x_purified - mean) / std
        return self.classifier(x_norm)


# ========== モデル読み込み ==========
def load_models(args, device):
    # 分類器
    classifier = models.resnet50(weights=None)
    # チェックポイントがnn.Sequential形式（fc.1.weight）で保存されている場合に対応
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(classifier.fc.in_features, 2)
    )
    checkpoint = torch.load(args.clf_ckpt, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    # VAE
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    if 'args' in vae_ckpt:
        latent_dim = vae_ckpt['args'].get('latent_dim', args.latent_dim)
        base_ch = vae_ckpt['args'].get('base_ch', args.base_ch)
    else:
        latent_dim = args.latent_dim
        base_ch = args.base_ch
    
    print(f"VAE config: latent_dim={latent_dim}, base_ch={base_ch}")
    
    # PCam用VAEはRGB (img_channels=3)
    vae = VAE(img_channels=3, base_ch=base_ch, latent_dim=latent_dim).to(device)
    
    # キー名の互換性対応
    if 'vae_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['vae_state_dict'])
    elif 'ema_state_dict' in vae_ckpt and vae_ckpt['ema_state_dict'] is not None:
        vae.load_state_dict(vae_ckpt['ema_state_dict'])
        print("Using EMA weights")
    elif 'model_state_dict' in vae_ckpt:
        vae.load_state_dict(vae_ckpt['model_state_dict'])
    else:
        vae.load_state_dict(vae_ckpt)
    
    vae.eval()
    print(f"Loaded VAE from {args.vae_ckpt}")
    
    return classifier, vae


# ========== データ読み込み ==========
def load_cached_samples(path, device):
    data = torch.load(path, map_location='cpu', weights_only=False)
    
    # キー名の互換性対応
    if 'images' in data:
        x_test = data['images']
    elif 'x_test' in data:
        x_test = data['x_test']
    elif 'x' in data:
        x_test = data['x']
    else:
        raise KeyError(f"Cannot find image data. Available keys: {data.keys()}")
    
    if 'labels' in data:
        y_test = data['labels']
    elif 'y_test' in data:
        y_test = data['y_test']
    elif 'y' in data:
        y_test = data['y']
    else:
        raise KeyError(f"Cannot find label data. Available keys: {data.keys()}")
    
    print(f"Loaded {len(x_test)} cached samples from {path}")
    print(f"Data shape: {x_test.shape}, Label shape: {y_test.shape}")
    
    return x_test.to(device), y_test.to(device)


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, alpha, steps, device):
    """PGD攻撃"""
    x = x.clone().to(device)
    y = y.to(device)
    
    # ランダム初期化
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta = torch.clamp(x + delta, 0, 1) - x
    delta.requires_grad = True
    
    for _ in range(steps):
        outputs = model(x + delta)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        
        # 勾配ステップ
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = torch.clamp(x + delta.data, 0, 1) - x
        delta.grad.zero_()
    
    x_adv = x + delta.detach()
    return x_adv


# ========== 評価関数 ==========
def get_accuracy(model, x_test, y_test, bs=32, device='cuda'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(x_test), bs):
            x_batch = x_test[i:i+bs].to(device)
            y_batch = y_test[i:i+bs].to(device)
            outputs = model(x_batch)
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


def get_predictions(model, x_test, bs=32, device='cuda'):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(x_test), bs):
            x_batch = x_test[i:i+bs].to(device)
            outputs = model(x_batch)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return np.array(predictions)


def print_confusion_matrix(y_true, y_pred, title, classes):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print("-" * 50)
    print(f"{'':>15} {'Pred ' + classes[0]:>15} {'Pred ' + classes[1]:>15}")
    print(f"{'True ' + classes[0]:>15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'True ' + classes[1]:>15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv,
                       y_true, classes, save_dir, max_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"pgd_eps{args.epsilon:.4f}_steps{args.pgd_steps}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    classifier, vae = load_models(args, device)
    
    purifier = VAEPurifier(vae, device).to(device)
    
    classifier_model = ClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = VAEDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # PCamのクラス名
    classes = ['0', '1']  # 0: 転移なし, 1: 転移あり
    print(f"Classes: {classes}")
    
    x_test, y_test = load_cached_samples(args.cached_samples, device)
    
    print(f"\n{'='*70}")
    print("PGD Attack + VAE (MagNet) Defense Evaluation for PCam")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255)")
    print(f"Steps: {args.pgd_steps}")
    print(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # 1. クリーン画像の精度
    print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # 2. クリーン画像を浄化した後の精度
    print("\n[2/4] Evaluating clean images with VAE purification...")
    clean_purified_acc = get_accuracy(defense_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with VAE): {clean_purified_acc:.4f}")
    results['clean_acc_with_vae'] = clean_purified_acc
    
    # 3. PGD攻撃 & 敵対的画像の精度（防御なし）
    print("\n[3/4] Running PGD attack and evaluating adversarial images...")
    start_time = time.time()
    
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="PGD Attack"):
        x_batch = x_test[i:i+args.batch_size].to(device)
        y_batch = y_test[i:i+args.batch_size].to(device)
        x_adv_batch = pgd_attack(classifier_model, x_batch, y_batch, 
                                  args.epsilon, args.alpha, args.pgd_steps, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # 4. 敵対的画像を浄化した後の精度（防御あり）
    print("\n[4/4] Evaluating adversarial images with VAE purification...")
    adv_defended_acc = get_accuracy(defense_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with VAE): {adv_defended_acc:.4f}")
    results['adv_acc_with_vae'] = adv_defended_acc
    
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # 最終結果
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"        Alpha: {args.alpha:.4f}, Steps: {args.pgd_steps}")
    print(f"Defense: VAE (MagNet-style)")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:             {results['clean_acc_classifier']:.4f}")
    print(f"  With VAE purification:       {results['clean_acc_with_vae']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (PGD):")
    print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    print(f"  With VAE purification:       {results['adv_acc_with_vae']:.4f}")
    print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # 混同行列
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_clean_purified = get_predictions(defense_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions(defense_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (Classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with VAE)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with VAE)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # サンプル画像保存
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified_clean = []
    x_purified_adv = []
    
    with torch.no_grad():
        for i in range(n_samples):
            x_purified_clean.append(purifier(x_test[i:i+1].to(device)).cpu())
            x_purified_adv.append(purifier(x_adv[i:i+1].to(device)).cpu())
    
    x_purified_clean = torch.cat(x_purified_clean, dim=0)
    x_purified_adv = torch.cat(x_purified_adv, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(),
        x_adv[:n_samples].cpu(),
        x_purified_clean,
        x_purified_adv,
        y_test[:n_samples],
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # 結果保存（テキスト形式）
    with open(os.path.join(log_dir, 'results.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("PGD Attack + VAE (MagNet) Defense Evaluation Results for PCam\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"        Alpha: {args.alpha:.4f}, Steps: {args.pgd_steps}\n")
        f.write(f"Defense: VAE (MagNet-style)\n")
        f.write(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}\n")
        f.write(f"Samples: {len(x_test)}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Clean Accuracy:\n")
        f.write("-"*70 + "\n")
        f.write(f"  Classifier only:             {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With VAE purification:       {results['clean_acc_with_vae']:.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("Adversarial Accuracy (PGD):\n")
        f.write("-"*70 + "\n")
        f.write(f"  Without defense:             {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With VAE purification:       {results['adv_acc_with_vae']:.4f}\n")
        f.write(f"  Defense improvement:         {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("="*70 + "\n")
        f.write("Confusion Matrices\n")
        f.write("="*70 + "\n\n")
        
        cm_names = [
            "1. Clean Images (Classifier only)",
            "2. Clean Images (with VAE)",
            "3. Adversarial Images (No Defense)",
            "4. Adversarial Images (with VAE)"
        ]
        cm_keys = ['clean', 'clean_purified', 'adv_no_defense', 'adv_defended']
        
        for name, key in zip(cm_names, cm_keys):
            cm = results['confusion_matrices'][key]
            f.write(f"{name}\n")
            f.write("-"*50 + "\n")
            f.write(f"{'':>15} {'Pred 0':>15} {'Pred 1':>15}\n")
            f.write(f"{'True 0':>15} {cm['tn']:>15} {cm['fp']:>15}\n")
            f.write(f"{'True 1':>15} {cm['fn']:>15} {cm['tp']:>15}\n\n")
            
            f.write(f"  Accuracy:  {cm['accuracy']:.4f}\n")
            f.write(f"  Precision: {cm['precision']:.4f}\n")
            f.write(f"  Recall:    {cm['recall']:.4f}\n")
            f.write(f"  F1-score:  {cm['f1']:.4f}\n\n")
    
    # 設定保存（テキスト形式）
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("Configuration\n")
        f.write("="*70 + "\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    # JSON形式でも保存
    results_json = {k: v for k, v in results.items()}
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
