"""
ChestX-ray Dataset - AutoAttack + VAE (MagNet) Defense (ViT Classifier)
AutoAttackによる強力な敵対的攻撃に対するVAE防御の検証

AutoAttack:
- APGD-CE: Auto-PGD with cross-entropy loss
- APGD-DLR: Auto-PGD with difference of logits ratio loss  
- FAB: Fast Adaptive Boundary attack
- Square: Square attack (query-based)

Reference:
"MagNet: a Two-Pronged Defense against Adversarial Examples"
Meng & Chen, ACM CCS 2017

実行例:
python vae_autoattack_eval.py --epsilon 0.031 --gpu 0
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
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

# AutoAttackのインポート
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not found. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='VAE (MagNet) AutoAttack Evaluation (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Attack norm')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # VAE設定
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--base_ch', type=int, default=48,
                        help='Base channels (v3: 48)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/chestxray/vae/checkpoints_v3/20260105_175040/best_model.pth',
                        help='VAE checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/vae/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Residual Block (v3アーキテクチャ用) ==========
class ResBlockEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # skip層はチャネル数が変わる場合のみ
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = None
    
    def forward(self, x):
        h = F.leaky_relu(self.bn1(x), 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(self.bn2(h), 0.2)
        h = self.conv2(h)
        if self.skip is not None:
            x = self.skip(x)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        return h + x


class ResBlockDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # skip層はチャネル数が変わる場合のみ
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = None
    
    def forward(self, x):
        h = F.relu(self.bn1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if self.skip is not None:
            x = self.skip(x)
        return h + x


# ========== VAE Model ==========
class Encoder(nn.Module):
    def __init__(self, img_channels=1, base_ch=48, latent_dim=256):
        super().__init__()
        self.conv_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        # block1: 48->48, block2: 48->96, block3: 96->192, block4: 192->384, block5: 384->384
        self.block1 = ResBlockEncoder(base_ch, base_ch, downsample=True)
        self.block2 = ResBlockEncoder(base_ch, base_ch*2, downsample=True)
        self.block3 = ResBlockEncoder(base_ch*2, base_ch*4, downsample=True)
        self.block4 = ResBlockEncoder(base_ch*4, base_ch*8, downsample=True)
        self.block5 = ResBlockEncoder(base_ch*8, base_ch*8, downsample=True)
        self.bn_out = nn.BatchNorm2d(base_ch*8)
        self.fc_mu = nn.Linear(base_ch*8*7*7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch*8*7*7, latent_dim)
        self.base_ch = base_ch
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.leaky_relu(self.bn_out(h), 0.2)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, img_channels=1, base_ch=48, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch*8*7*7)
        # block1: 384->384, block2: 384->192, block3: 192->96, block4: 96->48, block5: 48->48
        self.block1 = ResBlockDecoder(base_ch*8, base_ch*8, upsample=True)
        self.block2 = ResBlockDecoder(base_ch*8, base_ch*4, upsample=True)
        self.block3 = ResBlockDecoder(base_ch*4, base_ch*2, upsample=True)
        self.block4 = ResBlockDecoder(base_ch*2, base_ch, upsample=True)
        self.block5 = ResBlockDecoder(base_ch, base_ch, upsample=True)
        self.bn_out = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
        self.base_ch = base_ch
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch*8, 7, 7)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.relu(self.bn_out(h))
        return torch.sigmoid(self.conv_out(h))


class VAE(nn.Module):
    def __init__(self, img_channels=1, base_ch=48, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(img_channels, base_ch, latent_dim)
        self.decoder = Decoder(img_channels, base_ch, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def reconstruct(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ========== VAE Purifier ==========
class VAEPurifier(nn.Module):
    """VAE浄化器（グレースケール用）"""
    def __init__(self, vae, device):
        super().__init__()
        self.vae = vae
        self.device = device
    
    def _rgb_to_gray(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    def _gray_to_rgb(self, x):
        return x.repeat(1, 3, 1, 1)
    
    def purify(self, x):
        x_gray = self._rgb_to_gray(x)
        x_rec = self.vae.reconstruct(x_gray)
        x_rec = x_rec.clamp(0, 1)
        return self._gray_to_rgb(x_rec)


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


# ========== VAE防御付き分類器ラッパー ==========
class VAEDefendedClassifier(nn.Module):
    """VAE防御 + ViT分類器のラッパー（AutoAttack用）"""
    def __init__(self, classifier_wrapper, purifier):
        super().__init__()
        self.classifier = classifier_wrapper
        self.purifier = purifier
    
    def forward(self, x):
        x_purified = self.purifier.purify(x)
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
    
    # VAEモデル
    vae = VAE(img_channels=1, base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    if 'vae_state_dict' in ckpt:
        vae.load_state_dict(ckpt['vae_state_dict'])
        print(f"Loaded VAE (vae_state_dict) from {args.vae_ckpt}")
    elif 'model_state_dict' in ckpt:
        vae.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded VAE (model_state_dict) from {args.vae_ckpt}")
    else:
        try:
            vae.load_state_dict(ckpt, strict=False)
            print(f"Loaded VAE from {args.vae_ckpt}")
        except Exception as e:
            print(f"Warning: {e}")
    vae.eval()
    print(f"VAE config: latent_dim={args.latent_dim}, base_ch={args.base_ch}")
    
    return classifier, vae


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
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
    classifier, vae = load_models(args, device)
    purifier = VAEPurifier(vae, device).to(device)
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # VAE防御付き分類器
    defended_model = VAEDefendedClassifier(classifier_model, purifier).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + VAE (MagNet) Defense Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"Version: {args.version}")
    print(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with VAE purification...")
    clean_purified_acc = get_accuracy(defended_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with VAE): {clean_purified_acc:.4f}")
    results['clean_purified_acc'] = clean_purified_acc
    
    # ========== 3. AutoAttack ==========
    print("\n[3/4] Running AutoAttack...")
    start_time = time.time()
    
    adversary = AutoAttack(classifier_model, norm=args.norm, eps=args.epsilon, version=args.version, device=device)
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
    
    attack_time = time.time() - start_time
    print(f"AutoAttack completed in {attack_time:.2f}s")
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度 ==========
    print("\n[4/4] Evaluating adversarial images with VAE purification...")
    adv_defended_acc = get_accuracy(defended_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with VAE): {adv_defended_acc:.4f}")
    results['adv_defended_acc'] = adv_defended_acc
    
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f}, Norm: {args.norm}")
    print(f"Defense: VAE (MagNet)")
    print(f"-"*70)
    print(f"Clean accuracy:              {results['clean_acc']:.4f}")
    print(f"Clean accuracy (with VAE):   {results['clean_purified_acc']:.4f}")
    print(f"Adversarial (no defense):    {results['adv_acc_no_defense']:.4f}")
    print(f"Adversarial (with VAE):      {results['adv_defended_acc']:.4f}")
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
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "3. AutoAttack Images (with VAE)", classes)
    
    # ==================== サンプル画像保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified = []
    for i in range(n_samples):
        x_pur = purifier.purify(x_adv[i:i+1].to(device))
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
        f.write("ChestX-ray - AutoAttack + VAE (MagNet) Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: AutoAttack ({args.version})\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Norm: {args.norm}\n")
        f.write(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        f.write(f"Clean accuracy:              {results['clean_acc']:.4f}\n")
        f.write(f"Clean accuracy (with VAE):   {results['clean_purified_acc']:.4f}\n")
        f.write(f"Adversarial (no defense):    {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"Adversarial (with VAE):      {results['adv_defended_acc']:.4f}\n")
        f.write(f"Defense improvement:         {results['defense_improvement']:+.4f}\n\n")
        f.write(f"Attack time: {results['attack_time']:.2f}s\n")
    
    # JSON保存
    results_json = {
        'classifier': 'ViT-B/16',
        'attack': 'autoattack',
        'version': args.version,
        'norm': args.norm,
        'epsilon': args.epsilon,
        'vae_latent_dim': args.latent_dim,
        'vae_base_ch': args.base_ch,
        **results
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")
    
    return results


if __name__ == '__main__':
    main()
