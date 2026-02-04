"""
VAE (MagNet-style) FGSM Evaluation Script for DermMel Dataset (ViT Classifier)

Reference:
"MagNet: a Two-Pronged Defense against Adversarial Examples"
Meng & Chen, ACM CCS 2017

MagNetの防御:
VAEで画像を再構成することで敵対的摂動を除去

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. FGSM敵対的画像の分類精度（防御なし）
4. FGSM敵対的画像を浄化した後の分類精度（防御あり）

実行例:
python vae_fgsm_eval.py --vae_ckpt /path/to/vae.pth --gpu 0
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


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='VAE (MagNet) FGSM Evaluation (ViT) - DermMel')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # VAE設定
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base channels')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/dermmel/vae/checkpoints/20260110_211546/best_model.pth',
                        help='VAE checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vae/fgsm/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Residual Block ==========
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
        """
        x_rgb = x_rgb.to(self.device)
        x_rgb = torch.clamp(x_rgb, 0, 1)
        
        with torch.no_grad():
            x_purified = self.vae.reconstruct(x_rgb)
        
        return x_purified


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
    """ViT分類器のラッパー"""
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 2クラスロジット"""
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== VAE + 分類器ラッパー ==========
class VAEClassifierWrapper(nn.Module):
    """VAE浄化 + 分類器"""
    def __init__(self, vae_purifier, classifier, mean, std):
        super().__init__()
        self.vae_purifier = vae_purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → VAE浄化 → 2クラスロジット"""
        x_purified = self.vae_purifier(x)
        x_norm = (x_purified - self.mean) / self.std
        return self.classifier(x_norm)


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM攻撃"""
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
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NotMelanoma', 'Melanoma'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  - Class distribution: {torch.bincount(y_test).tolist()}")
    print(f"  - Image shape: {x_test.shape}")
    print(f"  - Image range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ViT分類器を読み込み"""
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
    
    return classifier


def load_vae(args, device):
    """VAEを読み込み"""
    vae = VAE(img_channels=3, base_ch=args.base_ch, latent_dim=args.latent_dim)
    
    checkpoint = torch.load(args.vae_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['model_state_dict'])
    elif 'vae_state_dict' in checkpoint:
        vae.load_state_dict(checkpoint['vae_state_dict'])
    else:
        vae.load_state_dict(checkpoint)
    
    vae = vae.to(device).eval()
    print(f"Loaded VAE from {args.vae_ckpt}")
    
    return vae


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    """精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    all_preds = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = correct / len(x)
    return accuracy, np.array(all_preds)


# ========== メイン ==========
def main():
    args = parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("DermMel VAE (MagNet) FGSM Evaluation (ViT)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.1f}/255)")
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    vae = load_vae(args, device)
    
    # ラッパー作成
    clf_wrapper = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
    clf_wrapper.eval()
    
    vae_purifier = VAEPurifier(vae, device)
    vae_clf_wrapper = VAEClassifierWrapper(vae_purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
    vae_clf_wrapper.eval()
    
    # ========== 1. クリーン画像の精度 ==========
    print(f"\n[1/4] Evaluating clean images...")
    clean_acc, clean_preds = get_accuracy(clf_wrapper, x_test, y_test, args.batch_size, device)
    print(f"  Clean accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    
    # ========== 2. クリーン画像 + VAE浄化の精度 ==========
    print(f"\n[2/4] Evaluating clean images + VAE purification...")
    clean_vae_acc, clean_vae_preds = get_accuracy(vae_clf_wrapper, x_test, y_test, args.batch_size, device)
    print(f"  Clean + VAE accuracy: {clean_vae_acc:.4f} ({clean_vae_acc*100:.2f}%)")
    
    # ========== 3. FGSM敵対的画像の生成と評価 ==========
    print(f"\n[3/4] Generating FGSM adversarial examples (eps={args.epsilon:.5f})...")
    
    x_adv_list = []
    n_batches = (len(x_test) + args.batch_size - 1) // args.batch_size
    
    for i in tqdm(range(n_batches), desc="FGSM attack"):
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(x_test))
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        x_adv_batch = fgsm_attack(clf_wrapper, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    
    x_adv = torch.cat(x_adv_list, dim=0)
    
    # 敵対的画像の精度（防御なし）
    adv_acc, adv_preds = get_accuracy(clf_wrapper, x_adv, y_test, args.batch_size, device)
    print(f"  Adversarial accuracy (no defense): {adv_acc:.4f} ({adv_acc*100:.2f}%)")
    
    # ========== 4. FGSM敵対的画像 + VAE浄化の精度 ==========
    print(f"\n[4/4] Evaluating adversarial images + VAE purification...")
    adv_vae_acc, adv_vae_preds = get_accuracy(vae_clf_wrapper, x_adv, y_test, args.batch_size, device)
    print(f"  Adversarial + VAE accuracy: {adv_vae_acc:.4f} ({adv_vae_acc*100:.2f}%)")
    
    # ========== 結果サマリー ==========
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: DermMel")
    print(f"Model: ViT-B/16")
    print(f"Attack: FGSM (eps={args.epsilon:.5f})")
    print(f"Defense: VAE (MagNet-style)")
    print(f"-" * 60)
    print(f"{'Condition':<35} {'Accuracy':>10}")
    print(f"-" * 60)
    print(f"{'Clean':<35} {clean_acc:>10.4f}")
    print(f"{'Clean + VAE':<35} {clean_vae_acc:>10.4f}")
    print(f"{'FGSM (no defense)':<35} {adv_acc:>10.4f}")
    print(f"{'FGSM + VAE':<35} {adv_vae_acc:>10.4f}")
    print(f"{'='*60}")
    print(f"Defense improvement: {adv_vae_acc - adv_acc:+.4f} ({(adv_vae_acc - adv_acc)*100:+.2f}%)")
    print(f"Clean accuracy drop: {clean_vae_acc - clean_acc:+.4f} ({(clean_vae_acc - clean_acc)*100:+.2f}%)")
    
    # 結果保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'dataset': 'dermmel',
        'model': 'ViT-B/16',
        'attack': 'FGSM',
        'defense': 'VAE (MagNet)',
        'epsilon': args.epsilon,
        'clean_acc': clean_acc,
        'clean_vae_acc': clean_vae_acc,
        'adv_acc': adv_acc,
        'adv_vae_acc': adv_vae_acc,
        'defense_improvement': adv_vae_acc - adv_acc,
        'clean_acc_drop': clean_vae_acc - clean_acc,
        'n_samples': len(x_test),
        'timestamp': timestamp
    }
    
    result_path = os.path.join(args.output_dir, f'fgsm_vae_results_{timestamp}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")
    
    # 可視化サンプル保存
    n_vis = min(8, len(x_test))
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # VAE浄化画像を生成
    with torch.no_grad():
        x_vae = vae.reconstruct(x_adv[:n_vis].to(device)).cpu()
    
    comparison = torch.cat([
        x_test[:n_vis],      # Clean
        x_adv[:n_vis],       # Adversarial
        x_vae                # Adversarial + VAE
    ], dim=0)
    
    grid = make_grid(comparison, nrow=n_vis, normalize=False, padding=2)
    save_path = os.path.join(vis_dir, f'comparison_fgsm_{timestamp}.png')
    save_image(grid, save_path)
    print(f"Visualization saved to: {save_path}")


if __name__ == '__main__':
    main()
