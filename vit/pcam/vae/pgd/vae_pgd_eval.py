"""
PCam Dataset - VAE (MagNet-style) PGD Evaluation (ViT Classifier)

VAEによるMagNet-style防御に対してPGD攻撃を評価するスクリプト。
PCamデータセット用のViT分類器を使用。

Reference:
    "MagNet: a Two-Pronged Defense against Adversarial Examples"
    Meng & Chen, CCS 2017
    https://arxiv.org/abs/1705.09064

Usage:
    python vae_pgd_eval.py --epsilon 0.031 --alpha 0.007 --pgd_steps 10 --gpu 0
    python vae_pgd_eval.py --epsilon 0.062 --alpha 0.015 --pgd_steps 20 --gpu 0
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
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='VAE (MagNet-style) + PGD Attack Evaluation for PCam (ViT Classifier)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 攻撃パラメータ
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD攻撃の最大摂動量 (0-1スケール)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD攻撃のステップサイズ (0-1スケール)')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='PGD攻撃のステップ数')
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='PGD攻撃でランダムスタートを使用')
    
    # VAEパラメータ
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='VAEの潜在次元')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='VAEのベースチャンネル数')
    
    # 実行パラメータ
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--seed', type=int, default=42,
                        help='乱数シード')
    
    # パス
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt',
                        help='キャッシュされたサンプルファイル')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/pcam/vae/checkpoints_v2/20260108_165711/best_model.pth',
                        help='VAEチェックポイントファイル')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth',
                        help='ViT分類器チェックポイントファイル')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/vae/pgd/results',
                        help='結果出力ディレクトリ')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用するGPU番号')
    
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
        x_rgb: (B, 3, H, W), [0, 1]
        return: 浄化されたRGB画像 (B, 3, H, W), [0, 1]
        """
        self.vae.eval()
        with torch.no_grad():
            x_recon = self.vae.reconstruct(x_rgb)
        return x_recon


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
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
    """VAE浄化 + ViT分類器のラッパー"""
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


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, alpha, steps, device, random_start=True):
    """
    PGD攻撃
    
    Args:
        model: 分類器（入力は[0,1]のRGB画像）
        x: 入力画像 [B, 3, H, W] in [0, 1]
        y: ラベル [B]
        epsilon: 最大摂動量（ピクセルスケール 0-1）
        alpha: ステップサイズ（ピクセルスケール 0-1）
        steps: PGDステップ数
        device: デバイス
        random_start: ランダム初期化を使用するか
    
    Returns:
        x_adv: 敵対的画像 [B, 3, H, W] in [0, 1]
    """
    x_orig = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    
    # ランダムスタート
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_orig.clone()
    
    # PGDイテレーション
    for _ in range(steps):
        x_adv.requires_grad = True
        
        # Forward pass
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        
        # Update: x_adv = x_adv + alpha * sign(grad)
        x_adv = x_adv.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball
        perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = torch.clamp(x_orig + perturbation, 0.0, 1.0).detach()
    
    return x_adv


# ========== モデル読み込み ==========
def load_models(args, device):
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
    
    # VAE
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device)
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
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み（ViT分類器で正しく分類されたサンプル）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['normal', 'tumor'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    """モデルの精度を計算"""
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


# ========== 予測取得 ==========
def get_predictions(model, x, bs=32, device=None):
    """モデルの予測を取得"""
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
    """混同行列をテキスト出力"""
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


# ========== PGD攻撃実行 ==========
def run_pgd_attack(model, x_test, y_test, epsilon, alpha, steps, device, batch_size, random_start=True):
    """PGD攻撃を実行して敵対的サンプルを生成"""
    print(f"\nRunning PGD attack with epsilon={epsilon:.4f}, alpha={alpha:.4f}, steps={steps}...")
    
    n_batches = (len(x_test) + batch_size - 1) // batch_size
    x_adv_list = []
    
    for i in tqdm(range(n_batches), desc="PGD Attack"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        x_batch = x_test[start_idx:end_idx].to(device)
        y_batch = y_test[start_idx:end_idx].to(device)
        
        x_adv_batch = pgd_attack(model, x_batch, y_batch, epsilon, alpha, steps, device, random_start)
        x_adv_list.append(x_adv_batch.cpu())
    
    x_adv = torch.cat(x_adv_list, dim=0)
    print(f"Generated {len(x_adv)} adversarial samples")
    
    return x_adv


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, 
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # 4枚を並べて保存: Clean, Clean+VAE, Adv, Adv+VAE
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Clean+VAE | Adversarial | Adv+VAE]")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing GPU: {args.gpu}")
    print(f"Device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"pgd_eps{args.epsilon:.4f}_steps{args.pgd_steps}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, vae = load_models(args, device)
    
    # 浄化器
    purifier = VAEPurifier(vae, device).to(device)
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = VAEDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Classes: {classes}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("PGD Attack + VAE Defense Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Dataset: PCam")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255)")
    print(f"Steps: {args.pgd_steps}")
    print(f"Random start: {args.random_start}")
    print(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}, RGB input")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with VAE purification...")
    clean_purified_acc = get_accuracy(defense_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with VAE): {clean_purified_acc:.4f}")
    results['clean_acc_with_vae'] = clean_purified_acc
    
    # ========== 3. PGD攻撃 & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running PGD attack and evaluating adversarial images...")
    start_time = time.time()
    x_adv = run_pgd_attack(classifier_model, x_test, y_test, args.epsilon, args.alpha, 
                           args.pgd_steps, device, args.batch_size, args.random_start)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with VAE purification...")
    adv_defended_acc = get_accuracy(defense_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with VAE): {adv_defended_acc:.4f}")
    results['adv_acc_with_vae'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Dataset: PCam")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"        Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255), Steps: {args.pgd_steps}")
    print(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}, RGB input")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}")
    print(f"  With VAE purification:    {results['clean_acc_with_vae']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (PGD):")
    print(f"  Without defense:          {results['adv_acc_no_defense']:.4f}")
    print(f"  With VAE purification:    {results['adv_acc_with_vae']:.4f}")
    print(f"  Defense improvement:      {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # 予測取得
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_clean_purified = get_predictions(defense_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions(defense_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with VAE)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with VAE)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== 浄化画像を生成して保存 ====================
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
        y_test[:n_samples].cpu().numpy(), 
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # ==================== 敵対的サンプル保存 ====================
    torch.save({
        'x_clean': x_test.cpu(),
        'x_adv': x_adv.cpu(),
        'y': y_test.cpu(),
        'epsilon': args.epsilon,
        'alpha': args.alpha,
        'pgd_steps': args.pgd_steps,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PCam - PGD Attack + VAE Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: PCam\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: PGD\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Alpha: {args.alpha:.4f} ({args.alpha*255:.1f}/255)\n")
        f.write(f"Steps: {args.pgd_steps}\n")
        f.write(f"Random start: {args.random_start}\n")
        f.write(f"VAE: latent_dim={args.latent_dim}, base_ch={args.base_ch}, RGB input\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With VAE purification:    {results['clean_acc_with_vae']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (PGD):\n")
        f.write(f"  Without defense:          {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With VAE purification:    {results['adv_acc_with_vae']:.4f}\n")
        f.write(f"  Defense improvement:      {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (ViT Classifier)", cm_clean), 
                         ("Clean (with VAE)", cm_clean_purified),
                         ("Adversarial (No Defense)", cm_adv_no_def),
                         ("Adversarial (with VAE)", cm_adv_defended)]:
            if cm:
                f.write(f"{name}:\n")
                f.write(f"  TN: {cm['tn']:4d}  FP: {cm['fp']:4d}\n")
                f.write(f"  FN: {cm['fn']:4d}  TP: {cm['tp']:4d}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.4f}\n")
                f.write(f"  Precision: {cm['precision']:.4f}, Recall: {cm['recall']:.4f}, F1: {cm['f1']:.4f}\n\n")
    
    # JSON形式でも保存
    results_json = {
        'dataset': 'PCam',
        'classifier': 'ViT-B/16',
        'defense': 'VAE',
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'clean_acc_with_vae': results['clean_acc_with_vae'],
        'adv_acc_no_defense': results['adv_acc_no_defense'],
        'adv_acc_with_vae': results['adv_acc_with_vae'],
        'defense_improvement': results['defense_improvement'],
        'attack_time': results['attack_time'],
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")
    print(f"✅ Summary: {summary_path}")
    
    return results


if __name__ == '__main__':
    main()
