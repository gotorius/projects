"""
DermMel Dataset - VAE (MagNet-style) AutoAttack Evaluation (ViT Classifier)

VAEによるMagNet-style防御に対してAutoAttackを評価するスクリプト。
DermMelデータセット用のViT分類器を使用。

Reference:
    "MagNet: a Two-Pronged Defense against Adversarial Examples"
    Meng & Chen, CCS 2017
    https://arxiv.org/abs/1705.09064

    "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks"
    Croce & Hein, ICML 2020
    https://arxiv.org/abs/2003.01690

Usage:
    python vae_autoattack_eval.py --epsilon 0.031 --norm Linf --aa_version standard --gpu 0
    python vae_autoattack_eval.py --epsilon 0.5 --norm L2 --aa_version standard --gpu 0
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
from autoattack import AutoAttack


def parse_args():
    parser = argparse.ArgumentParser(
        description='VAE (MagNet-style) + AutoAttack Evaluation for DermMel (ViT Classifier)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 攻撃パラメータ
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttackの最大摂動量 (0-1スケール)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='摂動のノルム (Linf or L2)')
    parser.add_argument('--aa_version', type=str, default='standard', 
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttackのバージョン')
    
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
                        default='/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt',
                        help='キャッシュされたサンプルファイル')
    parser.add_argument('--vae_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/dermmel/vae/checkpoints/20260110_211546/best_model.pth',
                        help='VAEチェックポイントファイル')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth',
                        help='ViT分類器チェックポイントファイル')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vae/autoattack/results',
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
        """RGB画像 [0,1] を浄化"""
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
        x_purified = self.vae_purifier(x)
        x_norm = (x_purified - self.mean) / self.std
        return self.classifier(x_norm)


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


# ========== メイン ==========
def main():
    args = parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # 出力ディレクトリ作成 (MMDDHHMM形式)
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # 結果ファイル
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    write_and_print(f"\n{'='*70}")
    write_and_print("AutoAttack + VAE Defense Evaluation (ViT Classifier) - DermMel")
    write_and_print(f"{'='*70}")
    write_and_print(f"Device: {device}")
    write_and_print(f"Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Norm: {args.norm}")
    write_and_print(f"AutoAttack Version: {args.aa_version}")
    
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
    
    # ========== 3. AutoAttack敵対的画像の生成と評価 ==========
    print(f"\n[3/4] Running AutoAttack ({args.aa_version}, eps={args.epsilon:.5f})...")
    print("  This may take a while...")
    
    # AutoAttack実行
    adversary = AutoAttack(clf_wrapper, norm=args.norm, eps=args.epsilon, 
                           version=args.aa_version, verbose=True)
    
    start_time = time.time()
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), 
                                               bs=args.batch_size)
    attack_time = time.time() - start_time
    print(f"  AutoAttack completed in {attack_time:.1f}s")
    
    # 敵対的画像の精度（防御なし）
    adv_acc, adv_preds = get_accuracy(clf_wrapper, x_adv.cpu(), y_test, args.batch_size, device)
    print(f"  Adversarial accuracy (no defense): {adv_acc:.4f} ({adv_acc*100:.2f}%)")
    
    # ========== 4. AutoAttack敵対的画像 + VAE浄化の精度 ==========
    print(f"\n[4/4] Evaluating adversarial images + VAE purification...")
    adv_vae_acc, adv_vae_preds = get_accuracy(vae_clf_wrapper, x_adv.cpu(), y_test, args.batch_size, device)
    print(f"  Adversarial + VAE accuracy: {adv_vae_acc:.4f} ({adv_vae_acc*100:.2f}%)")
    
    # ========== 最終結果 ==========
    results = {
        'clean_acc': clean_acc,
        'clean_acc_with_vae': clean_vae_acc,
        'adv_acc_no_defense': adv_acc,
        'adv_acc_with_vae': adv_vae_acc,
        'defense_improvement': adv_vae_acc - adv_acc,
        'clean_acc_drop': clean_vae_acc - clean_acc,
        'attack_time_sec': attack_time
    }
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS (ViT Classifier) - DermMel")
    write_and_print(f"{'='*70}")
    write_and_print(f"Classifier: ViT-B/16")
    write_and_print(f"Attack: AutoAttack ({args.aa_version}, {args.norm})")
    write_and_print(f"        Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Defense: VAE (MagNet-style)")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With VAE purification:       {results['clean_acc_with_vae']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (AutoAttack):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With VAE purification:       {results['adv_acc_with_vae']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Clean accuracy drop:           {results['clean_acc_drop']:+.4f}")
    write_and_print(f"Attack time:                   {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    # 混同行列
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, clean_preds, "1. Clean Images", classes, results_file)
    cm_results['clean_vae'] = print_confusion_matrix(y_true, clean_vae_preds, "2. Clean Images (with VAE)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, adv_preds, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_vae'] = print_confusion_matrix(y_true, adv_vae_preds, "4. Adversarial Images (with VAE)", classes, results_file)
    
    # 可視化サンプル保存
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    n_vis = min(8, len(x_test))
    x_adv_cpu = x_adv[:n_vis].cpu()
    with torch.no_grad():
        x_vae_clean = vae.reconstruct(x_test[:n_vis].to(device)).cpu()
        x_vae_adv = vae.reconstruct(x_adv_cpu.to(device)).cpu()
    
    comparison = torch.cat([
        x_test[:n_vis],      # Clean
        x_adv_cpu,           # Adversarial
        x_vae_clean,         # Clean + VAE
        x_vae_adv            # Adversarial + VAE
    ], dim=0)
    
    grid = make_grid(comparison, nrow=n_vis, normalize=False, padding=2)
    save_path = os.path.join(samples_dir, 'comparison_autoattack.png')
    save_image(grid, save_path)
    write_and_print(f"Visualization saved to: {save_path}")
    
    results_file.close()
    
    print(f"\nResults saved to {log_dir}")
    print(f"Text results: {os.path.join(log_dir, 'results.txt')}")


if __name__ == '__main__':
    main()
