"""
Defense-GAN Adversarial Defense Evaluation for PCam Dataset (ViT Classifier) - AutoAttack

Defense-GANは、GANの生成器を使って入力画像を「浄化」し、
敵対的摂動を除去する防御手法です。

論文: "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
      Samangouei et al., ICLR 2018

AutoAttack:
- 複数の攻撃手法を組み合わせた強力な評価フレームワーク
- APGD-CE, APGD-DLR, FAB, Square Attack
- "Reliable evaluation of adversarial robustness with an ensemble of diverse
  parameter-free attacks" (Croce & Hein, 2020)

PCam特有の考慮:
- RGB画像（3チャンネル）→ Generator出力は3チャンネル
- 病理画像（組織切片のH&E染色画像）
- クラス: normal (0), tumor (1)

実行例:
python gan_autoattack_eval.py --epsilon 0.031 --aa_version standard
python gan_autoattack_eval.py --epsilon 0.031 --norm L2
"""

import os
import sys
import argparse
import time
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm
from autoattack import AutoAttack


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation for PCam (ViT) - AutoAttack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Lp norm for perturbation')
    parser.add_argument('--aa_version', type=str, default='standard', 
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true', default=True,
                        help='Enable Defense-GAN purification')
    parser.add_argument('--rec_iters', type=int, default=150,
                        help='Number of reconstruction iterations (reduced from 500 for speed)')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=3,
                        help='Number of random restarts (reduced from 10 for speed)')
    parser.add_argument('--early_stop_threshold', type=float, default=0.01,
                        help='Early stopping threshold for reconstruction loss')
    
    # パス設定
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/pcam/gan/checkpoints_v3/20251225_230534/checkpoint_epoch_0010.pth',
                        help='GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/gan/autoattack/results',
                        help='Output directory')
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Self-Attention ==========
class SelfAttention(nn.Module):
    """Self-Attention Module for capturing long-range dependencies"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.query = nn.utils.spectral_norm(self.query)
        self.key = nn.utils.spectral_norm(self.key)
        self.value = nn.utils.spectral_norm(self.value)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x


# ========== ResNet Blocks ==========
class ResBlockUp(nn.Module):
    """Residual Block with Upsampling for Generator"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.conv1, self.conv2, self.shortcut]:
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = self.bn1(x)
        h = F.relu(h)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.shortcut(x)
        
        return h + x


# ========== Generator for PCam (RGB, nc=3) ==========
class Generator(nn.Module):
    """
    ResNet-based Generator for 224x224 RGB images with Self-Attention
    Structure: latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    
    Note: nc=3 for RGB PCam images
    """
    def __init__(self, latent_dim=512, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.nc = nc
        self.init_size = 7
        
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        self.block1 = ResBlockUp(ngf * 8, ngf * 8)
        self.block2 = ResBlockUp(ngf * 8, ngf * 4)
        self.block3 = ResBlockUp(ngf * 4, ngf * 2)
        self.attention = SelfAttention(ngf * 2)
        self.block4 = ResBlockUp(ngf * 2, ngf)
        self.block5 = ResBlockUp(ngf, ngf // 2)
        
        self.bn_out = nn.BatchNorm2d(ngf // 2)
        self.conv_out = nn.Conv2d(ngf // 2, nc, 3, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 512, self.init_size, self.init_size)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.attention(h)
        h = self.block4(h)
        h = self.block5(h)
        
        h = self.bn_out(h)
        h = F.relu(h)
        h = self.conv_out(h)
        h = torch.tanh(h)
        
        return h


# ========== Defense-GAN Purification for PCam ==========
class DefenseGAN:
    """
    Defense-GAN for PCam: 敵対的画像をGANの生成器を使って浄化
    
    PCam特有の処理:
    - Generator出力は3チャンネル（RGB）
    - 最適化はRGB空間で直接行う
    
    最適化済み:
    - 早期停止による高速化
    - パラメータ削減による実用的な実行時間
    """
    def __init__(self, generator, latent_dim=512, rec_iters=150, rec_lr=0.01, 
                 rec_rr=3, early_stop_threshold=0.01, device='cuda'):
        self.generator = generator
        self.generator.eval()
        self.latent_dim = latent_dim
        self.rec_iters = rec_iters
        self.rec_lr = rec_lr
        self.rec_rr = rec_rr
        self.early_stop_threshold = early_stop_threshold
        self.device = device
    
    def _to_tanh_space(self, x):
        """[0,1] -> [-1,1]"""
        return x * 2 - 1
    
    def _from_tanh_space(self, x):
        """[-1,1] -> [0,1]"""
        return (x + 1) / 2
    
    def reconstruct(self, x):
        """
        入力画像xを生成器で再構成 (バッチ単位)
        x: (B, 3, H, W), [0,1]範囲
        return: 再構成画像 (B, 3, H, W), [0,1]範囲
        
        最適化: 早期停止条件により収束したら即座に終了
        """
        batch_size = x.size(0)
        
        # RGB -> tanh空間
        x_target = self._to_tanh_space(x)  # [-1, 1]
        
        best_z_list = [None] * batch_size
        best_loss_list = [float('inf')] * batch_size
        
        # Multiple random restarts
        for r in range(self.rec_rr):
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr, betas=(0.9, 0.999))
            
            for iter_idx in range(self.rec_iters):
                optimizer.zero_grad()
                
                x_gen = self.generator(z)  # [B, 3, H, W] in [-1, 1]
                loss = F.mse_loss(x_gen, x_target, reduction='none')
                loss = loss.view(batch_size, -1).mean(dim=1)
                
                total_loss = loss.sum()
                total_loss.backward()
                optimizer.step()
                
                # 早期停止: 全サンプルが閾値以下なら終了
                if loss.max().item() < self.early_stop_threshold:
                    break
            
            # Update best z for each sample
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                for i in range(batch_size):
                    if final_loss[i].item() < best_loss_list[i]:
                        best_loss_list[i] = final_loss[i].item()
                        best_z_list[i] = z[i].clone()
            
            # 早期停止: 十分良い解が見つかったらリスタートを打ち切る
            if max(best_loss_list) < self.early_stop_threshold:
                break
        
        # Generate final reconstruction
        best_z = torch.stack([z if z is not None else torch.randn(self.latent_dim, device=self.device) 
                             for z in best_z_list])
        
        with torch.no_grad():
            x_rec = self.generator(best_z)  # [B, 3, H, W] in [-1, 1]
            x_rec = self._from_tanh_space(x_rec)  # [0, 1]
            x_rec = x_rec.clamp(0, 1)
        
        return x_rec


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


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ViT分類器を読み込み"""
    # ViT分類器（2クラス: normal, tumor）
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


def load_generator(args, device):
    """GAN生成器を読み込み (PCam用、RGB)"""
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    # パラメータ読み込み
    latent_dim = 512
    ngf = 64
    nc = 3  # RGB
    
    # チェックポイントから設定を取得
    if 'args' in checkpoint:
        config = checkpoint['args']
        latent_dim = config.get('latent_dim', 512)
        ngf = config.get('ngf', 64)
    
    generator = Generator(latent_dim=latent_dim, ngf=ngf, nc=nc).to(device)
    
    # 重みを読み込む
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        print(f"Loaded generator (normal weights) from {args.gan_ckpt}")
    elif 'ema_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        print(f"Loaded generator (EMA) from {args.gan_ckpt}")
    else:
        try:
            generator.load_state_dict(checkpoint, strict=False)
            print(f"Loaded generator from {args.gan_ckpt}")
        except Exception as e:
            print(f"Warning: {e}")
    
    generator.eval()
    print(f"Generator config: latent_dim={latent_dim}, ngf={ngf}, nc={nc} (RGB)")
    
    return generator, latent_dim


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


# ========== Defense-GAN浄化の精度計算と予測取得（統合版）==========
def get_accuracy_and_predictions_with_defense(model, x, y, defense_gan, bs=4, device=None):
    """
    Defense-GANによる浄化後の精度と予測を同時に計算（重複計算を削減）
    
    Returns:
        accuracy: 精度
        predictions: 予測ラベル (numpy array)
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    all_preds = []
    
    for i in tqdm(range(n_batches), desc="Defense-GAN Purification"):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, len(x))
        x_batch = x[start_idx:end_idx].to(device)
        y_batch = y[start_idx:end_idx].to(device)
        
        # Defense-GANで浄化
        x_purified = defense_gan.reconstruct(x_batch)
        
        with torch.no_grad():
            outputs = model(x_purified)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            all_preds.append(preds.cpu())
    
    accuracy = correct / len(x)
    predictions = torch.cat(all_preds).numpy()
    
    return accuracy, predictions


def get_accuracy_with_defense(model, x, y, defense_gan, bs=4, device=None):
    """Defense-GANによる浄化後の精度を計算（互換性維持）"""
    accuracy, _ = get_accuracy_and_predictions_with_defense(model, x, y, defense_gan, bs, device)
    return accuracy


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


def get_predictions_with_defense(model, x, y, defense_gan, bs=4, device=None):
    """Defense-GANによる浄化後の予測を取得（互換性維持）"""
    _, predictions = get_accuracy_and_predictions_with_defense(model, x, y, defense_gan, bs, device)
    return predictions


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


# ========== AutoAttack実行 ==========
def run_autoattack(model, x_test, y_test, epsilon, norm, version, device, batch_size=32):
    """AutoAttackを実行して敵対的サンプルを生成"""
    print(f"\nRunning AutoAttack with epsilon={epsilon:.4f}, norm={norm}, version={version}...")
    
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, verbose=True)
    adversary.seed = 42
    
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(
            x_test.to(device), 
            y_test.to(device), 
            bs=batch_size
        )
    
    print(f"Generated {len(x_adv)} adversarial samples")
    return x_adv.cpu()


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_adv, 
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # 3枚を並べて保存: Clean, Adv, Adv+GAN
        triplet = torch.cat([
            x_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(triplet, nrow=3, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Adversarial | Adv+GAN]")


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
    log_dir = os.path.join(args.output_dir, f"autoattack_eps{args.epsilon:.4f}_{args.norm}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    generator, latent_dim = load_generator(args, device)
    
    # Defense-GAN
    defense_gan = DefenseGAN(
        generator, 
        latent_dim=latent_dim,
        rec_iters=args.rec_iters,
        rec_lr=args.rec_lr,
        rec_rr=args.rec_rr,
        early_stop_threshold=args.early_stop_threshold,
        device=device
    )
    
    # 推定実行時間を表示
    estimated_time_per_sample = args.rec_iters * args.rec_rr * 0.008  # 約8ms/iteration
    estimated_total = estimated_time_per_sample * 500  # 1回の浄化処理（敵対的画像のみ）
    print(f"Estimated time: ~{estimated_total/60:.1f} minutes (with early stopping, likely faster)")
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Classes: {classes}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + Defense-GAN Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"AutoAttack Version: {args.aa_version}")
    print(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/3] Evaluating clean images (ViT classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. AutoAttack & 敵対的画像の精度（防御なし） ==========
    print("\n[2/3] Running AutoAttack and evaluating adversarial images...")
    start_time = time.time()
    x_adv = run_autoattack(classifier_model, x_test, y_test, args.epsilon, args.norm, 
                           args.aa_version, device, args.batch_size)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 3. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[3/3] Evaluating adversarial images with Defense-GAN purification...")
    adv_defended_acc, pred_adv_defended = get_accuracy_and_predictions_with_defense(
        classifier_model, x_adv, y_test, defense_gan, bs=4, device=device)
    print(f"Adversarial accuracy (with Defense-GAN): {adv_defended_acc:.4f}")
    results['adv_acc_with_gan'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Dataset: PCam")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: AutoAttack ({args.aa_version}), Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255), Norm: {args.norm}")
    print(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}")
    print(f"Note: Generator is trained on RGB images")
    print(f"-"*70)
    print(f"Clean Accuracy (ViT classifier): {results['clean_acc_classifier']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (AutoAttack):")
    print(f"  Without defense:           {results['adv_acc_no_defense']:.4f}")
    print(f"  With Defense-GAN:          {results['adv_acc_with_gan']:.4f}")
    print(f"  Defense improvement:       {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # 予測取得（Defense-GAN処理済みの予測は既に取得済み）
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    # pred_adv_defended は上で既に取得済み
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "2. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "3. Adversarial Images (with Defense-GAN)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== 浄化画像を生成して保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified_adv = []
    
    for i in range(n_samples):
        x_purified_adv.append(defense_gan.reconstruct(x_adv[i:i+1].to(device)).cpu())
    
    x_purified_adv = torch.cat(x_purified_adv, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(), 
        x_adv[:n_samples].cpu(),
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
        'norm': args.norm,
        'aa_version': args.aa_version,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PCam - AutoAttack + Defense-GAN (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Dataset: PCam\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: AutoAttack ({args.aa_version})\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Norm: {args.norm}\n")
        f.write(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}\n")
        f.write(f"Samples: {len(x_test)}\n")
        f.write(f"Note: Generator is trained on RGB images\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write(f"Clean Accuracy (ViT classifier): {results['clean_acc_classifier']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (AutoAttack):\n")
        f.write(f"  Without defense:           {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With Defense-GAN:          {results['adv_acc_with_gan']:.4f}\n")
        f.write(f"  Defense improvement:       {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (ViT Classifier)", cm_clean), 
                         ("Adversarial (No Defense)", cm_adv_no_def),
                         ("Adversarial (with Defense-GAN)", cm_adv_defended)]:
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
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'adv_acc_no_defense': results['adv_acc_no_defense'],
        'adv_acc_with_gan': results['adv_acc_with_gan'],
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
