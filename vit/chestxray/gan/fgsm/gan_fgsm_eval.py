"""
Defense-GAN Adversarial Defense Evaluation for ChestX-ray Dataset (ViT Classifier) - FGSM Attack

Defense-GANは、GANの生成器を使って入力画像を「浄化」し、
敵対的摂動を除去する防御手法です。

浄化プロセス:
1. 入力画像 x に対して、最適な潜在変数 z* を勾配降下法で探索
2. z* から生成された画像 G(z*) を分類器への入力として使用

ChestX-ray特有の考慮:
- グレースケール画像（1チャンネル）→ Generator出力は1チャンネル
- 分類器への入力は3チャンネル（RGB複製）
- クラス: NORMAL (0), PNEUMONIA (1)

実行例:
python gan_fgsm_eval.py --epsilon 0.031 --use_defense
python gan_fgsm_eval.py --epsilon 0.031 --rec_iters 500 --rec_lr 0.01
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


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation for ChestX-ray (ViT) - FGSM Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true', default=True,
                        help='Enable Defense-GAN purification')
    parser.add_argument('--rec_iters', type=int, default=500,
                        help='Number of reconstruction iterations')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=10,
                        help='Number of random restarts')
    
    # パス設定
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints/20251228_234231/final_model.pth',
                        help='GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/gan/fgsm/results',
                        help='Output directory')
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt',
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


# ========== Generator for ChestX-ray (Grayscale, nc=1) ==========
class Generator(nn.Module):
    """
    ResNet-based Generator for 224x224 Grayscale images with Self-Attention
    Structure: latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    
    Note: nc=1 for grayscale ChestX-ray images
    """
    def __init__(self, latent_dim=512, ngf=64, nc=1):
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


# ========== Defense-GAN Purification for ChestX-ray ==========
class DefenseGAN:
    """
    Defense-GAN for ChestX-ray: 敵対的画像をGANの生成器を使って浄化
    
    ChestX-ray特有の処理:
    - Generator出力は1チャンネル（グレースケール）
    - 分類器入力は3チャンネル（RGB複製）
    - 最適化はグレースケール空間で行う
    """
    def __init__(self, generator, latent_dim=512, rec_iters=500, rec_lr=0.01, 
                 rec_rr=10, device='cuda'):
        self.generator = generator
        self.generator.eval()
        self.latent_dim = latent_dim
        self.rec_iters = rec_iters
        self.rec_lr = rec_lr
        self.rec_rr = rec_rr
        self.device = device
    
    def _rgb_to_gray(self, x):
        """RGB画像をグレースケールに変換 (3ch -> 1ch)"""
        # x: [B, 3, H, W] in [0, 1]
        # 標準的なグレースケール変換
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray
    
    def _gray_to_rgb(self, x):
        """グレースケールをRGBに複製 (1ch -> 3ch)"""
        # x: [B, 1, H, W] in [0, 1]
        return x.repeat(1, 3, 1, 1)
    
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
        """
        batch_size = x.size(0)
        
        # RGB -> グレースケール -> tanh空間
        x_gray = self._rgb_to_gray(x)  # [B, 1, H, W]
        x_target = self._to_tanh_space(x_gray)  # [-1, 1]
        
        best_z_list = [None] * batch_size
        best_loss_list = [float('inf')] * batch_size
        
        # Multiple random restarts
        for r in range(self.rec_rr):
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr, betas=(0.9, 0.999))
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                
                x_gen = self.generator(z)  # [B, 1, H, W] in [-1, 1]
                loss = F.mse_loss(x_gen, x_target, reduction='none')
                loss = loss.view(batch_size, -1).mean(dim=1)
                
                total_loss = loss.sum()
                total_loss.backward()
                optimizer.step()
            
            # Update best z for each sample
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                for i in range(batch_size):
                    if final_loss[i].item() < best_loss_list[i]:
                        best_loss_list[i] = final_loss[i].item()
                        best_z_list[i] = z[i].clone()
        
        # Generate final reconstruction
        best_z = torch.stack([z if z is not None else torch.randn(self.latent_dim, device=self.device) 
                             for z in best_z_list])
        
        with torch.no_grad():
            x_rec = self.generator(best_z)  # [B, 1, H, W] in [-1, 1]
            x_rec = self._from_tanh_space(x_rec)  # [0, 1]
            x_rec = x_rec.clamp(0, 1)
            x_rec = self._gray_to_rgb(x_rec)  # [B, 3, H, W]
        
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
    # ViT分類器（2クラス: NORMAL, PNEUMONIA）
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
    """GAN生成器を読み込み (ChestX-ray用、グレースケール)"""
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    # パラメータ読み込み
    latent_dim = 512
    ngf = 64
    nc = 1  # グレースケール
    
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
    print(f"Generator config: latent_dim={latent_dim}, ngf={ngf}, nc={nc} (grayscale)")
    
    return generator, latent_dim


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み（ViT分類器で正しく分類されたサンプル）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NORMAL', 'PNEUMONIA'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM攻撃"""
    x = x.clone().to(device)
    x.requires_grad = True
    
    outputs = model(x)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


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


# ========== Defense-GAN浄化の精度計算 ==========
def get_accuracy_with_defense(model, x, y, defense_gan, bs=1, device=None):
    """Defense-GANによる浄化後の精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
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


def get_predictions_with_defense(model, x, defense_gan, bs=1, device=None):
    """Defense-GANによる浄化後の予測を取得"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    
    for i in tqdm(range(n_batches), desc="Getting predictions with defense"):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, len(x))
        x_batch = x[start_idx:end_idx].to(device)
        
        # Defense-GANで浄化
        x_purified = defense_gan.reconstruct(x_batch)
        
        with torch.no_grad():
            outputs = model(x_purified)
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


# ========== FGSM攻撃実行 ==========
def run_fgsm_attack(model, x_test, y_test, epsilon, device, batch_size=32):
    """FGSM攻撃を実行して敵対的サンプルを生成"""
    print(f"\nRunning FGSM attack with epsilon={epsilon:.4f}...")
    
    n_batches = (len(x_test) + batch_size - 1) // batch_size
    x_adv_list = []
    
    for i in tqdm(range(n_batches), desc="FGSM Attack"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        x_batch = x_test[start_idx:end_idx].to(device)
        y_batch = y_test[start_idx:end_idx].to(device)
        
        x_adv_batch = fgsm_attack(model, x_batch, y_batch, epsilon, device)
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
        
        # 4枚を並べて保存: Clean, Clean+GAN, Adv, Adv+GAN
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Clean+GAN | Adversarial | Adv+GAN]")


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
    log_dir = os.path.join(args.output_dir, f"fgsm_eps{args.epsilon:.4f}_{timestamp}")
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
        device=device
    )
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Classes: {classes}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("FGSM Attack + Defense-GAN Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with Defense-GAN purification...")
    clean_purified_acc = get_accuracy_with_defense(classifier_model, x_test, y_test, defense_gan, bs=1, device=device)
    print(f"Clean accuracy (with Defense-GAN): {clean_purified_acc:.4f}")
    results['clean_acc_with_gan'] = clean_purified_acc
    
    # ========== 3. FGSM攻撃 & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running FGSM attack and evaluating adversarial images...")
    start_time = time.time()
    x_adv = run_fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device, args.batch_size)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with Defense-GAN purification...")
    adv_defended_acc = get_accuracy_with_defense(classifier_model, x_adv, y_test, defense_gan, bs=1, device=device)
    print(f"Adversarial accuracy (with Defense-GAN): {adv_defended_acc:.4f}")
    results['adv_acc_with_gan'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}")
    print(f"Note: Generator is trained on grayscale images")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  ViT classifier only:       {results['clean_acc_classifier']:.4f}")
    print(f"  With Defense-GAN:          {results['clean_acc_with_gan']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (FGSM):")
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
    
    # 予測取得
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_clean_purified = get_predictions_with_defense(classifier_model, x_test, defense_gan, bs=1, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions_with_defense(classifier_model, x_adv, defense_gan, bs=1, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with Defense-GAN)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with Defense-GAN)", classes)
    
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
    
    for i in range(n_samples):
        x_purified_clean.append(defense_gan.reconstruct(x_test[i:i+1].to(device)).cpu())
        x_purified_adv.append(defense_gan.reconstruct(x_adv[i:i+1].to(device)).cpu())
    
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
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - FGSM Attack + Defense-GAN (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: FGSM\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Defense-GAN: rec_iters={args.rec_iters}, rec_lr={args.rec_lr}, rec_rr={args.rec_rr}\n")
        f.write(f"Samples: {len(x_test)}\n")
        f.write(f"Note: Generator is trained on grayscale images\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  ViT classifier only:       {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With Defense-GAN:          {results['clean_acc_with_gan']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (FGSM):\n")
        f.write(f"  Without defense:           {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With Defense-GAN:          {results['adv_acc_with_gan']:.4f}\n")
        f.write(f"  Defense improvement:       {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (ViT Classifier)", cm_clean), 
                         ("Clean (with Defense-GAN)", cm_clean_purified),
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
        'classifier': 'ViT-B/16',
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'clean_acc_with_gan': results['clean_acc_with_gan'],
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
