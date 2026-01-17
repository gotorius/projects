"""
Defense-GAN Adversarial Defense Evaluation for ChestX-ray Dataset - FGSM Attack

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
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation for ChestX-ray - FGSM Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
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
                        default='/mnt/data1/gotou/projects/chestxray/correct_samples_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints/20251228_234231/final_model.pth',
                        help='GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/fgsm/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
# ChestX-rayは[0,1]範囲の画像を使用（ImageNet正規化なし）
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


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ChestX-ray分類器を読み込み"""
    # 2クラス分類 (NORMAL, PNEUMONIA)
    num_classes = 2
    
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
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
    # 注意: EMA重みはBatchNormのrunning統計が不正確な場合があるため、通常の重みを優先
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        print(f"Loaded generator (normal weights) from {args.gan_ckpt}")
    elif 'ema_state_dict' in checkpoint:
        # EMAを使う場合はBatchNormの統計に注意
        generator.load_state_dict(checkpoint['ema_state_dict'], strict=False)
        print(f"Loaded generator (EMA - may have BN issues) from {args.gan_ckpt}")
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
def load_cached_samples(path):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    
    # クラス名（ChestX-ray固有）
    classes = ['NORMAL', 'PNEUMONIA']
    
    print(f"Loaded {len(x_test)} samples from {path}")
    print(f"Image shape: {x_test.shape}")
    print(f"Classes: {classes}")
    print(f"Label distribution: NORMAL={sum(y_test==0).item()}, PNEUMONIA={sum(y_test==1).item()}")
    
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM攻撃（ChestX-ray用、ImageNet正規化なし）"""
    x = x.clone().to(device)
    x.requires_grad = True
    
    # ImageNet正規化
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


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
    
    header = f"{'':>15}" + "".join([f"Pred {c:>10}" for c in classes])
    write_and_print(header)
    
    for i, true_class in enumerate(classes):
        row = f"{'True ' + true_class:>15}" + "".join([f"{cm[i, j]:>14}" for j in range(len(classes))])
        write_and_print(row)
    
    write_and_print(f"\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
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
    
    # Defense-GANが有効な場合のみサンプル画像を作成
    if x_purified_clean is not None and x_purified_adv is not None:
        for i in range(n):
            label = classes[labels[i]]
            
            images = [x_clean[i], x_adv[i], x_purified_clean[i], x_purified_adv[i]]
            grid = make_grid(images, nrow=4, padding=2, normalize=False)
            save_image(grid, os.path.join(save_dir, f'sample_{i}_{label}.png'))
    
    # 全体のグリッド画像も保存
    all_clean = make_grid(x_clean[:n], nrow=5, padding=2)
    all_adv = make_grid(x_adv[:n], nrow=5, padding=2)
    save_image(all_clean, os.path.join(save_dir, 'all_clean.png'))
    save_image(all_adv, os.path.join(save_dir, 'all_adversarial.png'))
    
    if x_purified_clean is not None:
        all_purified_clean = make_grid(x_purified_clean[:n], nrow=5, padding=2)
        save_image(all_purified_clean, os.path.join(save_dir, 'all_purified_clean.png'))
    
    if x_purified_adv is not None:
        all_purified_adv = make_grid(x_purified_adv[:n], nrow=5, padding=2)
        save_image(all_purified_adv, os.path.join(save_dir, 'all_purified_adv.png'))
    
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
    eps_str = f"eps{int(args.epsilon*255)}"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}_{eps_str}")
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
    write_and_print("FGSM Attack + Defense-GAN Evaluation for ChestX-ray")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Defense: Defense-GAN (enabled={args.use_defense})")
    if args.use_defense:
        write_and_print(f"  - Reconstruction iters: {args.rec_iters}")
        write_and_print(f"  - Reconstruction lr: {args.rec_lr}")
        write_and_print(f"  - Random restarts: {args.rec_rr}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"GAN checkpoint: {args.gan_ckpt}")
    write_and_print(f"Classifier checkpoint: {args.clf_ckpt}")
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
    
    # 3. FGSM攻撃
    write_and_print("\n[3/4] Running FGSM attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
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
        start_defense_time = time.time()
        adv_defense_acc, pred_adv_defense, x_purified_adv = evaluate_with_defense(
            defense_gan, classifier, x_adv, y_test, device, batch_size=args.batch_size, desc="Purifying adversarial images"
        )
        defense_time = time.time() - start_defense_time
        
        l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
        write_and_print(f"Adversarial accuracy (with Defense-GAN): {adv_defense_acc:.4f}")
        write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
        write_and_print(f"Defense time: {defense_time:.2f}s")
        results['adv_acc_with_defense'] = adv_defense_acc
        results['l2_adv_vs_purified'] = l2_adv_purified
        results['defense_improvement'] = adv_defense_acc - adv_acc
        results['defense_time'] = defense_time
    else:
        write_and_print("\n[4/4] Skipping Defense-GAN evaluation (defense disabled)")
    
    # 最終結果
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Dataset: ChestX-ray (NORMAL vs PNEUMONIA)")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Defense: Defense-GAN (enabled={args.use_defense})")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    if args.use_defense:
        write_and_print(f"  With Defense-GAN:            {results['clean_acc_with_defense']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (FGSM):")
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
    if args.use_defense:
        write_and_print(f"Defense time: {results['defense_time']:.2f}s")
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
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(
        x_test[:10], x_adv[:10], 
        x_purified_clean[:10] if x_purified_clean is not None else None,
        x_purified_adv[:10] if x_purified_adv is not None else None,
        y_test[:10], classes, samples_dir
    )
    
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
    print(f"JSON results: {os.path.join(log_dir, 'results.json')}")


if __name__ == '__main__':
    main()
