"""
Defense-GAN Adversarial Defense Evaluation for PCam Dataset - FGSM Attack (v3 Epoch20)

Defense-GANは、GANの生成器を使って入力画像を「浄化」し、
敵対的摂動を除去する防御手法です。

訓練済みモデル: checkpoints_v3/20251225_230534/checkpoint_epoch_0020.pth

浄化プロセス:
1. 入力画像 x に対して、最適な潜在変数 z* を勾配降下法で探索
2. z* から生成された画像 G(z*) を分類器への入力として使用

実行例:
python gan_fgsm_eval_v3_ep20.py --epsilon 0.031 --use_defense
python gan_fgsm_eval_v3_ep20.py --epsilon 0.031 --rec_iters 500 --rec_lr 0.01 --use_defense
python gan_fgsm_eval_v3_ep20.py --epsilon 0.031 --use_ema --use_defense
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
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation - FGSM Attack (v3 Epoch20)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true',
                        help='Enable Defense-GAN purification')
    parser.add_argument('--rec_iters', type=int, default=50,
                        help='Number of reconstruction iterations')
    parser.add_argument('--rec_lr', type=float, default=0.01,
                        help='Learning rate for reconstruction')
    parser.add_argument('--rec_rr', type=int, default=1,
                        help='Number of random restarts')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights for generator (recommended)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints_v3/20251225_230534/checkpoint_epoch_0020.pth',
                        help='GAN checkpoint path (v3 epoch 20)')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/fgsm/results_v3_ep20',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--defense_batch_size', type=int, default=4,
                        help='Batch size for Defense-GAN (smaller due to memory)')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Self-Attention (v3) ==========
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


# ========== ResNet Blocks (v3) ==========
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


# ========== Generator v3 ==========
class GeneratorV3(nn.Module):
    """
    ResNet-based Generator for 224x224 RGB images with Self-Attention
    Structure: latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """
    def __init__(self, latent_dim=512, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
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


# ========== Defense-GAN Purification (v3) ==========
class DefenseGAN:
    """
    Defense-GAN v3: 敵対的画像をGANの生成器を使って浄化
    
    参考論文:
    "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
    Pouya Samangouei, Maya Kabkab, Rama Chellappa - ICLR 2018
    
    アルゴリズム:
    1. 入力画像 x に対して、複数のランダム初期化 z から勾配降下法を実行
    2. L2損失 ||G(z) - x||^2 を最小化する z* を探索
    3. G(z*) を浄化された画像として出力
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
        
        # 勾配計算を無効化
        for param in self.generator.parameters():
            param.requires_grad = False
    
    def _denormalize_from_classifier(self, x):
        """ImageNet正規化 [0,1] を GAN空間 [-1,1] に変換"""
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        # ImageNet正規化を解除: x * std + mean -> [0,1]
        x = x * std + mean
        x = x.clamp(0, 1)
        # [0,1] -> [-1,1]
        return x * 2 - 1
    
    def _normalize_for_classifier(self, x):
        """GAN出力 [-1,1] を ImageNet正規化空間に変換"""
        # [-1,1] -> [0,1]
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        # ImageNet正規化
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def reconstruct_single(self, x_target):
        """
        単一画像の再構成（複数ランダムリスタートあり）
        x_target: (1, 3, H, W), GAN空間 [-1,1]
        """
        best_z = None
        best_loss = float('inf')
        
        for r in range(self.rec_rr):
            # ランダム初期化
            z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                x_gen = self.generator(z)
                loss = F.mse_loss(x_gen, x_target)
                loss.backward()
                optimizer.step()
            
            # 最終損失を計算
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target).item()
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_z = z.detach().clone()
        
        # 最良の z から画像を生成
        with torch.no_grad():
            x_rec = self.generator(best_z)
        
        return x_rec
    
    def reconstruct(self, x):
        """
        バッチ入力画像を再構成
        x: (B, 3, H, W), ImageNet正規化済み または [0,1]範囲
        return: 再構成画像 (B, 3, H, W), ImageNet正規化済み
        """
        batch_size = x.size(0)
        
        # GAN空間に変換
        x_target = self._denormalize_from_classifier(x)
        
        # バッチ処理
        best_z_list = []
        best_loss_list = [float('inf')] * batch_size
        
        # 各サンプルに対して最良のzを初期化
        for _ in range(batch_size):
            best_z_list.append(None)
        
        # 複数ランダムリスタート
        for r in range(self.rec_rr):
            # バッチ全体でランダム初期化
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr)
            
            # 反復最適化
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                x_gen = self.generator(z)
                # 各サンプルごとの損失
                loss_per_sample = F.mse_loss(x_gen, x_target, reduction='none')
                loss_per_sample = loss_per_sample.view(batch_size, -1).mean(dim=1)
                total_loss = loss_per_sample.sum()
                total_loss.backward()
                optimizer.step()
            
            # 最終損失を評価して最良のzを更新
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                for i in range(batch_size):
                    if final_loss[i].item() < best_loss_list[i]:
                        best_loss_list[i] = final_loss[i].item()
                        best_z_list[i] = z[i].detach().clone()
        
        # 最良のzをスタック
        best_z = torch.stack(best_z_list)
        
        # 最終再構成
        with torch.no_grad():
            x_rec = self.generator(best_z)
            # 分類器用の正規化空間に変換
            x_rec = self._normalize_for_classifier(x_rec)
        
        return x_rec


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
    
    # state_dict のキーを修正
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('fc.1.'):
            new_k = k.replace('fc.1.', 'fc.')
            fixed_state_dict[new_k] = v
        elif k.startswith('fc.0.') or k == 'fc.weight' or k == 'fc.bias':
            fixed_state_dict[k] = v
        else:
            fixed_state_dict[k] = v
    
    try:
        classifier.load_state_dict(fixed_state_dict)
    except RuntimeError:
        classifier.fc = nn.Linear(num_features, num_classes)
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
    print(f"✓ Loaded classifier from {args.clf_ckpt}")
    
    return classifier


def load_generator(args, device):
    """GAN生成器を読み込み (v3 epoch20対応)"""
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    # チェックポイントから設定を取得
    ckpt_args = checkpoint.get('args', {})
    latent_dim = ckpt_args.get('latent_dim', 512)
    ngf = ckpt_args.get('ngf', 64)
    
    print(f"  Generator config: latent_dim={latent_dim}, ngf={ngf}")
    
    # Generatorを作成
    generator = GeneratorV3(latent_dim=latent_dim, ngf=ngf, nc=3).to(device)
    
    # 重みを読み込む
    # EMA state_dictはパラメータのみ（バッファは含まない）なので、
    # まずgenerator_state_dictで全体を読み込み、その後EMAパラメータで上書き
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✓ Loaded generator v3 base weights from {args.gan_ckpt}")
        
        # EMA重みで上書き（推奨）
        if args.use_ema and 'ema_state_dict' in checkpoint:
            ema_state_dict = checkpoint['ema_state_dict']
            # EMAはtrainable parametersのみ含むので、部分的に更新
            current_state = generator.state_dict()
            for name, param in ema_state_dict.items():
                if name in current_state:
                    current_state[name] = param
            generator.load_state_dict(current_state)
            print(f"✓ Applied EMA weights")
    else:
        raise ValueError(f"Cannot find generator weights in checkpoint: {args.gan_ckpt}")
    
    generator.eval()
    
    # エポック情報を表示
    epoch = checkpoint.get('epoch', 'unknown')
    print(f"  Checkpoint epoch: {epoch}")
    
    return generator, latent_dim


# ========== データ読み込み ==========
def load_cached_samples(path):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f"✓ Loaded {len(x_test)} samples from {path}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM攻撃を実行"""
    x = x.clone().to(device)
    x.requires_grad = True
    
    # ImageNet正規化
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    # 摂動を加える
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


# ========== 評価関数 ==========
def evaluate(model, x_test, y_test, device, batch_size=16):
    """精度を計算（防御なし）"""
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
    reconstruction_losses = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
        x_batch = x_test[i:i+batch_size].to(device)
        y_batch = y_test[i:i+batch_size].to(device)
        
        # Defense-GAN purification
        x_purified = defense_gan.reconstruct(x_batch)
        x_purified_all.append(x_purified.cpu())
        
        # 分類
        with torch.no_grad():
            outputs = classifier(x_purified)
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
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
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
    
    n = min(len(x_clean), 16)
    
    # 個別画像を保存
    for i in range(min(n, 10)):
        label = classes[labels[i]]
        
        images = [x_clean[i], x_adv[i], x_purified_clean[i], x_purified_adv[i]]
        grid = make_grid(images, nrow=4, padding=2, normalize=False)
        save_image(grid, os.path.join(save_dir, f'sample_{i}_{label}.png'))
    
    # グリッド画像を保存
    # クリーン vs 敵対的
    grid_clean_adv = make_grid(
        torch.cat([x_clean[:n], x_adv[:n]], dim=0),
        nrow=n, padding=2, normalize=False
    )
    save_image(grid_clean_adv, os.path.join(save_dir, 'grid_clean_vs_adv.png'))
    
    # 浄化前後
    if x_purified_clean is not None and x_purified_adv is not None:
        grid_purified = make_grid(
            torch.cat([x_adv[:n], x_purified_adv[:n]], dim=0),
            nrow=n, padding=2, normalize=False
        )
        save_image(grid_purified, os.path.join(save_dir, 'grid_adv_vs_purified.png'))
    
    print(f"✓ Saved sample images to {save_dir}")


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
    print(f"\n{'='*70}")
    print(f"Defense-GAN v3 Evaluation (FGSM Attack)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M")
    defense_str = "defense" if args.use_defense else "no_defense"
    ema_str = "_ema" if args.use_ema else ""
    eps_str = f"eps{int(args.epsilon*255)}"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}{ema_str}_{eps_str}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")
    
    # 結果ファイル
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    # モデル読み込み
    write_and_print(f"\n{'='*70}")
    write_and_print("Loading Models")
    write_and_print(f"{'='*70}")
    
    classifier = load_classifier(args, device)
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
        write_and_print(f"\n✓ Defense-GAN enabled:")
        write_and_print(f"  - Reconstruction iterations: {args.rec_iters}")
        write_and_print(f"  - Learning rate: {args.rec_lr}")
        write_and_print(f"  - Random restarts: {args.rec_rr}")
        write_and_print(f"  - Using EMA weights: {args.use_ema}")
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("Experiment Configuration")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: FGSM")
    write_and_print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Defense: Defense-GAN v3 (enabled={args.use_defense})")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    
    results = {}
    
    # 1. クリーン画像の評価
    write_and_print(f"\n{'='*70}")
    write_and_print("[1/4] Evaluating clean images (classifier only)")
    write_and_print(f"{'='*70}")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    results['clean_acc'] = clean_acc
    
    # 2. クリーン画像 + Defense-GAN
    x_purified_clean = None
    if args.use_defense:
        write_and_print(f"\n{'='*70}")
        write_and_print("[2/4] Evaluating clean images with Defense-GAN")
        write_and_print(f"{'='*70}")
        start_time = time.time()
        clean_defense_acc, pred_clean_defense, x_purified_clean = evaluate_with_defense(
            defense_gan, classifier, x_test, y_test, device, 
            batch_size=args.defense_batch_size, desc="Purifying clean images"
        )
        defense_time_clean = time.time() - start_time
        
        l2_clean_purified = compute_l2_norm(x_test, x_purified_clean)
        write_and_print(f"Clean accuracy (with Defense-GAN): {clean_defense_acc:.4f} ({clean_defense_acc*100:.2f}%)")
        write_and_print(f"L2 norm (clean vs purified): {l2_clean_purified:.4f}")
        write_and_print(f"Defense time: {defense_time_clean:.1f}s ({defense_time_clean/len(x_test)*1000:.1f}ms/sample)")
        results['clean_acc_with_defense'] = clean_defense_acc
        results['l2_clean_vs_purified'] = l2_clean_purified
        results['defense_time_clean'] = defense_time_clean
    else:
        write_and_print(f"\n[2/4] Skipping Defense-GAN evaluation (defense disabled)")
    
    # 3. FGSM攻撃
    write_and_print(f"\n{'='*70}")
    write_and_print("[3/4] Running FGSM attack")
    write_and_print(f"{'='*70}")
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
    write_and_print(f"Adversarial accuracy (no defense): {adv_acc:.4f} ({adv_acc*100:.2f}%)")
    write_and_print(f"Attack success rate: {(clean_acc - adv_acc)/clean_acc*100:.2f}%")
    write_and_print(f"Attack time: {attack_time:.1f}s")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_clean_adv
    results['attack_time'] = attack_time
    results['attack_success_rate'] = (clean_acc - adv_acc) / clean_acc
    
    # 4. 敵対的画像 + Defense-GAN
    x_purified_adv = None
    if args.use_defense:
        write_and_print(f"\n{'='*70}")
        write_and_print("[4/4] Evaluating adversarial images with Defense-GAN")
        write_and_print(f"{'='*70}")
        start_time = time.time()
        adv_defense_acc, pred_adv_defense, x_purified_adv = evaluate_with_defense(
            defense_gan, classifier, x_adv, y_test, device, 
            batch_size=args.defense_batch_size, desc="Purifying adversarial images"
        )
        defense_time_adv = time.time() - start_time
        
        l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
        write_and_print(f"Adversarial accuracy (with Defense-GAN): {adv_defense_acc:.4f} ({adv_defense_acc*100:.2f}%)")
        write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
        write_and_print(f"Defense time: {defense_time_adv:.1f}s ({defense_time_adv/len(x_adv)*1000:.1f}ms/sample)")
        results['adv_acc_with_defense'] = adv_defense_acc
        results['l2_adv_vs_purified'] = l2_adv_purified
        results['defense_time_adv'] = defense_time_adv
        results['defense_improvement'] = adv_defense_acc - adv_acc
        results['recovery_rate'] = (adv_defense_acc - adv_acc) / (clean_acc - adv_acc) if clean_acc > adv_acc else 0
    else:
        write_and_print(f"\n[4/4] Skipping Defense-GAN evaluation (defense disabled)")
    
    # 最終結果
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"Defense: Defense-GAN v3 (enabled={args.use_defense}, EMA={args.use_ema})")
    write_and_print(f"-"*70)
    write_and_print(f"{'Metric':<40} {'Value':>15}")
    write_and_print(f"-"*70)
    write_and_print(f"{'Clean Accuracy (no defense)':<40} {results['clean_acc']*100:>14.2f}%")
    if args.use_defense:
        write_and_print(f"{'Clean Accuracy (with Defense-GAN)':<40} {results['clean_acc_with_defense']*100:>14.2f}%")
    write_and_print(f"-"*70)
    write_and_print(f"{'Adversarial Accuracy (no defense)':<40} {results['adv_acc_no_defense']*100:>14.2f}%")
    if args.use_defense:
        write_and_print(f"{'Adversarial Accuracy (with Defense-GAN)':<40} {results['adv_acc_with_defense']*100:>14.2f}%")
        write_and_print(f"-"*70)
        write_and_print(f"{'Defense Improvement':<40} {results['defense_improvement']*100:>+14.2f}%")
        write_and_print(f"{'Recovery Rate':<40} {results['recovery_rate']*100:>14.2f}%")
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
        save_sample_images(x_test[:16], x_adv[:16], x_purified_clean[:16], x_purified_adv[:16],
                           y_test[:16], classes, samples_dir)
    
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
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {log_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
