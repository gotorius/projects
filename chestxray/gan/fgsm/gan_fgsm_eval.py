"""
Defense-GAN FGSM Evaluation Script for ChestX-ray Dataset

Reference: 
"Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
Samangouei et al., ICLR 2018

Defense-GANの浄化方法:
1. 入力画像xに対して、潜在ベクトルzを最適化で探索
2. z* = argmin_z ||G(z) - x||_2^2
3. 再構成画像G(z*)を分類器に入力

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. FGSM敵対的画像の分類精度（防御なし）
4. FGSM敵対的画像を浄化した後の分類精度（防御あり）

実行例:
python gan_fgsm_eval.py --gan_ckpt ../checkpoints/best_model.pth --gpu 0
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
    parser = argparse.ArgumentParser(description='Defense-GAN FGSM Evaluation')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # Defense-GAN設定
    parser.add_argument('--z_dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--n_iter', type=int, default=500,
                        help='Number of optimization iterations for z')
    parser.add_argument('--n_restarts', type=int, default=20,
                        help='Number of random restarts')
    parser.add_argument('--lr_z', type=float, default=0.1,
                        help='Learning rate for z optimization')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use (None = all)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/correct_samples_500.pt',
                        help='Path to cached samples')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints/20251205_033202/best_model.pth',
                        help='Defense-GAN checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/fgsm/results',
                        help='Output directory')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Generator (DCGAN-style) ==========
class Generator(nn.Module):
    """DCGAN-style Generator for 224x224 grayscale images"""
    def __init__(self, z_dim=128, ngf=64, img_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.init_size = 7
        
        self.fc = nn.Sequential(
            nn.Linear(z_dim, ngf * 16 * self.init_size * self.init_size),
            nn.BatchNorm1d(ngf * 16 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, self.init_size, self.init_size)
        return self.main(x)


# ========== Defense-GAN Purifier ==========
class DefenseGANPurifier(nn.Module):
    """
    Defense-GAN: 入力画像に最も近い潜在ベクトルを最適化で探索し、
    GANで再構成することで敵対的摂動を除去
    """
    def __init__(self, generator, device, z_dim=128, n_iter=200, n_restarts=10, lr_z=0.01):
        super().__init__()
        self.generator = generator
        self.device = device
        self.z_dim = z_dim
        self.n_iter = n_iter
        self.n_restarts = n_restarts
        self.lr_z = lr_z
    
    def rgb_to_gray(self, x_rgb):
        """RGB [0,1] → グレースケール [0,1]"""
        weights = torch.tensor([0.299, 0.587, 0.114], device=x_rgb.device).view(1, 3, 1, 1)
        return (x_rgb * weights).sum(dim=1, keepdim=True)
    
    def gray_to_rgb(self, x_gray):
        """グレースケール [0,1] → RGB [0,1]"""
        return x_gray.repeat(1, 3, 1, 1)
    
    def pixel_to_gan(self, x):
        """[0,1] → [-1,1]"""
        return x * 2.0 - 1.0
    
    def gan_to_pixel(self, x):
        """[-1,1] → [0,1]"""
        return torch.clamp((x + 1.0) / 2.0, 0, 1)
    
    def find_best_z(self, x_gray, debug=False):
        """
        入力画像に最も近い潜在ベクトルを最適化で探索
        x_gray: グレースケール画像 [0,1], shape (B, 1, H, W)
        debug: デバッグ出力を行うかどうか
        """
        batch_size = x_gray.size(0)
        x_target = self.pixel_to_gan(x_gray)  # [-1, 1]
        
        best_z = None
        best_loss = float('inf') * torch.ones(batch_size, device=self.device)
        
        for restart in range(self.n_restarts):
            # ランダム初期化
            z = torch.randn(batch_size, self.z_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.lr_z)
            
            for i in range(self.n_iter):
                optimizer.zero_grad()
                
                x_recon = self.generator(z)
                loss = F.mse_loss(x_recon, x_target, reduction='none')
                loss = loss.view(batch_size, -1).mean(dim=1)
                
                loss.sum().backward()
                optimizer.step()
            
            # 最良のzを更新
            with torch.no_grad():
                x_recon = self.generator(z)
                final_loss = F.mse_loss(x_recon, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                if debug and restart == 0:
                    print(f"[PURIFIER] Restart {restart}: loss={final_loss[0].item():.6f}")
                
                for b in range(batch_size):
                    if final_loss[b] < best_loss[b]:
                        best_loss[b] = final_loss[b]
                        if best_z is None:
                            best_z = z.clone()
                        else:
                            best_z[b] = z[b].clone()
        
        if debug:
            print(f"[PURIFIER] Best loss after all restarts: {best_loss[0].item():.6f}")
        
        return best_z
    
    def forward(self, x_rgb, debug=False):
        """
        RGB画像 [0,1] を浄化
        x_rgb: (B, 3, H, W), [0, 1]
        return: 浄化されたRGB画像 (B, 3, H, W), [0, 1]
        """
        # RGB → グレースケール
        x_gray = self.rgb_to_gray(x_rgb)
        
        if debug:
            print(f"[PURIFIER] Input (RGB) shape: {x_rgb.shape}, gray shape: {x_gray.shape}")
            print(f"[PURIFIER] Input RGB stats: min={x_rgb.min():.4f}, max={x_rgb.max():.4f}, mean={x_rgb.mean():.4f}")
            print(f"[PURIFIER] Input gray stats: min={x_gray.min():.4f}, max={x_gray.max():.4f}, mean={x_gray.mean():.4f}")
        
        # 最適な潜在ベクトルを探索
        self.generator.eval()
        best_z = self.find_best_z(x_gray, debug=debug)
        
        # 再構成
        with torch.no_grad():
            x_recon = self.generator(best_z)
        
        if debug:
            print(f"[PURIFIER] Generator output stats: min={x_recon.min():.4f}, max={x_recon.max():.4f}, mean={x_recon.mean():.4f}")
        
        # [-1,1] → [0,1] → RGB
        x_recon_pixel = self.gan_to_pixel(x_recon)
        x_rgb_out = self.gray_to_rgb(x_recon_pixel)
        
        if debug:
            print(f"[PURIFIER] Final output stats: min={x_rgb_out.min():.4f}, max={x_rgb_out.max():.4f}, mean={x_rgb_out.mean():.4f}")
        
        return x_rgb_out


# ========== 分類器ラッパー ==========
class ClassifierWrapper(nn.Module):
    """ImageNet正規化を含む分類器ラッパー"""
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


# ========== Defense-GANラッパー ==========
class DefenseGANWrapper(nn.Module):
    """Defense-GAN + 分類器のラッパー"""
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
    """分類器とDefense-GANを読み込み"""
    # 分類器
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    # Generator
    gan_ckpt = torch.load(args.gan_ckpt, map_location=device)
    if 'args' in gan_ckpt:
        z_dim = gan_ckpt['args'].get('z_dim', args.z_dim)
        ngf = gan_ckpt['args'].get('ngf', 64)
    else:
        z_dim = args.z_dim
        ngf = 64
    
    generator = Generator(z_dim=z_dim, ngf=ngf, img_channels=1).to(device)
    generator.load_state_dict(gan_ckpt['generator'])
    generator.eval()
    print(f"Loaded Generator from {args.gan_ckpt}")
    
    return classifier, generator


# ========== データ読み込み ==========
def load_cached_samples(path, device):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    print(f"Loaded {len(x_test)} cached samples from {path}")
    return x_test, y_test


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


# ========== 評価関数 ==========
def get_accuracy(model, x_test, y_test, bs=32, device='cuda'):
    """精度を計算"""
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
    """予測を取得"""
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
    """混同行列を表示"""
    # 両クラスのラベルを指定して正しい形状の混同行列を取得
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{title}")
    print("-" * 50)
    print(f"{'':>15} {'Pred ' + classes[0]:>15} {'Pred ' + classes[1]:>15}")
    print(f"{'True ' + classes[0]:>15} {cm[0,0]:>15} {cm[0,1]:>15}")
    print(f"{'True ' + classes[1]:>15} {cm[1,0]:>15} {cm[1,1]:>15}")
    
    # cm.ravel()が4要素でない場合に対応
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# ========== 浄化評価（Defense-GANは遅いため別関数）==========
def evaluate_with_purification(purifier, classifier_model, x_test, y_test, device, desc="Purifying", debug=False):
    """Defense-GANで浄化しながら評価"""
    purifier.eval()
    classifier_model.eval()
    
    correct = 0
    total = 0
    predictions = []
    
    # バッチサイズ1で処理（メモリとの兼ね合い）
    for i in tqdm(range(len(x_test)), desc=desc):
        x = x_test[i:i+1].to(device)
        y = y_test[i:i+1].to(device)
        
        # デバッグ: 入力画像の統計
        if debug and i == 0:
            print(f"\n[DEBUG] Input image stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
        
        # 浄化（デバッグモードで詳細出力、最初の1つだけ）
        x_purified = purifier(x, debug=debug and i == 0)
        
        # デバッグ: 浄化後の画像の統計
        if debug and i == 0:
            print(f"[DEBUG] Purified image stats: min={x_purified.min():.4f}, max={x_purified.max():.4f}, mean={x_purified.mean():.4f}")
        
        # 正規化して分類
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        x_norm = (x_purified - mean) / std
        
        if debug and i == 0:
            print(f"[DEBUG] Normalized image stats: min={x_norm.min():.4f}, max={x_norm.max():.4f}, mean={x_norm.mean():.4f}")
        
        with torch.no_grad():
            outputs = classifier_model.classifier(x_norm)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            if debug and i < 3:
                print(f"[DEBUG] Sample {i}: true={y.item()}, pred={predicted.item()}, probs={probs[0].cpu().numpy()}")
            
            correct += (predicted == y).sum().item()
            total += 1
            predictions.append(predicted.cpu().numpy()[0])
    
    accuracy = correct / total
    return accuracy, np.array(predictions)


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv,
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
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
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"fgsm_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, generator = load_models(args, device)
    
    # 浄化器
    purifier = DefenseGANPurifier(
        generator, device,
        z_dim=args.z_dim,
        n_iter=args.n_iter,
        n_restarts=args.n_restarts,
        lr_z=args.lr_z
    ).to(device)
    
    # ラッパー
    classifier_model = ClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # クラス名
    test_dir = os.path.join(args.data_dir, 'test')
    classes = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
    print(f"Classes: {classes}")
    
    # データ読み込み
    x_test, y_test = load_cached_samples(args.cached_samples, device)
    
    # サンプル数を制限
    if args.num_samples is not None and args.num_samples < len(x_test):
        x_test = x_test[:args.num_samples]
        y_test = y_test[:args.num_samples]
        print(f"Limited to {args.num_samples} samples for quick testing")
    
    # クラス分布を表示
    unique, counts = torch.unique(y_test, return_counts=True)
    print(f"Class distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    print(f"  Class 0 ({classes[0]}): {(y_test == 0).sum().item()}")
    print(f"  Class 1 ({classes[1]}): {(y_test == 1).sum().item()}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("FGSM Attack + Defense-GAN Evaluation")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Defense-GAN: n_iter={args.n_iter}, n_restarts={args.n_restarts}, lr_z={args.lr_z}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with Defense-GAN purification...")
    clean_purified_acc, pred_clean_purified = evaluate_with_purification(
        purifier, classifier_model, x_test, y_test, device, desc="Purifying clean images", debug=True
    )
    print(f"Clean accuracy (with Defense-GAN): {clean_purified_acc:.4f}")
    results['clean_acc_with_defensegan'] = clean_purified_acc
    
    # ========== 3. FGSM攻撃 & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running FGSM attack and evaluating adversarial images...")
    start_time = time.time()
    
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM Attack"):
        x_batch = x_test[i:i+args.batch_size].to(device)
        y_batch = y_test[i:i+args.batch_size].to(device)
        x_adv_batch = fgsm_attack(classifier_model, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with Defense-GAN purification...")
    adv_defended_acc, pred_adv_defended = evaluate_with_purification(
        purifier, classifier_model, x_adv, y_test, device, desc="Purifying adversarial images"
    )
    print(f"Adversarial accuracy (with Defense-GAN): {adv_defended_acc:.4f}")
    results['adv_acc_with_defensegan'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Defense-GAN: n_iter={args.n_iter}, n_restarts={args.n_restarts}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:             {results['clean_acc_classifier']:.4f}")
    print(f"  With Defense-GAN:            {results['clean_acc_with_defensegan']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (FGSM):")
    print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    print(f"  With Defense-GAN:            {results['adv_acc_with_defensegan']:.4f}")
    print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (Classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with Defense-GAN)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with Defense-GAN)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== サンプル画像保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified_clean_samples = []
    x_purified_adv_samples = []
    
    for i in tqdm(range(n_samples), desc="Generating samples"):
        x_purified_clean_samples.append(purifier(x_test[i:i+1].to(device)).cpu())
        x_purified_adv_samples.append(purifier(x_adv[i:i+1].to(device)).cpu())
    
    x_purified_clean_samples = torch.cat(x_purified_clean_samples, dim=0)
    x_purified_adv_samples = torch.cat(x_purified_adv_samples, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(),
        x_adv[:n_samples].cpu(),
        x_purified_clean_samples,
        x_purified_adv_samples,
        y_test[:n_samples],
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # 結果保存
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        results_save = {k: v for k, v in results.items() if k != 'confusion_matrices'}
        results_save['confusion_matrices'] = {
            k: {kk: int(vv) if isinstance(vv, (int, np.integer)) else float(vv)
                for kk, vv in v.items()}
            for k, v in results['confusion_matrices'].items()
        }
        json.dump(results_save, f, indent=2)
    
    # 設定保存
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
