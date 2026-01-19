"""
Defense-GAN Adversarial Defense Evaluation for PCam Dataset - AutoAttack (v4 Improved)

v4改善点（ChestX-ray v2と同等）:
1. L-BFGS最適化（より高速で正確な収束）
2. Perceptual Loss（VGGベース、オプション）
3. 潜在空間の正則化
4. より効率的な再構成アルゴリズム
5. 複数初期化によるランダムリスタート（デフォルト5回）
6. RGB画像への対応

AutoAttack:
- 複数の攻撃を組み合わせた強力な自動攻撃
- APGD-CE, APGD-DLR, FAB, Square Attackの組み合わせ
- パラメータフリーで再現性が高い

実行例:
python gan_autoattack_eval_v4.py --epsilon 0.031 --use_defense
python gan_autoattack_eval_v4.py --epsilon 0.031 --version standard
python gan_autoattack_eval_v4.py --epsilon 0.031 --rec_iters 200 --rec_rr 5
"""

import os
import sys
import argparse
import time
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from datetime import datetime
from tqdm.auto import tqdm

# AutoAttack
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not installed. Please run: pip install autoattack")
    sys.exit(1)


# ========== 引数 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN AutoAttack Evaluation v4')
    
    # AutoAttack設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Maximum perturbation (L-inf)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Perturbation norm')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    parser.add_argument('--attacks_to_run', type=str, nargs='+', 
                        default=None,
                        help='Specific attacks to run (e.g., apgd-ce apgd-dlr fab square)')
    
    # Defense-GAN設定
    parser.add_argument('--use_defense', action='store_true')
    parser.add_argument('--rec_iters', type=int, default=200,
                        help='L-BFGS iterations (fewer needed)')
    parser.add_argument('--rec_rr', type=int, default=5,
                        help='Random restarts')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Perceptual loss weight (0 recommended for medical images)')
    parser.add_argument('--use_lbfgs', action='store_true', default=True,
                        help='Use L-BFGS optimizer (faster convergence)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA weights for generator (recommended)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints_v3/20251225_230534/checkpoint_epoch_0020.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/autoattack/results_v4')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent dimension (match GAN checkpoint)')
    
    # クイックテスト
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with first 10 samples')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to use (overrides quick_test)')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Self-Attention ==========
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ========== ResBlock ==========
class ResBlockUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        h = F.relu(self.bn1(x))
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        x = self.shortcut(F.interpolate(x, scale_factor=2, mode='nearest'))
        return h + x


# ========== Generator (RGB対応) ==========
class Generator(nn.Module):
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
    
    def forward(self, z):
        h = self.fc(z).view(-1, 512, self.init_size, self.init_size)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.attention(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.relu(self.bn_out(h))
        return torch.tanh(self.conv_out(h))


# ========== Perceptual Loss用VGG ==========
class VGGFeatures(nn.Module):
    """VGG16から特徴量を抽出（軽量版）"""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
        self.features = vgg.to(device).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ImageNet正規化用
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    def forward(self, x):
        # x: [B, 3, H, W] in [0, 1]
        x = (x - self.mean) / self.std
        return self.features(x)


# ========== 改良版 Defense-GAN (v4) ==========
class DefenseGANv4:
    """
    Defense-GAN with improved reconstruction for RGB images:
    1. L-BFGS optimizer for faster convergence
    2. Perceptual loss for better quality (optional)
    3. Multiple random restarts
    4. RGB image support
    """
    def __init__(self, generator, latent_dim=512, rec_iters=200, rec_rr=5,
                 perceptual_weight=0.0, use_lbfgs=True, device='cuda'):
        self.generator = generator
        self.generator.eval()
        self.latent_dim = latent_dim
        self.rec_iters = rec_iters
        self.rec_rr = rec_rr
        self.perceptual_weight = perceptual_weight
        self.use_lbfgs = use_lbfgs
        self.device = device
        
        # VGG for perceptual loss
        if perceptual_weight > 0:
            self.vgg = VGGFeatures(device)
        else:
            self.vgg = None
    
    def _to_gan_space(self, x):
        """[0,1] -> [-1,1]"""
        return x * 2 - 1
    
    def _from_gan_space(self, x):
        """[-1,1] -> [0,1]"""
        return (x + 1) / 2
    
    def _compute_loss(self, z, x_target):
        """Total reconstruction loss"""
        x_gen = self.generator(z)  # [B, 3, H, W] in [-1, 1]
        
        # MSE loss in GAN space [-1, 1]
        x_target_gan = self._to_gan_space(x_target)  # [0,1] -> [-1,1]
        mse_loss = F.mse_loss(x_gen, x_target_gan)
        
        # Perceptual loss (optional)
        if self.vgg is not None and self.perceptual_weight > 0:
            x_gen_01 = self._from_gan_space(x_gen)  # [-1,1] -> [0,1]
            
            feat_gen = self.vgg(x_gen_01)
            feat_target = self.vgg(x_target)
            perceptual_loss = F.mse_loss(feat_gen, feat_target)
            
            total_loss = mse_loss + self.perceptual_weight * perceptual_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def _reconstruct_single(self, x_target):
        """Reconstruct a single image with multiple restarts"""
        best_z = None
        best_loss = float('inf')
        
        for r in range(self.rec_rr):
            # Initialize z
            z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
            
            if self.use_lbfgs:
                # L-BFGS optimizer (faster convergence)
                optimizer = torch.optim.LBFGS(
                    [z], lr=0.5, max_iter=20, line_search_fn='strong_wolfe'
                )
                
                for _ in range(self.rec_iters // 20):
                    def closure():
                        optimizer.zero_grad()
                        loss = self._compute_loss(z, x_target)
                        loss.backward()
                        return loss
                    optimizer.step(closure)
            else:
                # Adam optimizer (fallback)
                optimizer = torch.optim.Adam([z], lr=0.05)
                for _ in range(self.rec_iters):
                    optimizer.zero_grad()
                    loss = self._compute_loss(z, x_target)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate final loss
            with torch.no_grad():
                final_loss = self._compute_loss(z, x_target).item()
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_z = z.detach().clone()
        
        return best_z, best_loss
    
    def reconstruct(self, x):
        """
        Reconstruct batch of images
        x: [B, 3, H, W] in [0, 1]
        """
        batch_size = x.size(0)
        
        reconstructed = []
        losses = []
        
        for i in range(batch_size):
            x_target = x[i:i+1]
            
            best_z, best_loss = self._reconstruct_single(x_target)
            losses.append(best_loss)
            
            with torch.no_grad():
                x_rec = self.generator(best_z)  # [-1, 1]
                x_rec = self._from_gan_space(x_rec)  # [0, 1]
                x_rec = x_rec.clamp(0, 1)
            
            reconstructed.append(x_rec)
        
        return torch.cat(reconstructed, dim=0), np.mean(losses)


# ========== AutoAttack用のモデルラッパー ==========
class ClassifierWrapper(nn.Module):
    """AutoAttack用に分類器をラップ
    
    AutoAttackは入力を[0, 1]範囲で期待するため、
    内部で正規化を行う
    """
    def __init__(self, classifier, device):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1))
    
    def forward(self, x):
        # x: [B, 3, H, W] in [0, 1]
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


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
    print(f"Loaded classifier from {args.clf_ckpt}")
    return classifier


def load_generator(args, device):
    ckpt = torch.load(args.gan_ckpt, map_location=device)
    
    # 設定を取得
    latent_dim = args.latent_dim
    ngf = 64
    nc = 3  # RGB
    
    if 'args' in ckpt:
        config = ckpt['args']
        latent_dim = config.get('latent_dim', latent_dim)
        ngf = config.get('ngf', ngf)
    
    generator = Generator(latent_dim=latent_dim, ngf=ngf, nc=nc).to(device)
    
    # 重みを読み込む
    if 'generator_state_dict' in ckpt:
        generator.load_state_dict(ckpt['generator_state_dict'], strict=False)
        print(f"Loaded generator (normal weights) from {args.gan_ckpt}")
        
        # EMA重みで上書き（推奨）
        if args.use_ema and 'ema_state_dict' in ckpt:
            ema_state_dict = ckpt['ema_state_dict']
            current_state = generator.state_dict()
            for name, param in ema_state_dict.items():
                if name in current_state:
                    current_state[name] = param
            generator.load_state_dict(current_state)
            print(f"Applied EMA weights")
    elif 'ema_state_dict' in ckpt:
        generator.load_state_dict(ckpt['ema_state_dict'], strict=False)
        print(f"Loaded generator (EMA) from {args.gan_ckpt}")
    
    generator.eval()
    print(f"Generator: latent_dim={latent_dim}, ngf={ngf}, nc={nc}")
    
    # エポック情報を表示
    epoch = ckpt.get('epoch', 'unknown')
    print(f"Checkpoint epoch: {epoch}")
    
    return generator, latent_dim


def load_cached_samples(path):
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    
    print(f"Loaded {len(x_test)} samples")
    print(f"Shape: {x_test.shape}, Range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    print(f"Classes: {classes}")
    
    return x_test, y_test, classes


# ========== 評価 ==========
def evaluate(model, x_test, y_test, device, batch_size=32):
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


def evaluate_with_defense(defense_gan, classifier, x_test, y_test, device, 
                          batch_size=4, desc="Defense-GAN"):
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_purified_all = []
    all_losses = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
        x_batch = x_test[i:i+batch_size].to(device)
        y_batch = y_test[i:i+batch_size].to(device)
        
        # Defense-GAN purification
        x_purified, avg_loss = defense_gan.reconstruct(x_batch)
        x_purified_all.append(x_purified.cpu())
        all_losses.append(avg_loss)
        
        # 分類
        with torch.no_grad():
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
        
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        predictions.extend(predicted.cpu().numpy())
    
    x_purified_all = torch.cat(x_purified_all, dim=0)
    print(f"  Average reconstruction loss: {np.mean(all_losses):.4f}")
    
    return correct / total, np.array(predictions), x_purified_all


def compute_l2_norm(x1, x2):
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def print_confusion_matrix(y_true, y_pred, title, classes, file=None):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def write(text):
        print(text)
        if file:
            file.write(text + '\n')
    
    write(f"\n{title}")
    write("-" * 60)
    
    header = f"{'':>15}" + "".join([f"Pred {c:>10}" for c in classes])
    write(header)
    
    for i, tc in enumerate(classes):
        row = f"True {tc:>10}" + "".join([f"{cm[i, j]:>15}" for j in range(len(classes))])
        write(row)
    
    write(f"\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {'cm': cm, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, labels, classes, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), 10)
    
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
    
    # Comparison grid
    if x_purified_adv is not None:
        comparison = []
        for i in range(min(5, n)):
            comparison.extend([x_clean[i], x_adv[i], x_purified_adv[i]])
        comparison = torch.stack(comparison)
        save_image(comparison, os.path.join(save_dir, 'comparison_clean_adv_purified.png'), nrow=3, padding=2)
    
    print(f"Saved sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M")
    defense_str = "defense_v4" if args.use_defense else "no_defense"
    eps_str = f"eps{int(args.epsilon*255)}"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}_{eps_str}_aa_{args.version}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")
    
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def log(text):
        print(text)
        results_file.write(text + '\n')
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    generator, latent_dim = load_generator(args, device)
    
    # Defense-GAN
    defense_gan = None
    if args.use_defense:
        defense_gan = DefenseGANv4(
            generator=generator,
            latent_dim=latent_dim,
            rec_iters=args.rec_iters,
            rec_rr=args.rec_rr,
            perceptual_weight=args.perceptual_weight,
            use_lbfgs=args.use_lbfgs,
            device=device
        )
        log(f"Defense-GAN v4: iters={args.rec_iters}, rr={args.rec_rr}, "
            f"perceptual_weight={args.perceptual_weight}, use_lbfgs={args.use_lbfgs}")
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # クイックテストまたはサンプル数制限
    if args.num_samples is not None:
        n_samples = min(args.num_samples, len(x_test))
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
        log(f"Using {n_samples} samples (--num_samples)")
    elif args.quick_test:
        n_samples = min(10, len(x_test))
        x_test = x_test[:n_samples]
        y_test = y_test[:n_samples]
        log(f"Quick test mode: using first {n_samples} samples")
    
    log(f"\n{'='*70}")
    log("AutoAttack + Defense-GAN v4 Evaluation (PCam)")
    log(f"{'='*70}")
    log(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    log(f"Norm: {args.norm}")
    log(f"Version: {args.version}")
    if args.attacks_to_run:
        log(f"Attacks: {args.attacks_to_run}")
    log(f"Defense: {args.use_defense}")
    log(f"Samples: {len(x_test)}")
    
    results = {}
    
    # 1. Clean accuracy
    log("\n[1/4] Clean images...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    log(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # 2. Clean + Defense
    x_purified_clean = None
    if args.use_defense:
        log("\n[2/4] Clean + Defense-GAN...")
        start = time.time()
        clean_def_acc, pred_clean_def, x_purified_clean = evaluate_with_defense(
            defense_gan, classifier, x_test, y_test, device, batch_size=2
        )
        defense_clean_time = time.time() - start
        log(f"Clean + Defense accuracy: {clean_def_acc:.4f}")
        log(f"L2 (clean vs purified): {compute_l2_norm(x_test, x_purified_clean):.4f}")
        results['clean_def_acc'] = clean_def_acc
        results['defense_clean_time'] = defense_clean_time
    
    # 3. AutoAttack
    log("\n[3/4] AutoAttack...")
    
    # ラッパーモデルを作成
    wrapped_model = ClassifierWrapper(classifier, device)
    wrapped_model.eval()
    
    # AutoAttack初期化
    adversary = AutoAttack(
        wrapped_model, 
        norm=args.norm, 
        eps=args.epsilon, 
        version=args.version,
        verbose=True
    )
    
    # 特定の攻撃のみ実行する場合
    if args.attacks_to_run:
        adversary.attacks_to_run = args.attacks_to_run
    
    start = time.time()
    
    # AutoAttackの実行
    x_test_device = x_test.to(device)
    y_test_device = y_test.to(device)
    
    x_adv = adversary.run_standard_evaluation(x_test_device, y_test_device, bs=args.batch_size)
    x_adv = x_adv.cpu()
    
    attack_time = time.time() - start
    
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    log(f"L2 (clean vs adv): {compute_l2_norm(x_test, x_adv):.4f}")
    log(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    log(f"Attack time: {attack_time:.1f}s")
    results['adv_acc'] = adv_acc
    results['attack_time'] = attack_time
    
    # 4. Adversarial + Defense
    x_purified_adv = None
    if args.use_defense:
        log("\n[4/4] Adversarial + Defense-GAN...")
        start = time.time()
        adv_def_acc, pred_adv_def, x_purified_adv = evaluate_with_defense(
            defense_gan, classifier, x_adv, y_test, device, batch_size=2
        )
        defense_adv_time = time.time() - start
        log(f"Adversarial + Defense accuracy: {adv_def_acc:.4f}")
        log(f"L2 (adv vs purified): {compute_l2_norm(x_adv, x_purified_adv):.4f}")
        log(f"Defense improvement: +{adv_def_acc - adv_acc:.4f}")
        results['adv_def_acc'] = adv_def_acc
        results['defense_adv_time'] = defense_adv_time
    
    # Final results
    log(f"\n{'='*70}")
    log("FINAL RESULTS")
    log(f"{'='*70}")
    log(f"Clean Accuracy: {results['clean_acc']:.4f}")
    if args.use_defense:
        log(f"Clean + Defense: {results['clean_def_acc']:.4f}")
    log(f"Adversarial (no defense): {results['adv_acc']:.4f}")
    if args.use_defense:
        log(f"Adversarial + Defense: {results['adv_def_acc']:.4f}")
    log(f"{'='*70}")
    
    # Confusion matrices
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "Clean", classes, results_file)
    cm_results['adv'] = print_confusion_matrix(y_true, pred_adv, "Adversarial (AutoAttack)", classes, results_file)
    if args.use_defense:
        cm_results['adv_def'] = print_confusion_matrix(y_true, pred_adv_def, "Adversarial + Defense", classes, results_file)
    
    # Save images
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(
        x_test[:10], x_adv[:10],
        x_purified_clean[:10] if x_purified_clean is not None else None,
        x_purified_adv[:10] if x_purified_adv is not None else None,
        y_test[:10], classes, samples_dir
    )
    
    results_file.close()
    
    # Save JSON
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump({
            'config': vars(args),
            'results': {k: float(v) if isinstance(v, (float, np.floating)) else v 
                       for k, v in results.items()}
        }, f, indent=2)
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
