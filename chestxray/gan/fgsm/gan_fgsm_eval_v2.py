"""
Defense-GAN Adversarial Defense Evaluation for ChestX-ray - FGSM Attack (v2 Improved)

改善点:
1. L-BFGS最適化（より高速で正確な収束）
2. Perceptual Loss（VGGベース）
3. 潜在空間の正則化
4. より効率的な再構成アルゴリズム
5. 複数初期化の並列処理

実行例:
python gan_fgsm_eval_v2.py --epsilon 0.031 --use_defense
python gan_fgsm_eval_v2.py --epsilon 0.031 --rec_iters 200 --rec_rr 5
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


# ========== 引数 ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Evaluation v2')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255)
    
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
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/correct_samples_500.pt')
    parser.add_argument('--gan_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints/20251228_234231/final_model.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/fgsm/results')
    
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


# ========== Generator ==========
class Generator(nn.Module):
    def __init__(self, latent_dim=512, ngf=64, nc=1):
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


# ========== 改良版 Defense-GAN ==========
class DefenseGANv2:
    """
    Defense-GAN with improved reconstruction:
    1. L-BFGS optimizer for faster convergence
    2. Perceptual loss for better quality
    3. Latent space regularization
    """
    def __init__(self, generator, latent_dim=512, rec_iters=200, rec_rr=5,
                 perceptual_weight=0.1, use_lbfgs=True, device='cuda'):
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
    
    def _rgb_to_gray(self, x):
        """RGB -> Gray"""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    
    def _gray_to_rgb(self, x):
        """Gray -> RGB"""
        return x.repeat(1, 3, 1, 1)
    
    def _compute_loss(self, z, x_target_gray, x_target_rgb=None):
        """Total reconstruction loss"""
        x_gen = self.generator(z)  # [B, 1, H, W] in [-1, 1]
        
        # MSE loss in grayscale space
        x_target_tanh = x_target_gray * 2 - 1  # [0,1] -> [-1,1]
        mse_loss = F.mse_loss(x_gen, x_target_tanh)
        
        # Perceptual loss (optional)
        if self.vgg is not None and x_target_rgb is not None and self.perceptual_weight > 0:
            x_gen_01 = (x_gen + 1) / 2  # [-1,1] -> [0,1]
            x_gen_rgb = self._gray_to_rgb(x_gen_01)  # [B, 3, H, W]
            
            feat_gen = self.vgg(x_gen_rgb)
            feat_target = self.vgg(x_target_rgb)
            perceptual_loss = F.mse_loss(feat_gen, feat_target)
            
            total_loss = mse_loss + self.perceptual_weight * perceptual_loss
        else:
            total_loss = mse_loss
        
        return total_loss
    
    def _reconstruct_single(self, x_target_gray, x_target_rgb):
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
                        loss = self._compute_loss(z, x_target_gray, x_target_rgb)
                        loss.backward()
                        return loss
                    optimizer.step(closure)
            else:
                # Adam optimizer (fallback)
                optimizer = torch.optim.Adam([z], lr=0.05)
                for _ in range(self.rec_iters):
                    optimizer.zero_grad()
                    loss = self._compute_loss(z, x_target_gray, x_target_rgb)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate final loss
            with torch.no_grad():
                final_loss = self._compute_loss(z, x_target_gray, x_target_rgb).item()
            
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
        
        # Convert to grayscale
        x_gray = self._rgb_to_gray(x)  # [B, 1, H, W] in [0, 1]
        
        reconstructed = []
        losses = []
        
        for i in range(batch_size):
            x_target_gray = x_gray[i:i+1]
            x_target_rgb = x[i:i+1]
            
            best_z, best_loss = self._reconstruct_single(x_target_gray, x_target_rgb)
            losses.append(best_loss)
            
            with torch.no_grad():
                x_rec = self.generator(best_z)  # [-1, 1]
                x_rec = (x_rec + 1) / 2  # [0, 1]
                x_rec = x_rec.clamp(0, 1)
                x_rec = self._gray_to_rgb(x_rec)  # [B, 3, H, W]
            
            reconstructed.append(x_rec)
        
        return torch.cat(reconstructed, dim=0), np.mean(losses)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    return classifier


def load_generator(args, device):
    ckpt = torch.load(args.gan_ckpt, map_location=device)
    
    # 設定を取得
    latent_dim = args.latent_dim
    ngf = 64
    nc = 1
    
    if 'args' in ckpt:
        config = ckpt['args']
        latent_dim = config.get('latent_dim', latent_dim)
        ngf = config.get('ngf', ngf)
    
    generator = Generator(latent_dim=latent_dim, ngf=ngf, nc=nc).to(device)
    
    # 注意: EMA重みはBatchNormの統計が壊れている可能性があるため、通常の重みを優先
    if 'generator_state_dict' in ckpt:
        generator.load_state_dict(ckpt['generator_state_dict'], strict=False)
        print(f"Loaded generator (normal weights) from {args.gan_ckpt}")
    elif 'ema_state_dict' in ckpt:
        generator.load_state_dict(ckpt['ema_state_dict'], strict=False)
        print(f"Loaded generator (EMA - may have BN issues) from {args.gan_ckpt}")
    
    generator.eval()
    print(f"Generator: latent_dim={latent_dim}, ngf={ngf}, nc={nc}")
    
    return generator, latent_dim


def load_cached_samples(path):
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = ['NORMAL', 'PNEUMONIA']
    
    print(f"Loaded {len(x_test)} samples")
    print(f"Shape: {x_test.shape}, Range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().to(device)
    x.requires_grad = True
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


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
    # labelsを指定して、全クラスが含まれるようにする
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
    defense_str = "defense_v2" if args.use_defense else "no_defense"
    eps_str = f"eps{int(args.epsilon*255)}"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}_{eps_str}")
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
        defense_gan = DefenseGANv2(
            generator=generator,
            latent_dim=latent_dim,
            rec_iters=args.rec_iters,
            rec_rr=args.rec_rr,
            perceptual_weight=args.perceptual_weight,
            use_lbfgs=args.use_lbfgs,
            device=device
        )
        log(f"Defense-GAN v2: iters={args.rec_iters}, rr={args.rec_rr}, "
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
    log("FGSM Attack + Defense-GAN v2 Evaluation")
    log(f"{'='*70}")
    log(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
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
    
    # 3. FGSM attack
    log("\n[3/4] FGSM attack...")
    start = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start
    
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    log(f"L2 (clean vs adv): {compute_l2_norm(x_test, x_adv):.4f}")
    log(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
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
    cm_results['adv'] = print_confusion_matrix(y_true, pred_adv, "Adversarial", classes, results_file)
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
