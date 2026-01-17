"""
Defense-GAN 浄化効果の視覚的確認 (ChestX-ray用)

少数の画像を使って、Defense-GANが敵対的摂動をどの程度除去できているかを
視覚的に確認するスクリプト

出力:
- クリーン画像
- 敵対的画像（FGSM攻撃後）
- 浄化済み敵対的画像（Defense-GAN適用後）
をグリッド表示して保存

使用例:
python gan_fgsm_visual_check.py --num_samples 10 --epsilon 0.031 --use_defense
python gan_fgsm_visual_check.py --num_samples 5 --epsilon 0.031 --rec_iters 500
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Visual Check of Defense-GAN Purification (ChestX-ray)')
    
    # 表示・確認設定
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    
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
                        default='/mnt/data1/gotou/projects/chestxray/gan/fgsm/visual_check',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--gpu', type=int, default=2,
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


# ========== Generator for ChestX-ray ==========
class Generator(nn.Module):
    """
    ResNet-based Generator for 224x224 Grayscale images with Self-Attention
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


# ========== Defense-GAN Purification ==========
class DefenseGAN:
    """Defense-GAN for ChestX-ray"""
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
        """RGB to grayscale"""
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray
    
    def _gray_to_rgb(self, x):
        """Grayscale to RGB (replicate channels)"""
        return x.repeat(1, 3, 1, 1)
    
    def _to_tanh_space(self, x):
        """[0,1] -> [-1,1]"""
        return x * 2 - 1
    
    def _from_tanh_space(self, x):
        """[-1,1] -> [0,1]"""
        return (x + 1) / 2
    
    def reconstruct(self, x):
        """Reconstruct image using GAN"""
        batch_size = x.size(0)
        
        x_gray = self._rgb_to_gray(x)
        x_target = self._to_tanh_space(x_gray)
        
        best_z_list = [None] * batch_size
        best_loss_list = [float('inf')] * batch_size
        
        for r in range(self.rec_rr):
            z = torch.randn(batch_size, self.latent_dim, device=self.device, requires_grad=True)
            optimizer = torch.optim.Adam([z], lr=self.rec_lr, betas=(0.9, 0.999))
            
            for _ in range(self.rec_iters):
                optimizer.zero_grad()
                
                x_gen = self.generator(z)
                loss = F.mse_loss(x_gen, x_target, reduction='none')
                loss = loss.view(batch_size, -1).mean(dim=1)
                
                total_loss = loss.sum()
                total_loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                x_gen = self.generator(z)
                final_loss = F.mse_loss(x_gen, x_target, reduction='none')
                final_loss = final_loss.view(batch_size, -1).mean(dim=1)
                
                for i in range(batch_size):
                    if final_loss[i].item() < best_loss_list[i]:
                        best_loss_list[i] = final_loss[i].item()
                        best_z_list[i] = z[i].clone()
        
        best_z = torch.stack([z if z is not None else torch.randn(self.latent_dim, device=self.device) 
                             for z in best_z_list])
        
        with torch.no_grad():
            x_rec = self.generator(best_z)
            x_rec = self._from_tanh_space(x_rec)
            x_rec = x_rec.clamp(0, 1)
            x_rec = self._gray_to_rgb(x_rec)
        
        return x_rec


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """Load classifier"""
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"✓ Loaded classifier")
    
    return classifier


def load_generator(args, device):
    """Load GAN generator"""
    checkpoint = torch.load(args.gan_ckpt, map_location=device)
    
    latent_dim = 512
    ngf = 64
    
    if 'args' in checkpoint:
        config = checkpoint['args']
        latent_dim = config.get('latent_dim', 512)
        ngf = config.get('ngf', 64)
    
    generator = Generator(latent_dim=latent_dim, ngf=ngf, nc=1).to(device)
    
    if 'ema_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['ema_state_dict'], strict=False)
    elif 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
    else:
        try:
            generator.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            pass
    
    generator.eval()
    print(f"✓ Loaded generator (latent_dim={latent_dim}, ngf={ngf})")
    
    return generator, latent_dim


# ========== データ読み込み ==========
def load_cached_samples(path, num_samples=10):
    """Load cached samples"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test'][:num_samples]
    y_test = data['y_test'][:num_samples]
    
    classes = ['NORMAL', 'PNEUMONIA']
    
    print(f"✓ Loaded {len(x_test)} samples")
    print(f"  - NORMAL: {sum(y_test==0).item()}")
    print(f"  - PNEUMONIA: {sum(y_test==1).item()}")
    
    return x_test, y_test, classes


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM attack"""
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


# ========== 分類 ==========
def classify(model, x, device):
    """Classify images"""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        x_norm = (x.to(device) - mean) / std
        outputs = model(x_norm)
        _, predicted = outputs.max(1)
    
    return predicted.cpu().numpy()


# ========== 画像保存ユーティリティ ==========
def create_comparison_grid(x_clean, x_adv, x_purified, y_true, classes):
    """Create comparison grid: [clean, adversarial, purified] for each sample"""
    n = len(x_clean)
    
    grids = []
    for i in range(n):
        # 3列 x 1行のグリッド (clean, adv, purified)
        grid = make_grid(
            [x_clean[i:i+1], x_adv[i:i+1], x_purified[i:i+1] if x_purified is not None else x_adv[i:i+1]],
            nrow=3,
            padding=4,
            pad_value=1.0  # White border
        )
        grids.append(grid)
    
    # すべてのグリッドを縦に結合
    combined = torch.cat([g.unsqueeze(0) for g in grids], dim=2)
    
    return combined


def save_detailed_comparison(x_clean, x_adv, x_purified, y_true, pred_clean, pred_adv, 
                            pred_purified, classes, save_dir):
    """Save detailed comparison images"""
    os.makedirs(save_dir, exist_ok=True)
    
    n = len(x_clean)
    
    print("\n" + "="*70)
    print("Individual Sample Comparison")
    print("="*70)
    
    for i in range(n):
        true_class = classes[y_true[i].item()]
        pred_clean_class = classes[pred_clean[i]]
        pred_adv_class = classes[pred_adv[i]]
        pred_purified_class = classes[pred_purified[i]] if pred_purified is not None else "?"
        
        # 画像をリストに追加（各画像は [C, H, W] 形式）
        images_to_grid = [x_clean[i].unsqueeze(0), x_adv[i].unsqueeze(0)]
        if x_purified is not None:
            images_to_grid.append(x_purified[i].unsqueeze(0))
        
        # リストをテンソルに変換してmake_gridに渡す [B, C, H, W]
        images_tensor = torch.cat(images_to_grid, dim=0)
        grid = make_grid(images_tensor, nrow=len(images_to_grid), padding=4, pad_value=1.0)
        
        filename = f'sample_{i:02d}_{true_class}.png'
        save_image(grid, os.path.join(save_dir, filename))
        
        status_clean = "✓" if pred_clean[i] == y_true[i].item() else "✗"
        status_adv = "✓" if pred_adv[i] == y_true[i].item() else "✗"
        status_purified = "✓" if (pred_purified is not None and pred_purified[i] == y_true[i].item()) else ("✗" if pred_purified is not None else "-")
        
        print(f"\nSample {i}: {true_class} (ID={y_true[i].item()})")
        print(f"  Clean         : Pred={pred_clean_class:10s} {status_clean}")
        print(f"  Adversarial   : Pred={pred_adv_class:10s} {status_adv}")
        if x_purified is not None:
            print(f"  Purified      : Pred={pred_purified_class:10s} {status_purified}")
    
    print("="*70)


def save_grid_images(x_clean, x_adv, x_purified, y_true, classes, save_dir):
    """Save grid images"""
    os.makedirs(save_dir, exist_ok=True)
    
    n = len(x_clean)
    
    # 全クリーン画像
    grid_clean = make_grid(x_clean, nrow=5, padding=4, pad_value=1.0)
    save_image(grid_clean, os.path.join(save_dir, '1_all_clean.png'))
    print(f"✓ Saved grid of clean images")
    
    # 全敵対的画像
    grid_adv = make_grid(x_adv, nrow=5, padding=4, pad_value=1.0)
    save_image(grid_adv, os.path.join(save_dir, '2_all_adversarial.png'))
    print(f"✓ Saved grid of adversarial images")
    
    # 全浄化画像
    if x_purified is not None:
        grid_purified = make_grid(x_purified, nrow=5, padding=4, pad_value=1.0)
        save_image(grid_purified, os.path.join(save_dir, '3_all_purified.png'))
        print(f"✓ Saved grid of purified images")


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
    print(f"Using device: {device}\n")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%m%d%H%M")
    defense_str = "with_defense" if args.use_defense else "no_defense"
    eps_str = f"eps{int(args.epsilon*255)}"
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{defense_str}_{eps_str}_n{args.num_samples}")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Output directory: {log_dir}\n")
    
    # モデル読み込み
    print("Loading models...")
    classifier = load_classifier(args, device)
    generator, latent_dim = load_generator(args, device)
    
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
        print(f"✓ Defense-GAN enabled (iters={args.rec_iters}, lr={args.rec_lr}, rr={args.rec_rr})\n")
    
    # データ読み込み
    print("Loading data...")
    x_test, y_test, classes = load_cached_samples(args.cached_samples, args.num_samples)
    print()
    
    # クリーン画像の分類
    print("Classifying clean images...")
    pred_clean = classify(classifier, x_test, device)
    clean_acc = (pred_clean == y_test.numpy()).mean()
    print(f"✓ Clean accuracy: {clean_acc:.2%}\n")
    
    # FGSM攻撃
    print(f"Running FGSM attack (epsilon={args.epsilon:.4f}, {args.epsilon*255:.1f}/255)...")
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), 4), desc="FGSM"):
        x_batch = x_test[i:i+4]
        y_batch = y_test[i:i+4]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    
    pred_adv = classify(classifier, x_adv, device)
    adv_acc = (pred_adv == y_test.numpy()).mean()
    print(f"✓ Adversarial accuracy (no defense): {adv_acc:.2%}\n")
    
    # Defense-GAN浄化
    x_purified = None
    pred_purified = None
    if args.use_defense:
        print("Applying Defense-GAN purification...")
        x_purified_list = []
        for i in tqdm(range(0, len(x_adv), 2), desc="Purifying"):
            x_batch = x_adv[i:i+2].to(device)
            x_purified_batch = defense_gan.reconstruct(x_batch)
            x_purified_list.append(x_purified_batch.cpu())
        x_purified = torch.cat(x_purified_list, dim=0)
        
        pred_purified = classify(classifier, x_purified, device)
        purified_acc = (pred_purified == y_test.numpy()).mean()
        print(f"✓ Adversarial accuracy (with Defense-GAN): {purified_acc:.2%}\n")
    
    # 画像保存
    print("Saving visualization images...")
    samples_dir = os.path.join(log_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    save_detailed_comparison(x_test, x_adv, x_purified, y_test, pred_clean, pred_adv, 
                            pred_purified, classes, samples_dir)
    
    save_grid_images(x_test, x_adv, x_purified, y_test, classes, samples_dir)
    
    # サマリー出力
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Dataset: ChestX-ray (samples={args.num_samples})")
    print(f"Attack: FGSM (epsilon={args.epsilon:.4f}, {args.epsilon*255:.1f}/255)")
    print(f"Defense: Defense-GAN (enabled={args.use_defense})")
    print(f"-"*70)
    print(f"Clean Accuracy:                    {clean_acc:.2%}")
    print(f"Adversarial Accuracy (no defense): {adv_acc:.2%}")
    if x_purified is not None:
        print(f"Adversarial Accuracy (w/ defense): {purified_acc:.2%}")
        print(f"Defense Improvement:               {purified_acc - adv_acc:+.2%}")
    print(f"-"*70)
    print(f"Output directory: {log_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
