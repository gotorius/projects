"""
Defense-GAN: ChestX-ray (胸部X線) データセット用 訓練コード (v3)

Reference:
    "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
    Pouya Samangouei, Maya Kabkab, Rama Chellappa
    ICLR 2018

v3 特徴:
    1. ResNetブロックを使用した強化されたアーキテクチャ
    2. Self-Attention機構の追加（高解像度画像の長距離依存性を捉える）
    3. Exponential Moving Average (EMA) による安定化
    4. Two Time-Scale Update Rule (TTUR) - D/Gで異なる学習率
    5. R1 gradient penalty（WGAN-GPより安定）
    6. より適切なハイパーパラメータ
    7. 224x224 グレースケール画像に最適化
    8. Hinge lossの採用（安定した訓練）
    9. Orthogonal初期化

胸部X線データセット特有の考慮:
    - グレースケール (1チャンネル)
    - 医療画像特有のコントラスト
    - NORMAL/PNEUMONIA の2クラス構成

Usage:
    python defense_gan_train_v3.py --epochs 200 --batch_size 16 --gpu_id 0

メモリ使用量目安 (latent_dim=512):
    - batch_size=16: ~6GB VRAM (グレースケールなのでPCamより少ない)
    - batch_size=8: ~4GB VRAM
    - batch_size=32: ~10GB VRAM
"""

import os
import argparse
import math
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ========== 設定 ==========
def get_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Training for ChestX-ray (v3)')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='訓練データのパス (train/test フォルダを含むディレクトリ)')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ (224固定)')
    parser.add_argument('--batch_size', type=int, default=8, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=200, help='エポック数')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generatorの学習率')
    parser.add_argument('--lr_d', type=float, default=4e-4, help='Discriminatorの学習率 (TTUR: D > G)')
    parser.add_argument('--latent_dim', type=int, default=512, help='潜在空間の次元')
    parser.add_argument('--ngf', type=int, default=64, help='Generator基本チャンネル数')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator基本チャンネル数')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam beta2')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--save_every', type=int, default=10, help='保存間隔(epochs)')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--n_critic', type=int, default=1, help='Critic更新回数/Generator更新 (TTURでは1)')
    parser.add_argument('--r1_weight', type=float, default=10.0, help='R1 gradient penalty重み')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--use_ema', action='store_true', default=True, help='EMAを使用')
    parser.add_argument('--use_val', action='store_true', default=False,
                        help='validationセット(val)を使用する場合はTrue')
    return parser.parse_args()


# ========== ChestX-ray Dataset ==========
class ChestXrayDataset(Dataset):
    """ChestX-ray グレースケールデータセット"""
    def __init__(self, root_dir, split='train', img_size=224, augment=True):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        self.augment = augment
        
        # データ拡張付きの変換（訓練用）
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1] に正規化
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # クラスディレクトリを検索
        if self.root_dir.exists():
            for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
                if class_dir.is_dir():
                    self.class_names.append(class_dir.name)
                    for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                        for img_path in class_dir.glob(ext):
                            self.image_paths.append(img_path)
                            self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {split} set")
        if self.class_names:
            print(f"Classes: {self.class_names}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, label


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
        
        # Spectral normalization for stability
        self.query = nn.utils.spectral_norm(self.query)
        self.key = nn.utils.spectral_norm(self.key)
        self.value = nn.utils.spectral_norm(self.value)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # B x N x C'
        key = self.key(x).view(batch_size, -1, H * W)  # B x C' x N
        value = self.value(x).view(batch_size, -1, H * W)  # B x C x N
        
        # Attention map
        attention = torch.bmm(query, key)  # B x N x N
        attention = F.softmax(attention, dim=-1)
        
        # Weighted sum
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
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
        
        # Shortcut
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.shortcut(x)
        
        return h + x


class ResBlockDown(nn.Module):
    """Residual Block with Downsampling for Discriminator"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.conv1, self.conv2, self.shortcut]:
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        
        # Shortcut
        x = self.shortcut(x)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        
        return h + x


# ========== Generator ==========
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
        self.init_size = 7  # 224 = 7 * 2^5
        
        # Initial projection
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        # ResNet blocks with upsampling
        # 7x7 -> 14x14
        self.block1 = ResBlockUp(ngf * 8, ngf * 8)
        # 14x14 -> 28x28
        self.block2 = ResBlockUp(ngf * 8, ngf * 4)
        # 28x28 -> 56x56
        self.block3 = ResBlockUp(ngf * 4, ngf * 2)
        # Self-attention at 56x56 (good balance of resolution and computation)
        self.attention = SelfAttention(ngf * 2)
        # 56x56 -> 112x112
        self.block4 = ResBlockUp(ngf * 2, ngf)
        # 112x112 -> 224x224
        self.block5 = ResBlockUp(ngf, ngf // 2)
        
        # Final convolution (output: 1 channel for grayscale)
        self.bn_out = nn.BatchNorm2d(ngf // 2)
        self.conv_out = nn.Conv2d(ngf // 2, nc, 3, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, z):
        # Project and reshape
        h = self.fc(z)
        h = h.view(-1, 512, self.init_size, self.init_size)  # ngf*8 = 512
        
        # ResNet blocks
        h = self.block1(h)   # 7 -> 14
        h = self.block2(h)   # 14 -> 28
        h = self.block3(h)   # 28 -> 56
        h = self.attention(h)  # Self-attention
        h = self.block4(h)   # 56 -> 112
        h = self.block5(h)   # 112 -> 224
        
        # Final output
        h = self.bn_out(h)
        h = F.relu(h)
        h = self.conv_out(h)
        h = torch.tanh(h)
        
        return h


# ========== Discriminator ==========
class Discriminator(nn.Module):
    """
    ResNet-based Discriminator for 224x224 Grayscale images with Self-Attention
    Structure: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1
    Uses Spectral Normalization throughout
    
    Note: nc=1 for grayscale ChestX-ray images
    """
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        self.nc = nc
        
        # Initial convolution (no activation before first layer)
        self.conv_in = nn.utils.spectral_norm(nn.Conv2d(nc, ndf // 2, 3, 1, 1))
        
        # ResNet blocks with downsampling
        # 224x224 -> 112x112
        self.block1 = ResBlockDown(ndf // 2, ndf, downsample=True)
        # 112x112 -> 56x56
        self.block2 = ResBlockDown(ndf, ndf * 2, downsample=True)
        # Self-attention at 56x56
        self.attention = SelfAttention(ndf * 2)
        # 56x56 -> 28x28
        self.block3 = ResBlockDown(ndf * 2, ndf * 4, downsample=True)
        # 28x28 -> 14x14
        self.block4 = ResBlockDown(ndf * 4, ndf * 8, downsample=True)
        # 14x14 -> 7x7
        self.block5 = ResBlockDown(ndf * 8, ndf * 8, downsample=True)
        # 7x7 -> 7x7 (no downsample)
        self.block6 = ResBlockDown(ndf * 8, ndf * 8, downsample=False)
        
        # Final layer
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * 8, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.orthogonal_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)   # 224 -> 112
        h = self.block2(h)   # 112 -> 56
        h = self.attention(h)  # Self-attention
        h = self.block3(h)   # 56 -> 28
        h = self.block4(h)   # 28 -> 14
        h = self.block5(h)   # 14 -> 7
        h = self.block6(h)   # 7 -> 7
        
        # Global sum pooling
        h = F.relu(h)
        h = torch.sum(h, dim=[2, 3])  # Global sum pooling
        
        # Output
        out = self.fc(h)
        return out.view(-1)


# ========== R1 Gradient Penalty ==========
def compute_r1_penalty(D, real_samples):
    """
    R1 gradient penalty (more stable than WGAN-GP)
    Only penalizes gradients on real samples
    """
    real_samples.requires_grad_(True)
    d_real = D(real_samples)
    
    gradients = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_samples,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
    return gradient_penalty


# ========== Exponential Moving Average ==========
class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


# ========== Hinge Loss ==========
def hinge_loss_d(d_real, d_fake):
    """Hinge loss for Discriminator"""
    loss_real = torch.mean(F.relu(1.0 - d_real))
    loss_fake = torch.mean(F.relu(1.0 + d_fake))
    return loss_real + loss_fake


def hinge_loss_g(d_fake):
    """Hinge loss for Generator"""
    return -torch.mean(d_fake)


# ========== グラフ描画 ==========
def plot_losses(g_losses, d_losses, d_real_scores, d_fake_scores, save_path):
    """訓練ロスのグラフを描画"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generator loss
    axes[0, 0].plot(g_losses, 'b-', alpha=0.7, linewidth=0.5)
    if len(g_losses) > 50:
        window = min(100, len(g_losses) // 10)
        moving_avg = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(g_losses)), moving_avg, 'r-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Generator Loss (lower = better fake images)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[0, 1].plot(d_losses, 'g-', alpha=0.7, linewidth=0.5)
    if len(d_losses) > 50:
        window = min(100, len(d_losses) // 10)
        moving_avg = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(d_losses)), moving_avg, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Discriminator Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # D(real) scores
    axes[1, 0].plot(d_real_scores, 'c-', alpha=0.7, linewidth=0.5)
    if len(d_real_scores) > 50:
        window = min(100, len(d_real_scores) // 10)
        moving_avg = np.convolve(d_real_scores, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(d_real_scores)), moving_avg, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('D(real) - should be positive')
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # D(fake) scores
    axes[1, 1].plot(d_fake_scores, 'm-', alpha=0.7, linewidth=0.5)
    if len(d_fake_scores) > 50:
        window = min(100, len(d_fake_scores) // 10)
        moving_avg = np.convolve(d_fake_scores, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(d_fake_scores)), moving_avg, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('D(fake) - should approach 0 (from negative)')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ========== メイン訓練関数 ==========
def train(args):
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
    # Directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(save_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu_id)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9:.1f} GB')
    
    # Dataset (ChestX-ray: Grayscale, 1 channel)
    print(f'\nLoading ChestX-ray dataset from: {args.data_dir}')
    
    dataset = ChestXrayDataset(
        args.data_dir,
        split='train',
        img_size=args.image_size,
        augment=True
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No images found in {args.data_dir}/train. "
                        f"Please check the data directory structure.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Image channels (grayscale = 1)
    nc = 1
    
    # Models
    print(f'\n{"="*60}')
    print(f'Building Generator (ngf={args.ngf}, latent_dim={args.latent_dim}, nc={nc})')
    G = Generator(latent_dim=args.latent_dim, ngf=args.ngf, nc=nc).to(device)
    g_params = sum(p.numel() for p in G.parameters())
    print(f'Generator parameters: {g_params:,}')
    
    print(f'Building Discriminator (ndf={args.ndf}, nc={nc})')
    D = Discriminator(ndf=args.ndf, nc=nc).to(device)
    d_params = sum(p.numel() for p in D.parameters())
    print(f'Discriminator parameters: {d_params:,}')
    print(f'{"="*60}\n')
    
    # EMA for Generator
    if args.use_ema:
        ema = EMA(G, decay=args.ema_decay)
        print(f'Using EMA with decay={args.ema_decay}')
    
    # Optimizers (TTUR: different learning rates for D and G)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    # Learning rate scheduler (cosine annealing)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.epochs, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=args.epochs, eta_min=1e-6)
    
    # Resume
    start_epoch = 0
    g_losses = []
    d_losses = []
    d_real_scores = []
    d_fake_scores = []
    
    if args.resume:
        print(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['generator_state_dict'])
        D.load_state_dict(ckpt['discriminator_state_dict'])
        optimizer_G.load_state_dict(ckpt['optimizer_g_state_dict'])
        optimizer_D.load_state_dict(ckpt['optimizer_d_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        g_losses = ckpt.get('g_losses', [])
        d_losses = ckpt.get('d_losses', [])
        d_real_scores = ckpt.get('d_real_scores', [])
        d_fake_scores = ckpt.get('d_fake_scores', [])
        if args.use_ema and 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        print(f'Resumed from epoch {start_epoch}')
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Save config
    config = vars(args).copy()
    config['g_params'] = g_params
    config['d_params'] = d_params
    config['save_dir'] = save_dir
    config['nc'] = nc
    config['dataset'] = 'ChestX-ray'
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training info
    print(f'Starting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Image size: {args.image_size}')
    print(f'Image channels: {nc} (Grayscale)')
    print(f'LR_G: {args.lr_g}, LR_D: {args.lr_d} (TTUR)')
    print(f'n_critic: {args.n_critic}, R1 weight: {args.r1_weight}')
    print(f'Loss type: Hinge Loss')
    print(f'Save directory: {save_dir}\n')
    
    # Training loop
    global_step = 0
    best_fid = float('inf')  # For future FID tracking
    
    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        epoch_r1 = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ========== Train Discriminator ==========
            for _ in range(args.n_critic):
                optimizer_D.zero_grad()
                
                # Real images score
                d_real = D(real_images)
                
                # Generate fake images
                z = torch.randn(batch_size, args.latent_dim, device=device)
                with torch.no_grad():
                    fake_images = G(z)
                
                # Fake images score
                d_fake = D(fake_images)
                
                # Hinge loss
                d_loss = hinge_loss_d(d_real, d_fake)
                
                # R1 gradient penalty (every 16 iterations for efficiency)
                if global_step % 16 == 0:
                    r1_penalty = compute_r1_penalty(D, real_images)
                    d_loss = d_loss + args.r1_weight * r1_penalty
                    epoch_r1 += r1_penalty.item()
                
                d_loss.backward()
                optimizer_D.step()
            
            # ========== Train Generator ==========
            optimizer_G.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = G(z)
            
            # Generator loss
            d_fake_for_g = D(fake_images)
            g_loss = hinge_loss_g(d_fake_for_g)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Update EMA
            if args.use_ema:
                ema.update()
            
            # Record losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_real += d_real.mean().item()
            epoch_d_fake += d_fake.mean().item()
            n_batches += 1
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            d_real_scores.append(d_real.mean().item())
            d_fake_scores.append(d_fake.mean().item())
            
            # Update progress bar
            pbar.set_postfix({
                'G': f'{epoch_g_loss / n_batches:.3f}',
                'D': f'{epoch_d_loss / n_batches:.3f}',
                'D(r)': f'{epoch_d_real / n_batches:.2f}',
                'D(f)': f'{epoch_d_fake / n_batches:.2f}'
            })
            
            global_step += 1
        
        # Learning rate decay
        scheduler_G.step()
        scheduler_D.step()
        
        # Epoch stats
        avg_g_loss = epoch_g_loss / n_batches
        avg_d_loss = epoch_d_loss / n_batches
        avg_d_real = epoch_d_real / n_batches
        avg_d_fake = epoch_d_fake / n_batches
        
        print(f'Epoch {epoch+1} | G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | '
              f'D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f} | '
              f'LR_G: {scheduler_G.get_last_lr()[0]:.6f}')
        
        # Generate samples every 10 epochs (use EMA weights if available)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            if args.use_ema:
                ema.apply_shadow()
            
            G.eval()
            with torch.no_grad():
                fake_samples = G(fixed_noise)
                fake_samples = (fake_samples + 1.0) / 2.0  # [-1,1] -> [0,1]
                fake_samples = fake_samples.clamp(0.0, 1.0)
            
            sample_path = os.path.join(samples_dir, f'epoch_{epoch+1:04d}.png')
            save_image(fake_samples, sample_path, nrow=8, padding=2)
            print(f'  -> Samples saved: {sample_path}')
            
            if args.use_ema:
                ema.restore()
            G.train()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            save_dict = {
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_g_state_dict': optimizer_G.state_dict(),
                'optimizer_d_state_dict': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'd_real_scores': d_real_scores,
                'd_fake_scores': d_fake_scores,
                'args': vars(args),
            }
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, ckpt_path)
            print(f'  -> Checkpoint saved: {ckpt_path}')
        
        # Plot losses
        if (epoch + 1) % 5 == 0:
            plot_losses(g_losses, d_losses, d_real_scores, d_fake_scores,
                       os.path.join(save_dir, 'training_losses.png'))
    
    # Final save
    final_path = os.path.join(save_dir, 'final_model.pth')
    save_dict = {
        'epoch': args.epochs - 1,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'args': vars(args),
    }
    if args.use_ema:
        save_dict['ema_state_dict'] = ema.state_dict()
    torch.save(save_dict, final_path)
    
    # Final plot
    plot_losses(g_losses, d_losses, d_real_scores, d_fake_scores,
               os.path.join(save_dir, 'training_losses_final.png'))
    
    # Save history
    history = {
        'g_losses': g_losses,
        'd_losses': d_losses,
        'd_real_scores': d_real_scores,
        'd_fake_scores': d_fake_scores,
    }
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Final model saved to: {final_path}')
    print(f'Samples saved to: {samples_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    args = get_args()
    train(args)
