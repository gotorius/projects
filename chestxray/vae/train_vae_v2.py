"""
VAE v2 (MagNet-style) Training Script for ChestX-ray Dataset

Reference:
"MagNet: a Two-Pronged Defense against Adversarial Examples"
Meng & Chen, ACM CCS 2017

v2 改善点:
1. Perceptual Loss (VGG特徴量ベース) - ぼやけ軽減
2. Adversarial Loss (VAE-GAN) - シャープさ向上
3. ResNet-based より深いアーキテクチャ
4. Spectral Normalization - 訓練安定化
5. β-VAE サイクリックスケジューリング
6. SSIM Loss - 構造的類似性保持
7. Multi-scale Discriminator - 異なる解像度でのリアルさ評価
8. Feature Matching Loss - より安定した訓練

実行例:
python train_vae_v2.py --epochs 300 --batch_size 32 --gpu 0

メモリ目安:
- batch_size=32: ~10GB VRAM
- batch_size=16: ~6GB VRAM
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE v2 for ChestX-ray')
    
    # モデル設定
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent space dimension')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-4,
                        help='VAE learning rate')
    parser.add_argument('--lr_disc', type=float, default=4e-4,
                        help='Discriminator learning rate')
    
    # Loss重み
    parser.add_argument('--beta_max', type=float, default=1.0,
                        help='Maximum KL weight')
    parser.add_argument('--lambda_perceptual', type=float, default=0.1,
                        help='Perceptual loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                        help='Adversarial loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=0.5,
                        help='SSIM loss weight')
    parser.add_argument('--lambda_fm', type=float, default=0.1,
                        help='Feature matching loss weight')
    
    # パス設定
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/vae/checkpoints_v2',
                        help='Output directory')
    
    # その他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save interval (epochs)')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Sample generation interval (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA for VAE')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')
    
    return parser.parse_args()


# ========== SSIM Loss ==========
def gaussian_kernel(size=11, sigma=1.5, channels=1, device='cuda'):
    """Create Gaussian kernel for SSIM"""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    kernel = kernel.expand(channels, 1, size, size)
    return kernel


def ssim_loss(x, y, kernel_size=11, sigma=1.5, reduction='mean'):
    """
    Compute SSIM loss (1 - SSIM)
    Returns value in [0, 1] where 0 = identical
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    channels = x.size(1)
    kernel = gaussian_kernel(kernel_size, sigma, channels, x.device)
    
    mu_x = F.conv2d(x, kernel, padding=kernel_size//2, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=kernel_size//2, groups=channels)
    
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.conv2d(x ** 2, kernel, padding=kernel_size//2, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, kernel, padding=kernel_size//2, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=kernel_size//2, groups=channels) - mu_xy
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    
    if reduction == 'mean':
        return 1 - ssim_map.mean()
    else:
        return 1 - ssim_map


# ========== Perceptual Loss (VGG) ==========
class PerceptualLoss(nn.Module):
    """
    VGG-based Perceptual Loss
    グレースケール画像を3チャンネルに拡張して使用
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 複数の層から特徴を抽出
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # conv1_2
            nn.Sequential(*list(vgg.features[4:9])),  # conv2_2
            nn.Sequential(*list(vgg.features[9:16])), # conv3_3
            nn.Sequential(*list(vgg.features[16:23])), # conv4_3
        ])
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # ImageNet正規化 (register_buffer を to(device) の前に)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
        self.eval()
    
    def forward(self, x, y):
        # グレースケール -> RGB
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # 正規化
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        
        return loss / len(self.blocks)


# ========== Self-Attention ==========
class SelfAttention(nn.Module):
    """Self-Attention for VAE"""
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        return self.gamma * out + x


# ========== Residual Blocks ==========
class ResBlockEncoder(nn.Module):
    """Residual Block for Encoder with optional downsampling"""
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
    """Residual Block for Decoder with optional upsampling"""
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
            h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.skip(x)
        
        return h + x


# ========== Encoder ==========
class Encoder(nn.Module):
    """
    ResNet-based Encoder with Self-Attention
    224 -> 112 -> 56 -> 28 -> 14 -> 7 -> latent
    """
    def __init__(self, img_channels=1, base_ch=64, latent_dim=512):
        super().__init__()
        
        # Initial conv
        self.conv_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        
        # Residual blocks
        self.block1 = ResBlockEncoder(base_ch, base_ch, downsample=True)       # 224 -> 112
        self.block2 = ResBlockEncoder(base_ch, base_ch * 2, downsample=True)   # 112 -> 56
        self.attention = SelfAttention(base_ch * 2)                              # Attention at 56x56
        self.block3 = ResBlockEncoder(base_ch * 2, base_ch * 4, downsample=True) # 56 -> 28
        self.block4 = ResBlockEncoder(base_ch * 4, base_ch * 8, downsample=True) # 28 -> 14
        self.block5 = ResBlockEncoder(base_ch * 8, base_ch * 8, downsample=True) # 14 -> 7
        
        self.bn_out = nn.BatchNorm2d(base_ch * 8)
        
        # Latent projections
        self.fc_mu = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.attention(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.leaky_relu(self.bn_out(h), 0.2)
        
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


# ========== Decoder ==========
class Decoder(nn.Module):
    """
    ResNet-based Decoder with Self-Attention
    latent -> 7 -> 14 -> 28 -> 56 -> 112 -> 224
    """
    def __init__(self, img_channels=1, base_ch=64, latent_dim=512):
        super().__init__()
        self.base_ch = base_ch
        
        # Latent projection
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 7 * 7)
        
        # Residual blocks
        self.block1 = ResBlockDecoder(base_ch * 8, base_ch * 8, upsample=True)   # 7 -> 14
        self.block2 = ResBlockDecoder(base_ch * 8, base_ch * 4, upsample=True)   # 14 -> 28
        self.block3 = ResBlockDecoder(base_ch * 4, base_ch * 2, upsample=True)   # 28 -> 56
        self.attention = SelfAttention(base_ch * 2)                               # Attention at 56x56
        self.block4 = ResBlockDecoder(base_ch * 2, base_ch, upsample=True)       # 56 -> 112
        self.block5 = ResBlockDecoder(base_ch, base_ch, upsample=True)           # 112 -> 224
        
        self.bn_out = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, 7, 7)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.attention(h)
        h = self.block4(h)
        h = self.block5(h)
        
        h = F.relu(self.bn_out(h))
        h = self.conv_out(h)
        
        return torch.sigmoid(h)


# ========== VAE ==========
class VAE(nn.Module):
    """VAE with ResNet architecture and Self-Attention"""
    def __init__(self, img_channels=1, base_ch=64, latent_dim=512):
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
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reconstruct(self, x):
        mu, _ = self.encoder(x)
        return self.decoder(mu)  # 再構成時はmu使用（ノイズなし）


# ========== Multi-scale Discriminator ==========
class DiscriminatorBlock(nn.Module):
    """Discriminator block with spectral normalization"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.skip = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1)) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        
        x = self.skip(x)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        
        return h + x


class Discriminator(nn.Module):
    """
    Multi-scale Discriminator for VAE-GAN
    Returns intermediate features for feature matching loss
    """
    def __init__(self, img_channels=1, base_ch=64):
        super().__init__()
        
        self.conv_in = nn.utils.spectral_norm(nn.Conv2d(img_channels, base_ch, 3, 1, 1))
        
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(base_ch, base_ch, downsample=True),       # 224 -> 112
            DiscriminatorBlock(base_ch, base_ch * 2, downsample=True),   # 112 -> 56
            DiscriminatorBlock(base_ch * 2, base_ch * 4, downsample=True), # 56 -> 28
            DiscriminatorBlock(base_ch * 4, base_ch * 8, downsample=True), # 28 -> 14
            DiscriminatorBlock(base_ch * 8, base_ch * 8, downsample=True), # 14 -> 7
        ])
        
        self.fc = nn.utils.spectral_norm(nn.Linear(base_ch * 8 * 7 * 7, 1))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        features = []
        
        h = F.leaky_relu(self.conv_in(x), 0.2)
        features.append(h)
        
        for block in self.blocks:
            h = block(h)
            features.append(h)
        
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        
        if return_features:
            return out.view(-1), features
        return out.view(-1)


# ========== EMA ==========
class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
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


# ========== データセット ==========
class ChestXrayDataset(Dataset):
    """ChestX-ray グレースケールデータセット with augmentation"""
    def __init__(self, root_dir, split='train', img_size=224, augment=True):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        self.augment = augment and (split == 'train')
        
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        
        self.image_paths = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for ext in ['*.jpeg', '*.jpg', '*.png']:
                    self.image_paths.extend(list(class_dir.glob(ext)))
        
        print(f"Loaded {len(self.image_paths)} images from {split} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# ========== 訓練関数 ==========
def train(args):
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / 'samples'
    samples_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 設定保存
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # データセット
    dataset = ChestXrayDataset(args.data_dir, split='train', img_size=args.img_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # モデル
    vae = VAE(img_channels=1, base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    disc = Discriminator(img_channels=1, base_ch=args.base_ch).to(device)
    
    # パラメータ数
    vae_params = sum(p.numel() for p in vae.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"VAE parameters: {vae_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Perceptual Loss
    perceptual_loss_fn = PerceptualLoss(device)
    
    # EMA
    if args.use_ema:
        ema = EMA(vae, decay=args.ema_decay)
        print(f"Using EMA with decay={args.ema_decay}")
    
    # オプティマイザ
    optimizer_vae = optim.Adam(vae.parameters(), lr=args.lr_vae, betas=(0.5, 0.999))
    optimizer_disc = optim.Adam(disc.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
    
    # スケジューラ
    scheduler_vae = optim.lr_scheduler.CosineAnnealingLR(optimizer_vae, T_max=args.epochs, eta_min=1e-6)
    scheduler_disc = optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=args.epochs, eta_min=1e-6)
    
    # 固定サンプル
    fixed_x = next(iter(dataloader))[:16].to(device)
    fixed_z = torch.randn(16, args.latent_dim, device=device)
    
    print(f"\n{'='*60}")
    print(f"Starting VAE v2 training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Image size: {args.img_size}")
    print(f"Latent dim: {args.latent_dim}, Base channels: {args.base_ch}")
    print(f"Loss weights: perceptual={args.lambda_perceptual}, adv={args.lambda_adv}, "
          f"ssim={args.lambda_ssim}, fm={args.lambda_fm}")
    print(f"{'='*60}\n")
    
    # 訓練履歴
    history = {
        'total_loss': [], 'recon_loss': [], 'kl_loss': [],
        'perceptual_loss': [], 'adv_loss': [], 'ssim_loss': [],
        'disc_loss': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        vae.train()
        disc.train()
        
        epoch_losses = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        # Cyclic KL annealing
        cycle_length = 50
        cycle_pos = epoch % cycle_length
        beta = args.beta_max * min(1.0, cycle_pos / (cycle_length * 0.5))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            batch_size = x.size(0)
            
            # ========== Train Discriminator ==========
            optimizer_disc.zero_grad()
            
            with torch.no_grad():
                recon_x, _, _ = vae(x)
            
            # Real/Fake discrimination
            d_real = disc(x)
            d_fake = disc(recon_x)
            
            # Hinge loss for discriminator
            disc_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            
            disc_loss.backward()
            optimizer_disc.step()
            
            epoch_losses['disc_loss'] += disc_loss.item()
            
            # ========== Train VAE ==========
            optimizer_vae.zero_grad()
            
            recon_x, mu, logvar = vae(x)
            
            # 1. Reconstruction loss (L1 + MSE)
            recon_loss = F.l1_loss(recon_x, x) + F.mse_loss(recon_x, x)
            
            # 2. KL divergence
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # 3. Perceptual loss
            perceptual = perceptual_loss_fn(recon_x, x)
            
            # 4. SSIM loss
            ssim = ssim_loss(recon_x, x)
            
            # 5. Adversarial loss (non-saturating)
            d_fake_for_g, features_fake = disc(recon_x, return_features=True)
            _, features_real = disc(x, return_features=True)
            adv_loss = -d_fake_for_g.mean()
            
            # 6. Feature matching loss
            fm_loss = 0.0
            for f_real, f_fake in zip(features_real, features_fake):
                fm_loss += F.l1_loss(f_fake, f_real.detach())
            fm_loss = fm_loss / len(features_real)
            
            # Total VAE loss
            total_loss = (recon_loss + 
                         beta * kl_loss +
                         args.lambda_perceptual * perceptual +
                         args.lambda_ssim * ssim +
                         args.lambda_adv * adv_loss +
                         args.lambda_fm * fm_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer_vae.step()
            
            # Update EMA
            if args.use_ema:
                ema.update()
            
            # Record losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['kl_loss'] += kl_loss.item()
            epoch_losses['perceptual_loss'] += perceptual.item()
            epoch_losses['ssim_loss'] += ssim.item()
            epoch_losses['adv_loss'] += adv_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss.item():.3f}',
                'rec': f'{recon_loss.item():.3f}',
                'kl': f'{kl_loss.item():.3f}',
                'β': f'{beta:.2f}'
            })
        
        # スケジューラ更新
        scheduler_vae.step()
        scheduler_disc.step()
        
        # エポック平均
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])
        
        print(f"Epoch {epoch}: total={epoch_losses['total_loss']:.4f}, "
              f"recon={epoch_losses['recon_loss']:.4f}, kl={epoch_losses['kl_loss']:.4f}, "
              f"perc={epoch_losses['perceptual_loss']:.4f}, ssim={epoch_losses['ssim_loss']:.4f}")
        
        # サンプル生成
        if epoch % args.sample_interval == 0:
            if args.use_ema:
                ema.apply_shadow()
            
            vae.eval()
            with torch.no_grad():
                # 再構成比較 (上段: オリジナル, 下段: 再構成)
                recon_fixed, _, _ = vae(fixed_x)
                comparison = torch.cat([fixed_x[:8], recon_fixed[:8]], dim=0)
                grid = make_grid(comparison, nrow=8, padding=2, normalize=True)
                save_image(grid, samples_dir / f'recon_epoch_{epoch:04d}.png')
                
                # ランダムサンプル
                samples = vae.decode(fixed_z)
                grid = make_grid(samples, nrow=4, padding=2, normalize=True)
                save_image(grid, samples_dir / f'random_epoch_{epoch:04d}.png')
            
            if args.use_ema:
                ema.restore()
            vae.train()
        
        # チェックポイント保存
        if epoch % args.save_interval == 0:
            save_dict = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'optimizer_vae': optimizer_vae.state_dict(),
                'optimizer_disc': optimizer_disc.state_dict(),
                'args': vars(args),
                'history': history
            }
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  -> Saved checkpoint at epoch {epoch}")
        
        # 最良モデル保存
        if epoch_losses['total_loss'] < best_loss:
            best_loss = epoch_losses['total_loss']
            save_dict = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'args': vars(args)
            }
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, output_dir / 'best_model.pth')
    
    # 最終モデル保存
    save_dict = {
        'epoch': args.epochs,
        'vae_state_dict': vae.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'args': vars(args),
        'history': history
    }
    if args.use_ema:
        save_dict['ema_state_dict'] = ema.state_dict()
    torch.save(save_dict, output_dir / 'final_model.pth')
    
    # 履歴保存
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 訓練曲線プロット
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*60}")


def plot_training_curves(history, save_path):
    """訓練曲線のプロット"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(history['recon_loss'])
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL loss
    axes[0, 2].plot(history['kl_loss'])
    axes[0, 2].set_title('KL Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Perceptual loss
    axes[1, 0].plot(history['perceptual_loss'])
    axes[1, 0].set_title('Perceptual Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, alpha=0.3)
    
    # SSIM loss
    axes[1, 1].plot(history['ssim_loss'])
    axes[1, 1].set_title('SSIM Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[1, 2].plot(history['disc_loss'])
    axes[1, 2].set_title('Discriminator Loss')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
