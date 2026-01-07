"""
VAE v2 (MagNet-style) Training Script for PCam - 再構成品質改善版

============================================================
v1からの主な改善点:
============================================================
1. latent_dim: 256 → 512 (情報量増加)
2. base_ch: 48 → 64 (ネットワーク容量増加)
3. β_max: 0.1 → 0.01 (KL正則化を弱め再構成優先)
4. U-Net風Skip Connection追加 (細部保持)
5. Self-Attention層追加 (構造的一貫性)
6. より強いPerceptual Loss
7. Laplacian Pyramid Loss追加 (エッジ保持)

============================================================
実行例:
============================================================
python train_vae_v2.py --epochs 300 --batch_size 16 --gpu 0

v1から再開する場合:
python train_vae_v2.py --epochs 300 --batch_size 16 --gpu 0 --from_scratch
"""

import os
import sys
import argparse
import random
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Train VAE v2 for PCam (improved reconstruction)')
    
    # モデル設定 (改善版)
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent space dimension (increased from 256)')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base number of channels (increased from 48)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (224 for ResNet compatibility)')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-4,
                        help='VAE learning rate (reduced for stability)')
    parser.add_argument('--lr_disc', type=float, default=1e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Learning rate warmup epochs')
    
    # Loss重み (再構成重視に大幅調整)
    parser.add_argument('--beta_max', type=float, default=0.01,
                        help='Maximum KL weight (greatly reduced for better reconstruction)')
    parser.add_argument('--lambda_perceptual', type=float, default=2.0,
                        help='Perceptual loss weight (increased)')
    parser.add_argument('--lambda_adv', type=float, default=0.02,
                        help='Adversarial loss weight (reduced)')
    parser.add_argument('--lambda_ssim', type=float, default=2.0,
                        help='SSIM loss weight (increased)')
    parser.add_argument('--lambda_fm', type=float, default=0.1,
                        help='Feature matching loss weight')
    parser.add_argument('--lambda_edge', type=float, default=1.0,
                        help='Edge (Laplacian) loss weight')
    
    # パス設定
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/train',
                        help='Training data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/vae/checkpoints_v2',
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
    parser.add_argument('--grad_accumulation', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use automatic mixed precision')
    
    return parser.parse_args()


# ========== SSIM Loss ==========
class SSIMLoss(nn.Module):
    """Multi-scale SSIM Loss for RGB images"""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))
    
    def _create_window(self, size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)
        return window
    
    def ssim(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        channels = x.size(1)
        window = self.window.expand(channels, 1, -1, -1).to(x.device)
        
        mu_x = F.conv2d(x, window, padding=self.window_size//2, groups=channels)
        mu_y = F.conv2d(y, window, padding=self.window_size//2, groups=channels)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x ** 2, window, padding=self.window_size//2, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, window, padding=self.window_size//2, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size//2, groups=channels) - mu_xy
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, x, y):
        # Multi-scale SSIM
        loss = 0.0
        weights = [0.5, 0.3, 0.2]
        
        for i, w in enumerate(weights):
            if i > 0:
                x = F.avg_pool2d(x, 2)
                y = F.avg_pool2d(y, 2)
            loss += w * (1 - self.ssim(x, y))
        
        return loss


# ========== Edge Loss (Laplacian) ==========
class EdgeLoss(nn.Module):
    """Laplacian Edge Loss for preserving fine details"""
    def __init__(self):
        super().__init__()
        # Laplacian kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('kernel', kernel)
    
    def forward(self, x, y):
        # Convert to grayscale
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        y_gray = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]
        
        # Apply Laplacian
        x_edge = F.conv2d(x_gray, self.kernel.to(x.device), padding=1)
        y_edge = F.conv2d(y_gray, self.kernel.to(y.device), padding=1)
        
        return F.l1_loss(x_edge, y_edge)


# ========== Perceptual Loss (Enhanced) ==========
class PerceptualLoss(nn.Module):
    """Enhanced VGG Perceptual Loss with more layers"""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # より多くの層を使用
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # conv1_2 (64ch)
            nn.Sequential(*list(vgg.features[4:9])),  # conv2_2 (128ch)
            nn.Sequential(*list(vgg.features[9:18])), # conv3_4 (256ch)
            nn.Sequential(*list(vgg.features[18:27])),# conv4_4 (512ch)
        ])
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
        self.eval()
        
        # 浅い層により大きな重み（テクスチャ重視）
        self.weights = [1.0, 0.75, 0.5, 0.25]
    
    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0.0
        x_feat = x
        y_feat = y
        
        for i, block in enumerate(self.blocks):
            x_feat = block(x_feat)
            with torch.no_grad():
                y_feat = block(y_feat)
            loss += self.weights[i] * F.l1_loss(x_feat, y_feat.detach())
        
        return loss


# ========== Self-Attention ==========
class SelfAttention(nn.Module):
    """Self-Attention for capturing long-range dependencies"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        
        attn = F.softmax(torch.bmm(q, k) / (C // 8) ** 0.5, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        
        return self.gamma * out + x


# ========== Residual Blocks ==========
class ResBlockEncoder(nn.Module):
    """Residual Block for Encoder with optional attention"""
    def __init__(self, in_ch, out_ch, downsample=True, use_attention=False):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        self.attention = SelfAttention(out_ch) if use_attention else None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = F.leaky_relu(self.bn1(x), 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(self.bn2(h), 0.2)
        h = self.conv2(h)
        
        x = self.skip(x)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        
        out = h + x
        
        if self.attention is not None:
            out = self.attention(out)
        
        return out


class ResBlockDecoder(nn.Module):
    """Residual Block for Decoder with optional attention"""
    def __init__(self, in_ch, out_ch, upsample=True, use_attention=False):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
        self.attention = SelfAttention(out_ch) if use_attention else None
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(m.bias)
    
    def forward(self, x):
        h = F.relu(self.bn1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.skip(x)
        
        out = h + x
        
        if self.attention is not None:
            out = self.attention(out)
        
        return out


# ========== U-Net風 Encoder ==========
class Encoder(nn.Module):
    """Encoder with skip connections for U-Net style VAE"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        
        self.conv_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.block1 = ResBlockEncoder(base_ch, base_ch, downsample=True)
        self.block2 = ResBlockEncoder(base_ch, base_ch * 2, downsample=True)
        self.block3 = ResBlockEncoder(base_ch * 2, base_ch * 4, downsample=True, use_attention=True)
        self.block4 = ResBlockEncoder(base_ch * 4, base_ch * 8, downsample=True)
        self.block5 = ResBlockEncoder(base_ch * 8, base_ch * 8, downsample=True, use_attention=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch * 8)
        
        # Latent projections
        self.fc_mu = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
    
    def forward(self, x):
        h = self.conv_in(x)
        
        # Save skip features
        skip1 = h                    # 224, base_ch
        h = self.block1(h)
        skip2 = h                    # 112, base_ch
        h = self.block2(h)
        skip3 = h                    # 56, base_ch*2
        h = self.block3(h)
        skip4 = h                    # 28, base_ch*4
        h = self.block4(h)
        skip5 = h                    # 14, base_ch*8
        h = self.block5(h)           # 7, base_ch*8
        
        h = F.leaky_relu(self.bn_out(h), 0.2)
        
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        return mu, logvar, [skip1, skip2, skip3, skip4, skip5]


# ========== U-Net風 Decoder ==========
class Decoder(nn.Module):
    """Decoder with skip connections for U-Net style VAE"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.base_ch = base_ch
        
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 7 * 7)
        
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        # Skip connections: 入力チャンネル数が2倍になる
        self.block1 = ResBlockDecoder(base_ch * 8 + base_ch * 8, base_ch * 8, upsample=True, use_attention=True)
        self.block2 = ResBlockDecoder(base_ch * 8 + base_ch * 4, base_ch * 4, upsample=True)
        self.block3 = ResBlockDecoder(base_ch * 4 + base_ch * 2, base_ch * 2, upsample=True, use_attention=True)
        self.block4 = ResBlockDecoder(base_ch * 2 + base_ch, base_ch, upsample=True)
        self.block5 = ResBlockDecoder(base_ch + base_ch, base_ch, upsample=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
    
    def forward(self, z, skips=None):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, 7, 7)  # 7x7
        
        if skips is not None:
            skip1, skip2, skip3, skip4, skip5 = skips
            
            # 各skipをhと同じサイズにリサイズしてから結合
            # h=7x7, skip5=14x14 → 7x7にリサイズ
            skip5_resized = F.interpolate(skip5, size=(7, 7), mode='bilinear', align_corners=False)
            h = torch.cat([h, skip5_resized], dim=1)
            h = self.block1(h)  # 7 -> 14
            
            # h=14x14, skip4=28x28 → 14x14にリサイズ
            skip4_resized = F.interpolate(skip4, size=(14, 14), mode='bilinear', align_corners=False)
            h = torch.cat([h, skip4_resized], dim=1)
            h = self.block2(h)  # 14 -> 28
            
            # h=28x28, skip3=56x56 → 28x28にリサイズ
            skip3_resized = F.interpolate(skip3, size=(28, 28), mode='bilinear', align_corners=False)
            h = torch.cat([h, skip3_resized], dim=1)
            h = self.block3(h)  # 28 -> 56
            
            # h=56x56, skip2=112x112 → 56x56にリサイズ
            skip2_resized = F.interpolate(skip2, size=(56, 56), mode='bilinear', align_corners=False)
            h = torch.cat([h, skip2_resized], dim=1)
            h = self.block4(h)  # 56 -> 112
            
            # h=112x112, skip1=224x224 → 112x112にリサイズ
            skip1_resized = F.interpolate(skip1, size=(112, 112), mode='bilinear', align_corners=False)
            h = torch.cat([h, skip1_resized], dim=1)
            h = self.block5(h)  # 112 -> 224
        else:
            # No skip connections (for random sampling)
            # ゼロテンソルでskipを埋める
            B = z.size(0)
            device = z.device
            
            dummy_skip = torch.zeros(B, self.base_ch * 8, 7, 7, device=device)
            h = torch.cat([h, dummy_skip], dim=1)
            h = self.block1(h)  # 7 -> 14
            
            dummy_skip = torch.zeros(B, self.base_ch * 4, 14, 14, device=device)
            h = torch.cat([h, dummy_skip], dim=1)
            h = self.block2(h)  # 14 -> 28
            
            dummy_skip = torch.zeros(B, self.base_ch * 2, 28, 28, device=device)
            h = torch.cat([h, dummy_skip], dim=1)
            h = self.block3(h)  # 28 -> 56
            
            dummy_skip = torch.zeros(B, self.base_ch, 56, 56, device=device)
            h = torch.cat([h, dummy_skip], dim=1)
            h = self.block4(h)  # 56 -> 112
            
            dummy_skip = torch.zeros(B, self.base_ch, 112, 112, device=device)
            h = torch.cat([h, dummy_skip], dim=1)
            h = self.block5(h)  # 112 -> 224
        
        h = F.relu(self.bn_out(h))
        h = self.conv_out(h)
        
        return torch.sigmoid(h)


# ========== VAE ==========
class VAE(nn.Module):
    """U-Net style VAE for high-quality reconstruction"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(img_channels, base_ch, latent_dim)
        self.decoder = Decoder(img_channels, base_ch, latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, skips)
        return recon, mu, logvar
    
    def encode(self, x):
        mu, logvar, _ = self.encoder(x)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z):
        return self.decoder(z, skips=None)
    
    def reconstruct(self, x):
        """Deterministic reconstruction using mean only"""
        mu, _, skips = self.encoder(x)
        return self.decoder(mu, skips)


# ========== Lightweight Discriminator ==========
class Discriminator(nn.Module):
    """Lightweight PatchGAN Discriminator"""
    def __init__(self, img_channels=3, base_ch=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 224 -> 112
            nn.utils.spectral_norm(nn.Conv2d(img_channels, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112 -> 56
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56 -> 28
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28 -> 14
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14 -> 7
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.utils.spectral_norm(nn.Linear(base_ch * 8 * 7 * 7, 1))
    
    def forward(self, x, return_features=False):
        features = []
        h = x
        
        for i, layer in enumerate(self.main):
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                features.append(h)
        
        h = h.view(h.size(0), -1)
        out = self.fc(h)
        
        if return_features:
            return out.view(-1), features
        return out.view(-1)


# ========== EMA ==========
class EMA:
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
        return {k: v.cpu() for k, v in self.shadow.items()}
    
    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k] = v.to(self.shadow[k].device)


# ========== データセット ==========
def get_dataloader(data_dir, img_size, batch_size, num_workers):
    """PCam ImageFolder からデータローダを作成"""
    transform = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Loaded {len(dataset)} images from {data_dir}")
    print(f"Classes: {dataset.classes}")
    
    return dataloader


# ========== 学習率スケジューラ ==========
def get_lr_lambda(warmup_epochs, total_epochs):
    """Warmup + Cosine Annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return lr_lambda


# ========== メモリ解放 ==========
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    
    # データローダ
    dataloader = get_dataloader(args.data_dir, args.img_size, args.batch_size, args.num_workers)
    
    # モデル (RGB画像用: img_channels=3)
    vae = VAE(img_channels=3, base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    disc = Discriminator(img_channels=3, base_ch=64).to(device)
    
    # パラメータ数
    vae_params = sum(p.numel() for p in vae.parameters())
    disc_params = sum(p.numel() for p in disc.parameters())
    print(f"VAE parameters: {vae_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Loss関数
    ssim_loss_fn = SSIMLoss().to(device)
    perceptual_loss_fn = PerceptualLoss(device)
    edge_loss_fn = EdgeLoss().to(device)
    
    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(vae, decay=args.ema_decay)
        print(f"Using EMA with decay={args.ema_decay}")
    
    # オプティマイザ
    optimizer_vae = optim.AdamW(vae.parameters(), lr=args.lr_vae, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_disc = optim.AdamW(disc.parameters(), lr=args.lr_disc, betas=(0.9, 0.999), weight_decay=0.01)
    
    # スケジューラ
    lr_lambda = get_lr_lambda(args.warmup_epochs, args.epochs)
    scheduler_vae = optim.lr_scheduler.LambdaLR(optimizer_vae, lr_lambda)
    scheduler_disc = optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda)
    
    # AMP
    scaler = GradScaler() if args.use_amp else None
    
    # Resume
    start_epoch = 1
    best_loss = float('inf')
    history = {
        'total_loss': [], 'recon_loss': [], 'kl_loss': [],
        'perceptual_loss': [], 'ssim_loss': [], 'edge_loss': [], 'disc_loss': []
    }
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        vae.load_state_dict(ckpt['vae_state_dict'])
        disc.load_state_dict(ckpt['disc_state_dict'])
        optimizer_vae.load_state_dict(ckpt['optimizer_vae'])
        optimizer_disc.load_state_dict(ckpt['optimizer_disc'])
        start_epoch = ckpt['epoch'] + 1
        history = ckpt.get('history', history)
        if args.use_ema and 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
    
    # 固定サンプル
    fixed_x, _ = next(iter(dataloader))
    fixed_x = fixed_x[:16].to(device)
    fixed_z = torch.randn(16, args.latent_dim, device=device)
    
    print(f"\n{'='*70}")
    print(f"Starting VAE v2 training for PCam dataset...")
    print(f"{'='*70}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}, Grad accumulation: {args.grad_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.grad_accumulation}")
    print(f"Image size: {args.img_size}x{args.img_size}, Channels: RGB (3)")
    print(f"Latent dim: {args.latent_dim}, Base channels: {args.base_ch}")
    print(f"Loss weights: β_max={args.beta_max}, perc={args.lambda_perceptual}, "
          f"adv={args.lambda_adv}, ssim={args.lambda_ssim}, edge={args.lambda_edge}")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        vae.train()
        disc.train()
        
        epoch_losses = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        # KL annealing (緩やかに上昇、より低い最大値)
        # epoch 50までに最大値に到達
        beta = args.beta_max * min(1.0, epoch / 50)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        optimizer_vae.zero_grad()
        optimizer_disc.zero_grad()
        
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            
            # ========== Train Discriminator ==========
            with torch.no_grad():
                recon_x, _, _ = vae(x)
            
            d_real = disc(x)
            d_fake = disc(recon_x)
            disc_loss = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()) / args.grad_accumulation
            
            disc_loss.backward()
            
            if (batch_idx + 1) % args.grad_accumulation == 0:
                nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                optimizer_disc.step()
                optimizer_disc.zero_grad()
            
            # ========== Train VAE ==========
            recon_x, mu, logvar = vae(x)
            
            # Losses
            recon_loss = F.l1_loss(recon_x, x) + F.mse_loss(recon_x, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            ssim = ssim_loss_fn(recon_x, x)
            perceptual = perceptual_loss_fn(recon_x, x)
            edge = edge_loss_fn(recon_x, x)
            
            # Adversarial loss
            d_fake_for_g, features_fake = disc(recon_x, return_features=True)
            with torch.no_grad():
                _, features_real = disc(x, return_features=True)
            adv_loss = -d_fake_for_g.mean()
            
            # Feature matching
            fm_loss = sum(F.l1_loss(f, r.detach()) for f, r in zip(features_fake, features_real)) / len(features_real)
            
            # Total loss (再構成重視)
            total_loss = (recon_loss + 
                         beta * kl_loss +
                         args.lambda_perceptual * perceptual +
                         args.lambda_ssim * ssim +
                         args.lambda_edge * edge +
                         args.lambda_adv * adv_loss +
                         args.lambda_fm * fm_loss) / args.grad_accumulation
            
            total_loss.backward()
            
            if (batch_idx + 1) % args.grad_accumulation == 0:
                nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
                optimizer_vae.step()
                optimizer_vae.zero_grad()
                
                if args.use_ema:
                    ema.update()
            
            # Record
            epoch_losses['total_loss'] += total_loss.item() * args.grad_accumulation
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['kl_loss'] += kl_loss.item()
            epoch_losses['perceptual_loss'] += perceptual.item()
            epoch_losses['ssim_loss'] += ssim.item()
            epoch_losses['edge_loss'] += edge.item()
            epoch_losses['disc_loss'] += disc_loss.item() * args.grad_accumulation
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss.item() * args.grad_accumulation:.3f}',
                'rec': f'{recon_loss.item():.3f}',
                'ssim': f'{ssim.item():.3f}',
                'β': f'{beta:.4f}'
            })
            
            # 定期的なメモリ解放
            if batch_idx % 100 == 0:
                clear_memory()
        
        # スケジューラ更新
        scheduler_vae.step()
        scheduler_disc.step()
        
        # エポック平均
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
            history[k].append(epoch_losses[k])
        
        current_lr = scheduler_vae.get_last_lr()[0]
        print(f"Epoch {epoch}: total={epoch_losses['total_loss']:.4f}, "
              f"recon={epoch_losses['recon_loss']:.4f}, kl={epoch_losses['kl_loss']:.4f}, "
              f"ssim={epoch_losses['ssim_loss']:.4f}, edge={epoch_losses['edge_loss']:.4f}, lr={current_lr:.6f}")
        
        # サンプル生成
        if epoch % args.sample_interval == 0:
            if args.use_ema:
                ema.apply_shadow()
            
            vae.eval()
            with torch.no_grad():
                # Deterministic reconstruction (using reconstruct method)
                recon_fixed = vae.reconstruct(fixed_x)
                comparison = torch.cat([fixed_x[:8], recon_fixed[:8]], dim=0)
                save_image(comparison, samples_dir / f'recon_epoch_{epoch:04d}.png', nrow=8, padding=2)
                
                samples = vae.decode(fixed_z)
                save_image(samples, samples_dir / f'random_epoch_{epoch:04d}.png', nrow=4, padding=2)
            
            if args.use_ema:
                ema.restore()
            vae.train()
            
            clear_memory()
        
        # チェックポイント
        if epoch % args.save_interval == 0:
            save_dict = {
                'epoch': epoch,
                'vae_state_dict': vae.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'optimizer_vae': optimizer_vae.state_dict(),
                'optimizer_disc': optimizer_disc.state_dict(),
                'history': history,
                'args': vars(args)
            }
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  -> Saved checkpoint at epoch {epoch}")
        
        # 最良モデル (recon_loss基準)
        if epoch_losses['recon_loss'] < best_loss:
            best_loss = epoch_losses['recon_loss']
            save_dict = {'epoch': epoch, 'vae_state_dict': vae.state_dict(), 'args': vars(args)}
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, output_dir / 'best_model.pth')
            print(f"  -> New best model (recon={best_loss:.4f})")
        
        clear_memory()
    
    # 最終保存
    torch.save({
        'epoch': args.epochs,
        'vae_state_dict': vae.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'history': history,
        'args': vars(args),
        'ema_state_dict': ema.state_dict() if args.use_ema else None
    }, output_dir / 'final_model.pth')
    
    # 履歴保存
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # プロット
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Best recon loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*70}")


def plot_training_curves(history, save_path):
    n_plots = len(history)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    
    # 未使用のサブプロットを非表示
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
