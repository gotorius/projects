"""
VAE (MagNet-style) Training Script for DermMel Dataset

============================================================
参考論文・技術一覧:
============================================================
1. MagNet: "MagNet: a Two-Pronged Defense against Adversarial Examples"
   - Meng & Chen, ACM CCS 2017
   - https://arxiv.org/abs/1705.09064

2. VAE: "Auto-Encoding Variational Bayes"
   - Kingma & Welling, ICLR 2014
   - https://arxiv.org/abs/1312.6114

3. β-VAE: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
   - Higgins et al., ICLR 2017
   - https://openreview.net/forum?id=Sy2fzU9gl

4. VAE-GAN: "Autoencoding beyond pixels using a learned similarity metric"
   - Larsen et al., ICML 2016
   - https://arxiv.org/abs/1512.09300

5. Perceptual Loss: "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
   - Johnson et al., ECCV 2016
   - https://arxiv.org/abs/1603.08155

6. SSIM: "Image Quality Assessment: From Error Visibility to Structural Similarity"
   - Wang et al., IEEE TIP 2004
   - https://ieeexplore.ieee.org/document/1284395

7. Spectral Normalization: "Spectral Normalization for Generative Adversarial Networks"
   - Miyato et al., ICLR 2018
   - https://arxiv.org/abs/1802.05957

8. Cyclical Annealing: "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
   - Fu et al., NAACL 2019
   - https://arxiv.org/abs/1903.10145

9. Feature Matching: "Improved Techniques for Training GANs"
   - Salimans et al., NeurIPS 2016
   - https://arxiv.org/abs/1606.03498

============================================================
DermMel データセット情報:
============================================================
- 画像サイズ: 224x224
- チャンネル: RGB (3チャンネル)
- クラス: 2クラス (Melanoma / NotMelanoma)
- データパス: /mnt/data1/Public/MedImages/DermMel/

============================================================
VAEアーキテクチャの特徴:
============================================================
- VAE-GAN: Discriminatorによる敵対的損失で鮮明な再構成
- Perceptual Loss: VGG16特徴量でのL1損失
- SSIM Loss: 構造的類似性を保持
- β-VAE: KL項の重みを調整可能（再構成重視）
- Cyclical Annealing: KL vanishing問題の緩和
- Feature Matching: Discriminator特徴量のマッチング
- Edge Loss: エッジ保存のための損失
- Spectral Normalization: Discriminatorの安定化
- EMA: パラメータの指数移動平均

目的: MagNet方式の敵対的防御（adversarial examplesの浄化）

============================================================
実行例:
============================================================
python train_vae.py --epochs 300 --batch_size 8 --gpu 0

メモリ目安:
- batch_size=4:  ~6GB VRAM
- batch_size=8:  ~10GB VRAM
- batch_size=16: ~18GB VRAM
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
    parser = argparse.ArgumentParser(description='Train VAE for DermMel')
    
    # モデル設定
    parser.add_argument('--latent_dim', type=int, default=512,
                        help='Latent space dimension')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size (224 for ResNet compatibility)')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr_vae', type=float, default=1e-4,
                        help='VAE learning rate')
    parser.add_argument('--lr_disc', type=float, default=1e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Learning rate warmup epochs')
    
    # Loss重み (再構成重視に調整)
    parser.add_argument('--beta_max', type=float, default=0.01,
                        help='Maximum KL weight (lower = better reconstruction)')
    parser.add_argument('--lambda_perceptual', type=float, default=2.0,
                        help='Perceptual loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.02,
                        help='Adversarial loss weight')
    parser.add_argument('--lambda_ssim', type=float, default=2.0,
                        help='SSIM loss weight')
    parser.add_argument('--lambda_fm', type=float, default=0.1,
                        help='Feature matching loss weight')
    parser.add_argument('--lambda_edge', type=float, default=1.0,
                        help='Edge preservation loss weight')
    
    # パス設定
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/DermMel',
                        help='Data directory (should contain train/test folders)')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/vae/checkpoints',
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
                        help='Use Automatic Mixed Precision')
    
    return parser.parse_args()


# ========== Edge Loss ==========
class EdgeLoss(nn.Module):
    """Sobel-based Edge Preservation Loss"""
    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # Expand for 3 channels (RGB)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).expand(3, 1, 3, 3).clone())
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).expand(3, 1, 3, 3).clone())
    
    def forward(self, x, y):
        # Compute edges
        edge_x_x = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        edge_x_y = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        edge_y_x = F.conv2d(y, self.sobel_x, padding=1, groups=3)
        edge_y_y = F.conv2d(y, self.sobel_y, padding=1, groups=3)
        
        # Edge magnitude
        edge_x = torch.sqrt(edge_x_x ** 2 + edge_x_y ** 2 + 1e-8)
        edge_y = torch.sqrt(edge_y_x ** 2 + edge_y_y ** 2 + 1e-8)
        
        return F.l1_loss(edge_x, edge_y)


# ========== SSIM Loss ==========
class SSIMLoss(nn.Module):
    """SSIM Loss for RGB images"""
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
    
    def forward(self, x, y):
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
        
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()


# ========== Perceptual Loss ==========
class PerceptualLoss(nn.Module):
    """VGG Perceptual Loss for RGB images"""
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 浅い層のみ使用 (メモリ節約 + 低レベル特徴重視)
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # conv1_2 (64ch)
            nn.Sequential(*list(vgg.features[4:9])),  # conv2_2 (128ch)
            nn.Sequential(*list(vgg.features[9:16])), # conv3_3 (256ch)
        ])
        
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
        self.eval()
        
        # 各層の重み
        self.weights = [1.0, 0.5, 0.25]
    
    def forward(self, x, y):
        # 正規化
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


# ========== Residual Blocks ==========
class ResBlockEncoder(nn.Module):
    """Residual Block for Encoder"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
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
        
        return h + x


class ResBlockDecoder(nn.Module):
    """Residual Block for Decoder"""
    def __init__(self, in_ch, out_ch, upsample=True):
        super().__init__()
        self.upsample = upsample
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
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
        
        return h + x


# ========== Encoder (RGB入力) ==========
class Encoder(nn.Module):
    """Encoder for RGB images: 224 -> 7 -> latent"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        
        self.conv_in = nn.Conv2d(img_channels, base_ch, 3, 1, 1)
        
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.block1 = ResBlockEncoder(base_ch, base_ch, downsample=True)
        self.block2 = ResBlockEncoder(base_ch, base_ch * 2, downsample=True)
        self.block3 = ResBlockEncoder(base_ch * 2, base_ch * 4, downsample=True)
        self.block4 = ResBlockEncoder(base_ch * 4, base_ch * 8, downsample=True)
        self.block5 = ResBlockEncoder(base_ch * 8, base_ch * 8, downsample=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch * 8)
        
        # Latent projections
        self.fc_mu = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
    
    def forward(self, x):
        h = self.conv_in(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.leaky_relu(self.bn_out(h), 0.2)
        
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


# ========== Decoder (RGB出力) ==========
class Decoder(nn.Module):
    """Decoder for RGB images: latent -> 7 -> 224"""
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        super().__init__()
        self.base_ch = base_ch
        
        self.fc = nn.Linear(latent_dim, base_ch * 8 * 7 * 7)
        
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.block1 = ResBlockDecoder(base_ch * 8, base_ch * 8, upsample=True)
        self.block2 = ResBlockDecoder(base_ch * 8, base_ch * 4, upsample=True)
        self.block3 = ResBlockDecoder(base_ch * 4, base_ch * 2, upsample=True)
        self.block4 = ResBlockDecoder(base_ch * 2, base_ch, upsample=True)
        self.block5 = ResBlockDecoder(base_ch, base_ch, upsample=True)
        
        self.bn_out = nn.BatchNorm2d(base_ch)
        self.conv_out = nn.Conv2d(base_ch, img_channels, 3, 1, 1)
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, 7, 7)
        
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        
        h = F.relu(self.bn_out(h))
        h = self.conv_out(h)
        
        return torch.sigmoid(h)


# ========== VAE ==========
class VAE(nn.Module):
    """VAE for RGB images"""
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
        """推論時用: 平均値を使用して再構成（ノイズなし）"""
        mu, _ = self.encoder(x)
        return self.decoder(mu)


# ========== Lightweight Discriminator ==========
class Discriminator(nn.Module):
    """Lightweight Discriminator with Spectral Normalization"""
    def __init__(self, img_channels=3, base_ch=32):
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
    """Exponential Moving Average for model parameters"""
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
    
    # データ変換
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size + 16, args.img_size + 16)),
        transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # データセット (ImageFolder形式)
    # DermMelは train_sep フォルダを使用
    train_data_dir = os.path.join(args.data_dir, 'train_sep')
    print(f"Loading training data from {train_data_dir}...")
    train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transform)
    print(f"Classes: {train_dataset.classes}")
    print(f"Total training samples: {len(train_dataset)}")
    
    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # モデル
    vae = VAE(img_channels=3, base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    disc = Discriminator(img_channels=3, base_ch=32).to(device)
    
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
    fixed_batch = next(iter(dataloader))
    if isinstance(fixed_batch, (list, tuple)):
        fixed_x = fixed_batch[0][:16].to(device)
    else:
        fixed_x = fixed_batch[:16].to(device)
    fixed_z = torch.randn(16, args.latent_dim, device=device)
    
    print(f"\n{'='*70}")
    print(f"Starting VAE training for DermMel - {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Grad accumulation: {args.grad_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.grad_accumulation}")
    print(f"Latent dim: {args.latent_dim}, Base channels: {args.base_ch}")
    print(f"Loss weights: β_max={args.beta_max}, perc={args.lambda_perceptual}, "
          f"adv={args.lambda_adv}, ssim={args.lambda_ssim}, edge={args.lambda_edge}")
    print(f"{'='*70}\n")
    
    for epoch in range(start_epoch, args.epochs + 1):
        vae.train()
        disc.train()
        
        epoch_losses = {k: 0.0 for k in history.keys()}
        n_batches = 0
        
        # Cyclic KL annealing (longer cycle)
        cycle_length = 100
        cycle_pos = epoch % cycle_length
        beta = args.beta_max * min(1.0, cycle_pos / (cycle_length * 0.3))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        optimizer_vae.zero_grad()
        optimizer_disc.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # ImageFolderは(images, labels)を返す
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device, non_blocking=True)
            else:
                x = batch.to(device, non_blocking=True)
            
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
            
            # Total loss
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
                'β': f'{beta:.3f}'
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
                recon_fixed, _, _ = vae(fixed_x)
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
        
        # 最良モデル
        if epoch_losses['total_loss'] < best_loss:
            best_loss = epoch_losses['total_loss']
            save_dict = {'epoch': epoch, 'vae_state_dict': vae.state_dict(), 'args': vars(args)}
            if args.use_ema:
                save_dict['ema_state_dict'] = ema.state_dict()
            torch.save(save_dict, output_dir / 'best_model.pth')
        
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
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*70}")


def plot_training_curves(history, save_path):
    n_metrics = len(history)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flat
    
    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
    
    # 余分なサブプロットを非表示
    for ax in axes[len(history):]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    train(args)
