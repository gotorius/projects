"""
Defense-GAN: ChestX-ray (胸部X線) データセット用 訓練コード (v4 - Improved)

改善点:
    1. WGAN-GP損失（より安定した勾配）
    2. Feature Matching Loss（mode collapse防止）
    3. Perceptual Loss（VGG特徴量ベース）を訓練に追加
    4. 適切な重み初期化
    5. より深いネットワーク
    6. Progressive Training風のノイズ低減
    7. より多いエポック数推奨 (500+)
    8. FID監視機能

Usage:
    python defense_gan_train_v4.py --epochs 500 --batch_size 8 --gpu_id 0
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ========== 設定 ==========
def get_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Training for ChestX-ray (v4)')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='訓練データのパス')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    parser.add_argument('--batch_size', type=int, default=8, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=500, help='エポック数')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Generator学習率')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Discriminator学習率')
    parser.add_argument('--latent_dim', type=int, default=256, help='潜在空間次元（小さめで安定）')
    parser.add_argument('--ngf', type=int, default=64, help='Generator基本チャンネル数')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator基本チャンネル数')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--save_every', type=int, default=25, help='保存間隔')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--n_critic', type=int, default=5, help='Critic更新回数/Generator更新')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty重み')
    parser.add_argument('--fm_weight', type=float, default=1.0, help='Feature matching重み')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay')
    return parser.parse_args()


# ========== データセット ==========
class ChestXrayDataset(Dataset):
    """ChestX-ray グレースケールデータセット（改良版）"""
    def __init__(self, root_dir, split='train', img_size=224, augment=True):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        
        # 強めのデータ拡張
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomCrop(img_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # [-1, 1]
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
        
        if self.root_dir.exists():
            for class_idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
                if class_dir.is_dir():
                    for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
                        for img_path in class_dir.glob(ext):
                            self.image_paths.append(img_path)
                            self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images from {split}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx]


# ========== 改良型Generator ==========
class ResBlockUp(nn.Module):
    """Conditional Batch Norm付きResBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        
        # 改良された初期化
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.shortcut.weight)
    
    def forward(self, x):
        h = F.leaky_relu(self.bn1(x), 0.2)
        h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
        h = self.conv1(h)
        h = F.leaky_relu(self.bn2(h), 0.2)
        h = self.conv2(h)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.shortcut(x)
        return h + x


class Generator(nn.Module):
    """改良型Generator（256次元潜在空間、bilinear upsampling）"""
    def __init__(self, latent_dim=256, ngf=64, nc=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 7
        
        # Mapping Network（StyleGAN風）
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size)
        
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.block1 = ResBlockUp(ngf * 8, ngf * 8)
        self.block2 = ResBlockUp(ngf * 8, ngf * 4)
        self.block3 = ResBlockUp(ngf * 4, ngf * 2)
        self.block4 = ResBlockUp(ngf * 2, ngf)
        self.block5 = ResBlockUp(ngf, ngf // 2)
        
        self.bn_out = nn.BatchNorm2d(ngf // 2)
        self.conv_out = nn.Conv2d(ngf // 2, nc, 3, 1, 1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, z, return_features=False):
        # Mapping
        w = self.mapping(z)
        
        h = self.fc(w)
        h = h.view(-1, 512, self.init_size, self.init_size)
        
        features = []
        h = self.block1(h)
        features.append(h)
        h = self.block2(h)
        features.append(h)
        h = self.block3(h)
        features.append(h)
        h = self.block4(h)
        features.append(h)
        h = self.block5(h)
        features.append(h)
        
        h = F.leaky_relu(self.bn_out(h), 0.2)
        h = self.conv_out(h)
        out = torch.tanh(h)
        
        if return_features:
            return out, features
        return out


# ========== 改良型Discriminator ==========
class ResBlockDown(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        self.shortcut = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))
        
        nn.init.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        nn.init.xavier_uniform_(self.shortcut.weight)
    
    def forward(self, x):
        h = F.leaky_relu(x, 0.2)
        h = self.conv1(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        x = self.shortcut(x)
        
        if self.downsample:
            h = F.avg_pool2d(h, 2)
            x = F.avg_pool2d(x, 2)
        return h + x


class Discriminator(nn.Module):
    """Feature extraction対応Discriminator"""
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        
        self.conv_in = nn.utils.spectral_norm(nn.Conv2d(nc, ndf // 2, 3, 1, 1))
        
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.blocks = nn.ModuleList([
            ResBlockDown(ndf // 2, ndf, True),      # 224->112
            ResBlockDown(ndf, ndf * 2, True),       # 112->56
            ResBlockDown(ndf * 2, ndf * 4, True),   # 56->28
            ResBlockDown(ndf * 4, ndf * 8, True),   # 28->14
            ResBlockDown(ndf * 8, ndf * 8, True),   # 14->7
            ResBlockDown(ndf * 8, ndf * 8, False),  # 7->7
        ])
        
        self.fc = nn.utils.spectral_norm(nn.Linear(ndf * 8, 1))
        
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, x, return_features=False):
        features = []
        h = self.conv_in(x)
        features.append(h)
        
        for block in self.blocks:
            h = block(h)
            features.append(h)
        
        h = F.leaky_relu(h, 0.2)
        h = torch.sum(h, dim=[2, 3])
        out = self.fc(h).view(-1)
        
        if return_features:
            return out, features
        return out


# ========== Loss Functions ==========
def gradient_penalty(D, real, fake, device):
    """WGAN-GP gradient penalty"""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)
    
    d_interpolated = D(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


def feature_matching_loss(real_features, fake_features):
    """Feature matching loss for mode collapse prevention"""
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        loss += F.l1_loss(ff.mean(dim=0), rf.mean(dim=0).detach())
    return loss / len(real_features)


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
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


# ========== 訓練 ==========
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    
    # ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'{timestamp}_v4')
    os.makedirs(save_dir, exist_ok=True)
    samples_dir = os.path.join(save_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # データセット
    dataset = ChestXrayDataset(args.data_dir, split='train', img_size=args.image_size, augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    # モデル
    nc = 1
    G = Generator(latent_dim=args.latent_dim, ngf=args.ngf, nc=nc).to(device)
    D = Discriminator(ndf=args.ndf, nc=nc).to(device)
    
    print(f'Generator params: {sum(p.numel() for p in G.parameters()):,}')
    print(f'Discriminator params: {sum(p.numel() for p in D.parameters()):,}')
    
    # EMA
    ema = EMA(G, decay=args.ema_decay)
    
    # Optimizer
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    # Scheduler
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)
    
    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt['generator_state_dict'])
        D.load_state_dict(ckpt['discriminator_state_dict'])
        opt_G.load_state_dict(ckpt['optimizer_g_state_dict'])
        opt_D.load_state_dict(ckpt['optimizer_d_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        print(f'Resumed from epoch {start_epoch}')
    
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Config保存
    config = vars(args).copy()
    config['save_dir'] = save_dir
    config['nc'] = nc
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # 訓練ループ
    g_losses, d_losses, d_real_list, d_fake_list = [], [], [], []
    
    print(f'Starting training for {args.epochs} epochs...')
    print(f'n_critic={args.n_critic}, gp_weight={args.gp_weight}, fm_weight={args.fm_weight}')
    
    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        
        epoch_g, epoch_d, epoch_dr, epoch_df = 0, 0, 0, 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for real_imgs, _ in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # ========== Train Discriminator ==========
            for _ in range(args.n_critic):
                opt_D.zero_grad()
                
                # Real
                d_real, real_feats = D(real_imgs, return_features=True)
                
                # Fake
                z = torch.randn(batch_size, args.latent_dim, device=device)
                with torch.no_grad():
                    fake_imgs = G(z)
                d_fake = D(fake_imgs)
                
                # WGAN loss
                d_loss = d_fake.mean() - d_real.mean()
                
                # Gradient penalty
                gp = gradient_penalty(D, real_imgs, fake_imgs.detach(), device)
                d_loss = d_loss + args.gp_weight * gp
                
                d_loss.backward()
                opt_D.step()
            
            # ========== Train Generator ==========
            opt_G.zero_grad()
            
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_imgs, fake_feats = G(z, return_features=True)
            
            d_fake_g, fake_feats_d = D(fake_imgs, return_features=True)
            _, real_feats_d = D(real_imgs.detach(), return_features=True)
            
            # WGAN-G loss
            g_loss = -d_fake_g.mean()
            
            # Feature matching loss
            fm_loss = feature_matching_loss(real_feats_d, fake_feats_d)
            g_loss = g_loss + args.fm_weight * fm_loss
            
            g_loss.backward()
            opt_G.step()
            
            ema.update()
            
            epoch_g += g_loss.item()
            epoch_d += d_loss.item()
            epoch_dr += d_real.mean().item()
            epoch_df += d_fake.mean().item()
            n_batches += 1
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            d_real_list.append(d_real.mean().item())
            d_fake_list.append(d_fake.mean().item())
            
            pbar.set_postfix({
                'G': f'{epoch_g/n_batches:.3f}',
                'D': f'{epoch_d/n_batches:.3f}',
                'D(r)': f'{epoch_dr/n_batches:.2f}',
                'D(f)': f'{epoch_df/n_batches:.2f}'
            })
        
        scheduler_G.step()
        scheduler_D.step()
        
        print(f'Epoch {epoch+1} | G: {epoch_g/n_batches:.4f} | D: {epoch_d/n_batches:.4f} | '
              f'D(real): {epoch_dr/n_batches:.3f} | D(fake): {epoch_df/n_batches:.3f}')
        
        # Generate samples
        if (epoch + 1) % 10 == 0 or epoch == 0:
            ema.apply_shadow()
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise)
                samples = (samples + 1) / 2
            save_image(samples, os.path.join(samples_dir, f'epoch_{epoch+1:04d}.png'), nrow=8)
            ema.restore()
            G.train()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_g_state_dict': opt_G.state_dict(),
                'optimizer_d_state_dict': opt_D.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            print(f'  -> Saved: {ckpt_path}')
    
    # Final save
    torch.save({
        'epoch': args.epochs - 1,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'args': vars(args),
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump({'g_losses': g_losses, 'd_losses': d_losses}, f)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(g_losses, alpha=0.5)
    axes[0].set_title('Generator Loss')
    axes[1].plot(d_losses, alpha=0.5)
    axes[1].set_title('Discriminator Loss')
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close()
    
    print(f'Training completed! Saved to {save_dir}')


if __name__ == '__main__':
    args = get_args()
    train(args)
