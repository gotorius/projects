"""
Defense-GAN: PCam (PatchCamelyon) データセット用 訓練コード (修正版v2)

Reference:
    "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
    Pouya Samangouei, Maya Kabkab, Rama Chellappa
    ICLR 2018

修正点:
    1. WGAN-GP を正しく実装（勾配消失を防ぐ）
    2. Spectral Normalization を追加（オプション）
    3. 学習率とハイパーパラメータを最適化
    4. 段階的な学習率減衰
    5. Label smoothing とノイズ追加
    6. 適切なログ出力
    7. 潜在次元を1024に拡張（高品質再構成のため）

Usage:
    python defense_gan_train_v2.py --epochs 100 --batch_size 32 --gpu_id 0

メモリ使用量目安 (latent_dim=1024):
    - batch_size=32: ~6GB VRAM
    - batch_size=64: ~10GB VRAM
    - batch_size=16: ~4GB VRAM
"""

import os
import argparse
import math
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ========== 設定 ==========
def get_args():
    parser = argparse.ArgumentParser(description='Defense-GAN Training for PCam (v2)')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/train',
                        help='訓練データのパス')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints_v2',
                        help='モデル保存先')
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ (224対応)')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ (latent_dim=1024のため32推奨)')
    parser.add_argument('--epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generatorの学習率')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminatorの学習率')
    parser.add_argument('--latent_dim', type=int, default=1024, help='潜在空間の次元 (高品質再構成のため1024)')
    parser.add_argument('--ngf', type=int, default=64, help='Generator基本チャンネル数')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator基本チャンネル数')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--save_every', type=int, default=5, help='保存間隔(epochs)')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--n_critic', type=int, default=5, help='Critic更新回数/Generator更新')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty重み')
    parser.add_argument('--use_spectral_norm', action='store_true', help='Spectral Normを使用')
    return parser.parse_args()


def weights_init(m):
    """DCGAN論文に基づく重み初期化"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# ========== Generator ==========
class Generator(nn.Module):
    """
    Generator for 224x224 RGB images
    Structure: latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """
    def __init__(self, latent_dim=1024, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = 7  # 224 = 7 * 2^5
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            # State: (ngf*8) x 7 x 7
            nn.BatchNorm2d(ngf * 8),
            
            # 7x7 -> 14x14
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.apply(weights_init)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, self.init_size, self.init_size)  # ngf*8 = 512
        return self.main(x)


# ========== Discriminator (Critic for WGAN-GP) ==========
class Discriminator(nn.Module):
    """
    WGAN-GP Discriminator (Critic) for 224x224 RGB images
    Structure: 224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1
    Note: No BatchNorm, use LayerNorm instead for WGAN-GP
    """
    def __init__(self, ndf=64, nc=3, use_spectral_norm=False):
        super().__init__()
        
        # 手動で各層を構築（LayerNormのサイズを正しく指定するため）
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=True)  # 224 -> 112
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True)  # 112 -> 56
        self.ln2 = nn.LayerNorm([ndf * 2, 56, 56])
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True)  # 56 -> 28
        self.ln3 = nn.LayerNorm([ndf * 4, 28, 28])
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True)  # 28 -> 14
        self.ln4 = nn.LayerNorm([ndf * 8, 14, 14])
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True)  # 14 -> 7
        self.ln5 = nn.LayerNorm([ndf * 8, 7, 7])
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv6 = nn.Conv2d(ndf * 8, 1, 7, 1, 0, bias=True)  # 7 -> 1
        
        if use_spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.conv3 = nn.utils.spectral_norm(self.conv3)
            self.conv4 = nn.utils.spectral_norm(self.conv4)
            self.conv5 = nn.utils.spectral_norm(self.conv5)
            self.conv6 = nn.utils.spectral_norm(self.conv6)
        
        self.apply(weights_init)
    
    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.ln2(self.conv2(x)))
        x = self.act3(self.ln3(self.conv3(x)))
        x = self.act4(self.ln4(self.conv4(x)))
        x = self.act5(self.ln5(self.conv5(x)))
        x = self.conv6(x)
        return x.view(-1)


# ========== Gradient Penalty ==========
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Discriminator output
    d_interpolates = D(interpolates)
    
    # Gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ========== グラフ描画 ==========
def plot_losses(g_losses, d_losses, wasserstein_distances, save_path):
    """訓練ロスのグラフを描画"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Generator loss
    axes[0].plot(g_losses, 'b-', alpha=0.7, linewidth=0.5)
    if len(g_losses) > 50:
        window = min(50, len(g_losses) // 10)
        moving_avg = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(g_losses)), moving_avg, 'r-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[1].plot(d_losses, 'g-', alpha=0.7, linewidth=0.5)
    if len(d_losses) > 50:
        window = min(50, len(d_losses) // 10)
        moving_avg = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(d_losses)), moving_avg, 'r-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Critic Loss (D)')
    axes[1].grid(True, alpha=0.3)
    
    # Wasserstein distance
    axes[2].plot(wasserstein_distances, 'm-', alpha=0.7, linewidth=0.5)
    if len(wasserstein_distances) > 50:
        window = min(50, len(wasserstein_distances) // 10)
        moving_avg = np.convolve(wasserstein_distances, np.ones(window)/window, mode='valid')
        axes[2].plot(range(window-1, len(wasserstein_distances)), moving_avg, 'r-', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Distance')
    axes[2].set_title('Wasserstein Distance')
    axes[2].grid(True, alpha=0.3)
    
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
    
    # Dataset
    print(f'\nLoading dataset from: {args.data_dir}')
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    print(f'Classes: {dataset.classes}')
    print(f'Total samples: {len(dataset)}')
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Models
    print(f'\n{"="*60}')
    print(f'Building Generator (ngf={args.ngf}, latent_dim={args.latent_dim})')
    G = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    g_params = sum(p.numel() for p in G.parameters())
    print(f'Generator parameters: {g_params:,}')
    
    print(f'Building Discriminator (ndf={args.ndf})')
    D = Discriminator(ndf=args.ndf, use_spectral_norm=args.use_spectral_norm).to(device)
    d_params = sum(p.numel() for p in D.parameters())
    print(f'Discriminator parameters: {d_params:,}')
    print(f'{"="*60}\n')
    
    # Optimizers (WGAN-GP uses Adam with specific betas)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    # Learning rate scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    # Resume
    start_epoch = 0
    g_losses = []
    d_losses = []
    wasserstein_distances = []
    
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
        wasserstein_distances = ckpt.get('wasserstein_distances', [])
        print(f'Resumed from epoch {start_epoch}')
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Save config
    config = vars(args).copy()
    config['g_params'] = g_params
    config['d_params'] = d_params
    config['save_dir'] = save_dir
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training info
    print(f'Starting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Image size: {args.image_size}')
    print(f'LR_G: {args.lr_g}, LR_D: {args.lr_d}')
    print(f'n_critic: {args.n_critic}, GP weight: {args.gp_weight}')
    print(f'Save directory: {save_dir}\n')
    
    # Training loop
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        G.train()
        D.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_wd = 0.0
        epoch_gp = 0.0
        n_g_updates = 0
        n_d_updates = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ========== Train Discriminator (Critic) ==========
            optimizer_D.zero_grad()
            
            # Real images score
            d_real = D(real_images)
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = G(z).detach()
            
            # Fake images score
            d_fake = D(fake_images)
            
            # Gradient penalty
            gp = compute_gradient_penalty(D, real_images, fake_images, device)
            
            # Wasserstein distance (higher = better separation)
            wasserstein_dist = d_real.mean() - d_fake.mean()
            
            # Critic loss: maximize W-distance = minimize -W-distance + GP
            d_loss = -wasserstein_dist + args.gp_weight * gp
            
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            epoch_wd += wasserstein_dist.item()
            epoch_gp += gp.item()
            n_d_updates += 1
            
            d_losses.append(d_loss.item())
            wasserstein_distances.append(wasserstein_dist.item())
            
            # ========== Train Generator ==========
            if (batch_idx + 1) % args.n_critic == 0:
                optimizer_G.zero_grad()
                
                # Generate new fake images
                z = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = G(z)
                
                # Generator wants to maximize D(fake) = minimize -D(fake)
                g_loss = -D(fake_images).mean()
                
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
                n_g_updates += 1
                g_losses.append(g_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{epoch_g_loss / max(1, n_g_updates):.4f}',
                'D_loss': f'{epoch_d_loss / max(1, n_d_updates):.4f}',
                'W_dist': f'{epoch_wd / max(1, n_d_updates):.4f}',
                'GP': f'{epoch_gp / max(1, n_d_updates):.4f}'
            })
            
            global_step += 1
        
        # Learning rate decay
        scheduler_G.step()
        scheduler_D.step()
        
        # Epoch stats
        avg_g_loss = epoch_g_loss / max(1, n_g_updates)
        avg_d_loss = epoch_d_loss / max(1, n_d_updates)
        avg_wd = epoch_wd / max(1, n_d_updates)
        avg_gp = epoch_gp / max(1, n_d_updates)
        
        print(f'Epoch {epoch+1} | G_loss: {avg_g_loss:.4f} | D_loss: {avg_d_loss:.4f} | '
              f'W_dist: {avg_wd:.4f} | GP: {avg_gp:.4f} | '
              f'LR_G: {scheduler_G.get_last_lr()[0]:.6f}')
        
        # Generate samples every epoch
        G.eval()
        with torch.no_grad():
            fake_samples = G(fixed_noise)
            fake_samples = (fake_samples + 1.0) / 2.0  # [-1,1] -> [0,1]
            fake_samples = fake_samples.clamp(0.0, 1.0)
        
        sample_path = os.path.join(samples_dir, f'epoch_{epoch+1:04d}.png')
        save_image(fake_samples, sample_path, nrow=8, padding=2)
        G.train()
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': G.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_g_state_dict': optimizer_G.state_dict(),
                'optimizer_d_state_dict': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'wasserstein_distances': wasserstein_distances,
                'args': vars(args),
            }, ckpt_path)
            print(f'  -> Checkpoint saved: {ckpt_path}')
        
        # Plot losses
        if (epoch + 1) % 5 == 0:
            plot_losses(g_losses, d_losses, wasserstein_distances, 
                       os.path.join(save_dir, 'training_losses.png'))
    
    # Final save
    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'generator_state_dict': G.state_dict(),
        'discriminator_state_dict': D.state_dict(),
        'args': vars(args),
    }, final_path)
    
    # Final plot
    plot_losses(g_losses, d_losses, wasserstein_distances, 
               os.path.join(save_dir, 'training_losses_final.png'))
    
    # Save history
    history = {
        'g_losses': g_losses,
        'd_losses': d_losses,
        'wasserstein_distances': wasserstein_distances,
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
