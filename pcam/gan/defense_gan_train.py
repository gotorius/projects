"""
Defense-GAN: PCam (PatchCamelyon) データセット用 訓練コード

Reference:
    "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
    Pouya Samangouei, Maya Kabkab, Rama Chellappa
    ICLR 2018

Defense-GANは、入力画像をGANの生成器を通じて「浄化」することで、
敵対的摂動を除去する防御手法です。

Usage:
    python defense_gan_train.py --epochs 100 --batch_size 64
"""

import os
import argparse
import math
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='Defense-GAN Training for PCam')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/train',
                        help='訓練データのパス')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/gan/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    parser.add_argument('--batch_size', type=int, default=128, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Generatorの学習率')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Discriminatorの学習率')
    parser.add_argument('--latent_dim', type=int, default=128, help='潜在空間の次元')
    parser.add_argument('--ngf', type=int, default=64, help='Generator基本チャンネル数')
    parser.add_argument('--ndf', type=int, default=64, help='Discriminator基本チャンネル数')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta1 (default=0 for WGAN)')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta2 (default=0.9 for WGAN)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--save_every', type=int, default=10, help='保存間隔(epochs)')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--n_critic', type=int, default=5, help='Discriminator更新回数/Generator更新')
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty重み')
    parser.add_argument('--gan_type', type=str, default='wgan-gp', choices=['wgan-gp', 'dcgan'],
                        help='GAN訓練方式 (wgan-gp推奨)')
    parser.add_argument('--use_wgan', action='store_true', default=True, 
                        help='WGAN-GPを使用 (推奨)')
    return parser.parse_args()


# ========== Generator (DCGAN-based) ==========
class Generator(nn.Module):
    """
    DCGAN-based Generator for 224x224 images
    latent_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """
    def __init__(self, latent_dim=128, ngf=64, nc=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial projection: latent -> (ngf*16) x 7 x 7
        self.init_size = 7
        self.fc = nn.Linear(latent_dim, ngf * 16 * self.init_size * self.init_size)
        
        self.main = nn.Sequential(
            # 7x7 -> 14x14
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # 14x14 -> 28x28
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # 28x28 -> 56x56
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # 56x56 -> 112x112
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # 112x112 -> 224x224
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, self.init_size, self.init_size)
        return self.main(x)


# ========== Discriminator (DCGAN-based) ==========
class Discriminator(nn.Module):
    """
    DCGAN-based Discriminator for 224x224 images
    224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1
    """
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.main(x).view(-1, 1)


# ========== WGAN-GP (optional, better stability) ==========
class WGANDiscriminator(nn.Module):
    """
    WGAN-GP Discriminator (no sigmoid, no BatchNorm)
    """
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        
        self.main = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            nn.LayerNorm([ndf * 2, 56, 56]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.LayerNorm([ndf * 4, 28, 28]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.LayerNorm([ndf * 8, 14, 14]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=True),
            nn.LayerNorm([ndf * 16, 7, 7]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=True),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        return self.main(x).view(-1, 1)


# ========== Gradient Penalty for WGAN-GP ==========
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ========== グラフ描画 ==========
def plot_losses(g_losses, d_losses, save_dir):
    """訓練ロスのグラフを描画"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Generator loss
    axes[0].plot(g_losses, 'b-', alpha=0.7, linewidth=0.5, label='G Loss')
    if len(g_losses) > 100:
        window = min(100, len(g_losses) // 10)
        moving_avg = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(g_losses)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Discriminator loss
    axes[1].plot(d_losses, 'g-', alpha=0.7, linewidth=0.5, label='D Loss')
    if len(d_losses) > 100:
        window = min(100, len(d_losses) // 10)
        moving_avg = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(d_losses)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Both losses
    axes[2].plot(g_losses, 'b-', alpha=0.5, linewidth=0.5, label='G Loss')
    axes[2].plot(d_losses, 'g-', alpha=0.5, linewidth=0.5, label='D Loss')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('G and D Losses')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
    plt.close()
    print(f'Loss plot saved to: {os.path.join(save_dir, "training_loss.png")}')


# ========== メイン訓練関数 ==========
def train(args):
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Directories
    os.makedirs(args.save_dir, exist_ok=True)
    samples_dir = os.path.join(args.save_dir, 'samples')
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
    print(f'\nBuilding Generator (ngf={args.ngf}, latent_dim={args.latent_dim})')
    generator = Generator(latent_dim=args.latent_dim, ngf=args.ngf).to(device)
    print(f'Generator parameters: {sum(p.numel() for p in generator.parameters()):,}')
    
    # WGAN-GPの場合はWGANDiscriminatorを使用
    print(f'Building Discriminator (ndf={args.ndf}) - WGAN-GP mode')
    discriminator = WGANDiscriminator(ndf=args.ndf).to(device)
    print(f'Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}')
    
    # Optimizers (WGAN-GPにはAdamを使用、beta1=0が重要)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    
    # Resume
    start_epoch = 0
    g_losses = []
    d_losses = []
    
    if args.resume:
        print(f'\nResuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        generator.load_state_dict(ckpt['generator_state_dict'])
        discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        g_losses = ckpt.get('g_losses', [])
        d_losses = ckpt.get('d_losses', [])
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.latent_dim, device=device)
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Image size: {args.image_size}')
    print(f'LR_G: {args.lr_g}, LR_D: {args.lr_d}')
    
    # Wasserstein距離を追跡
    wasserstein_distances = []
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        running_g_loss = 0.0
        running_d_loss = 0.0
        running_wd = 0.0
        running_gp = 0.0
        n_g_updates = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (real_images, _) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ========== Train Discriminator (Critic) ==========
            discriminator.zero_grad()
            
            # Real images score
            d_real = discriminator(real_images)
            
            # Generate fake images
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_images = generator(noise).detach()
            
            # Fake images score  
            d_fake = discriminator(fake_images)
            
            # Gradient penalty
            gp = compute_gradient_penalty(discriminator, real_images, fake_images, device)
            
            # Wasserstein distance (D(real) - D(fake))
            wasserstein_dist = d_real.mean() - d_fake.mean()
            
            # WGAN-GP loss: minimize -wasserstein_dist + lambda * gp
            d_loss = -wasserstein_dist + args.gp_weight * gp
            
            d_loss.backward()
            optimizer_d.step()
            
            running_d_loss += d_loss.item()
            running_wd += wasserstein_dist.item()
            running_gp += gp.item()
            d_losses.append(d_loss.item())
            wasserstein_distances.append(wasserstein_dist.item())
            
            # ========== Train Generator ==========
            # Generator is updated every n_critic iterations
            if (batch_idx + 1) % args.n_critic == 0:
                generator.zero_grad()
                
                noise = torch.randn(batch_size, args.latent_dim, device=device)
                fake_images = generator(noise)
                
                # Generator wants to maximize D(fake) = minimize -D(fake)
                g_loss = -discriminator(fake_images).mean()
                
                g_loss.backward()
                optimizer_g.step()
                
                running_g_loss += g_loss.item()
                n_g_updates += 1
                g_losses.append(g_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{running_g_loss / max(1, n_g_updates):.4f}',
                'D_loss': f'{d_loss.item():.4f}',
                'W_dist': f'{wasserstein_dist.item():.4f}',
                'GP': f'{gp.item():.4f}'
            })
        
        # Epoch stats
        avg_g_loss = running_g_loss / max(1, n_g_updates)
        avg_d_loss = running_d_loss / len(dataloader)
        avg_wd = running_wd / len(dataloader)
        avg_gp = running_gp / len(dataloader)
        print(f'Epoch {epoch+1} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | W_dist: {avg_wd:.4f} | GP: {avg_gp:.4f}')
        
        # Save checkpoint and samples
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            # Save checkpoint
            ckpt_path = os.path.join(args.save_dir, f'gan_epoch{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'wasserstein_distances': wasserstein_distances,
                'args': vars(args),
            }, ckpt_path)
            print(f'Checkpoint saved: {ckpt_path}')
            
            # Generate samples
            generator.eval()
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                fake_samples = (fake_samples + 1.0) / 2.0  # [-1,1] -> [0,1]
                fake_samples = fake_samples.clamp(0.0, 1.0)
            
            sample_path = os.path.join(samples_dir, f'samples_epoch{epoch+1:04d}.png')
            save_image(fake_samples, sample_path, nrow=8)
            print(f'Samples saved: {sample_path}')
            generator.train()
    
    # Save best model (latest)
    best_path = os.path.join(args.save_dir, 'best_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
        'wasserstein_distances': wasserstein_distances,
        'args': vars(args),
    }, best_path)
    print(f'Best model saved: {best_path}')
    
    # Plot losses
    print('\nGenerating loss plots...')
    plot_losses(g_losses, d_losses, args.save_dir)
    
    # Generate final samples
    print('\nGenerating final samples...')
    generator.eval()
    for i in range(3):
        noise = torch.randn(64, args.latent_dim, device=device)
        with torch.no_grad():
            fake_samples = generator(noise)
            fake_samples = (fake_samples + 1.0) / 2.0
            fake_samples = fake_samples.clamp(0.0, 1.0)
        sample_path = os.path.join(samples_dir, f'final_samples_{i+1}.png')
        save_image(fake_samples, sample_path, nrow=8)
        print(f'Final samples saved: {sample_path}')
    
    # Save training history
    history = {
        'g_losses': g_losses,
        'd_losses': d_losses,
        'wasserstein_distances': wasserstein_distances,
        'args': vars(args)
    }
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print('Training completed!')
    print(f'Final G Loss: {avg_g_loss:.4f}')
    print(f'Final D Loss: {avg_d_loss:.4f}')
    print(f'Final Wasserstein Distance: {avg_wd:.4f}')
    print(f'Models saved to: {args.save_dir}')
    print(f'Samples saved to: {samples_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    args = get_args()
    train(args)
