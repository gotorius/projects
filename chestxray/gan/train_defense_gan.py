"""
Defense-GAN Training Script for ChestX-ray Dataset

Reference: 
"Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
Samangouei et al., ICLR 2018
https://arxiv.org/abs/1805.06605

アーキテクチャ: WGAN-GP (Wasserstein GAN with Gradient Penalty)
入力: グレースケール画像 (1チャンネル)
出力サイズ: 224x224

実行例:
python train_defense_gan.py --epochs 200 --batch_size 64 --gpu 0
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='Train Defense-GAN (WGAN-GP) for ChestX-ray')
    
    # モデル設定
    parser.add_argument('--z_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Number of generator features')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Number of discriminator features')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='Adam beta2')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of critic iterations per generator iteration')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Gradient penalty coefficient')
    
    # パス設定
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/gan/checkpoints',
                        help='Output directory')
    
    # その他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save interval (epochs)')
    parser.add_argument('--sample_interval', type=int, default=5,
                        help='Sample generation interval (epochs)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


# ========== データセット ==========
class ChestXrayDataset(Dataset):
    """ChestX-ray グレースケールデータセット"""
    def __init__(self, root_dir, split='train', img_size=224):
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
        
        self.image_paths = []
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                for img_path in class_dir.glob('*.jpeg'):
                    self.image_paths.append(img_path)
                for img_path in class_dir.glob('*.jpg'):
                    self.image_paths.append(img_path)
                for img_path in class_dir.glob('*.png'):
                    self.image_paths.append(img_path)
        
        print(f"Loaded {len(self.image_paths)} images from {split} set")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# ========== Generator (DCGAN-style) ==========
class Generator(nn.Module):
    """
    DCGAN-style Generator for 224x224 grayscale images
    z_dim -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """
    def __init__(self, z_dim=128, ngf=64, img_channels=1):
        super().__init__()
        self.z_dim = z_dim
        
        # 計算: 224 = 7 * 2^5 = 224
        self.init_size = 7
        
        self.fc = nn.Sequential(
            nn.Linear(z_dim, ngf * 16 * self.init_size * self.init_size),
            nn.BatchNorm1d(ngf * 16 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.main = nn.Sequential(
            # 7x7 -> 14x14
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
            nn.ConvTranspose2d(ngf, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, self.init_size, self.init_size)
        return self.main(x)


# ========== Discriminator (Critic) ==========
class Discriminator(nn.Module):
    """
    DCGAN-style Discriminator (Critic for WGAN-GP)
    224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1
    """
    def __init__(self, ndf=64, img_channels=1):
        super().__init__()
        
        self.main = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(img_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 112x112 -> 56x56
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 2, 56, 56]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 56x56 -> 28x28
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 4, 28, 28]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 28x28 -> 14x14
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 8, 14, 14]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14x14 -> 7x7
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.LayerNorm([ndf * 16, 7, 7]),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 7x7 -> 1x1
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=False)
        )
    
    def forward(self, x):
        return self.main(x).view(-1)


# ========== Gradient Penalty ==========
def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """WGAN-GP gradient penalty"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


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
    G = Generator(z_dim=args.z_dim, ngf=args.ngf, img_channels=1).to(device)
    D = Discriminator(ndf=args.ndf, img_channels=1).to(device)
    
    # パラメータ数
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    
    # オプティマイザ
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    
    # 固定ノイズ（サンプル生成用）
    fixed_z = torch.randn(64, args.z_dim, device=device)
    
    # 訓練ループ
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, n_critic: {args.n_critic}")
    
    best_g_loss = float('inf')
    history = {'d_loss': [], 'g_loss': [], 'gp': []}
    
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_gp = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for i, real_imgs in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # ==================== Train Discriminator ====================
            optimizer_D.zero_grad()
            
            # 本物のスコア
            d_real = D(real_imgs)
            
            # 偽物を生成
            z = torch.randn(batch_size, args.z_dim, device=device)
            fake_imgs = G(z).detach()
            d_fake = D(fake_imgs)
            
            # Gradient Penalty
            gp = compute_gradient_penalty(D, real_imgs, fake_imgs, device)
            
            # Wasserstein loss + GP
            d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp
            d_loss.backward()
            optimizer_D.step()
            
            epoch_d_loss += d_loss.item()
            epoch_gp += gp.item()
            
            # ==================== Train Generator ====================
            if (i + 1) % args.n_critic == 0:
                optimizer_G.zero_grad()
                
                z = torch.randn(batch_size, args.z_dim, device=device)
                fake_imgs = G(z)
                g_loss = -D(fake_imgs).mean()
                
                g_loss.backward()
                optimizer_G.step()
                
                epoch_g_loss += g_loss.item()
            
            n_batches += 1
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{epoch_g_loss / max(1, n_batches // args.n_critic):.4f}',
                'GP': f'{gp.item():.4f}'
            })
        
        # エポック終了時の平均損失
        avg_d_loss = epoch_d_loss / n_batches
        avg_g_loss = epoch_g_loss / max(1, n_batches // args.n_critic)
        avg_gp = epoch_gp / n_batches
        
        history['d_loss'].append(avg_d_loss)
        history['g_loss'].append(avg_g_loss)
        history['gp'].append(avg_gp)
        
        print(f"Epoch {epoch}: D_loss={avg_d_loss:.4f}, G_loss={avg_g_loss:.4f}, GP={avg_gp:.4f}")
        
        # サンプル生成
        if epoch % args.sample_interval == 0:
            G.eval()
            with torch.no_grad():
                fake_samples = G(fixed_z)
                grid = make_grid(fake_samples, nrow=8, normalize=True, value_range=(-1, 1))
                save_image(grid, samples_dir / f'epoch_{epoch:04d}.png')
            G.train()
        
        # チェックポイント保存
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'args': vars(args),
                'history': history
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # 最良モデル保存
        if avg_g_loss < best_g_loss:
            best_g_loss = avg_g_loss
            torch.save({
                'epoch': epoch,
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'args': vars(args)
            }, output_dir / 'best_model.pth')
    
    # 最終モデル保存
    torch.save({
        'epoch': args.epochs,
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
        'args': vars(args),
        'history': history
    }, output_dir / 'final_model.pth')
    
    # 履歴保存
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best G_loss: {best_g_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
