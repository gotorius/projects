"""
VAE (MagNet-style) Training Script for ChestX-ray Dataset

Reference:
"MagNet: a Two-Pronged Defense against Adversarial Examples"
Meng & Chen, ACM CCS 2017
https://arxiv.org/abs/1705.09064

アーキテクチャ: Convolutional VAE (グレースケール)
入力: グレースケール画像 (1チャンネル)
出力サイズ: 224x224

MagNetの防御:
1. オートエンコーダ（VAE）で画像を再構成
2. 再構成により敵対的摂動を除去
3. 再構成画像を分類器に入力

実行例:
python train_vae.py --epochs 200 --batch_size 64 --gpu 0
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
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description='Train VAE (MagNet-style) for ChestX-ray')
    
    # モデル設定
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent space dimension')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Image size')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='KL divergence weight (beta-VAE)')
    parser.add_argument('--kl_anneal_epochs', type=int, default=10,
                        help='Epochs for KL annealing')
    
    # パス設定
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/vae/checkpoints',
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


# ========== Residual Block ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))


# ========== Encoder ==========
class Encoder(nn.Module):
    """
    Convolutional Encoder for VAE
    224x224 -> 112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> latent
    """
    def __init__(self, img_channels=1, base_ch=64, latent_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(img_channels, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.2),
            ResidualBlock(base_ch, base_ch),
            
            # 112 -> 56
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2),
            ResidualBlock(base_ch * 2, base_ch * 2),
            
            # 56 -> 28
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2),
            ResidualBlock(base_ch * 4, base_ch * 4),
            
            # 28 -> 14
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2),
            ResidualBlock(base_ch * 8, base_ch * 8),
            
            # 14 -> 7
            nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2),
        )
        
        # 7x7xbase_ch*8 -> latent
        self.fc_mu = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(base_ch * 8 * 7 * 7, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ========== Decoder ==========
class Decoder(nn.Module):
    """
    Convolutional Decoder for VAE
    latent -> 7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
    """
    def __init__(self, img_channels=1, base_ch=64, latent_dim=256):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 8 * 7 * 7),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # 7 -> 14
            nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(),
            ResidualBlock(base_ch * 8, base_ch * 8),
            
            # 14 -> 28
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(),
            ResidualBlock(base_ch * 4, base_ch * 4),
            
            # 28 -> 56
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(),
            ResidualBlock(base_ch * 2, base_ch * 2),
            
            # 56 -> 112
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(),
            ResidualBlock(base_ch, base_ch),
            
            # 112 -> 224
            nn.ConvTranspose2d(base_ch, img_channels, 4, 2, 1),
            nn.Sigmoid()  # [0, 1]
        )
        
        self.base_ch = base_ch
    
    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, self.base_ch * 8, 7, 7)
        return self.decoder(h)


# ========== VAE ==========
class VAE(nn.Module):
    """Convolutional VAE"""
    def __init__(self, img_channels=1, base_ch=64, latent_dim=256):
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
        """再構成のみ（MagNet防御用）"""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z)


# ========== 損失関数 ==========
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss = Reconstruction loss + beta * KL divergence"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


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
    model = VAE(img_channels=1, base_ch=args.base_ch, latent_dim=args.latent_dim).to(device)
    
    # パラメータ数
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {n_params:,}")
    
    # オプティマイザ
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 固定サンプル（可視化用）
    fixed_x = next(iter(dataloader))[:16].to(device)
    
    # 訓練ループ
    print(f"\nStarting training for {args.epochs} epochs...")
    
    best_loss = float('inf')
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        n_batches = 0
        
        # KL annealing
        if args.kl_anneal_epochs > 0:
            kl_weight = min(1.0, epoch / args.kl_anneal_epochs) * args.beta
        else:
            kl_weight = args.beta
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta=kl_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{kl_weight:.3f}'
            })
        
        # エポック終了時の平均損失
        avg_total_loss = epoch_total_loss / n_batches
        avg_recon_loss = epoch_recon_loss / n_batches
        avg_kl_loss = epoch_kl_loss / n_batches
        
        history['total_loss'].append(avg_total_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        
        print(f"Epoch {epoch}: total={avg_total_loss:.4f}, recon={avg_recon_loss:.4f}, kl={avg_kl_loss:.4f}")
        
        # サンプル生成
        if epoch % args.sample_interval == 0:
            model.eval()
            with torch.no_grad():
                # 再構成
                recon_fixed, _, _ = model(fixed_x)
                comparison = torch.cat([fixed_x[:8], recon_fixed[:8]], dim=0)
                grid = make_grid(comparison, nrow=8, padding=2)
                save_image(grid, samples_dir / f'recon_epoch_{epoch:04d}.png')
                
                # ランダムサンプル
                z_random = torch.randn(16, args.latent_dim, device=device)
                samples = model.decode(z_random)
                grid = make_grid(samples, nrow=4, padding=2)
                save_image(grid, samples_dir / f'random_epoch_{epoch:04d}.png')
            model.train()
        
        # チェックポイント保存
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
                'history': history
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"Saved checkpoint at epoch {epoch}")
        
        # 最良モデル保存
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'args': vars(args)
            }, output_dir / 'best_model.pth')
    
    # 最終モデル保存
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'history': history
    }, output_dir / 'final_model.pth')
    
    # 履歴保存
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == '__main__':
    args = parse_args()
    train(args)
