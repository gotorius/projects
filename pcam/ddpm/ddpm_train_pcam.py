"""
PCam (PatchCamelyon) データセット用 軽量DDPM 訓練コード
データセット: /mnt/data1/Public/MedImages/PCam_ImageFolder
メモリ効率重視版 (混合精度訓練対応)
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
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ========== 設定 ==========
def get_args():
    parser = argparse.ArgumentParser(description='Lightweight DDPM Training for PCam')
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder/train',
                        help='訓練データのパス')
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--image_size', type=int, default=224, help='画像サイズ')
    parser.add_argument('--batch_size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=100, help='エポック数')
    parser.add_argument('--lr', type=float, default=2e-4, help='学習率')
    parser.add_argument('--timesteps', type=int, default=1000, help='拡散ステップ数')
    parser.add_argument('--base_channels', type=int, default=64, help='UNetの基本チャンネル数')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--save_every', type=int, default=10, help='保存間隔(epochs)')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    return parser.parse_args()


# ========== 時刻埋め込み ==========
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ========== 軽量 ResBlock ==========
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.SiLU()
            )
        else:
            self.time_mlp = None
        
        self.act = nn.SiLU()

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
            h = h + time_emb
        h = self.act(h)
        h = self.norm2(self.conv2(h))
        h = self.act(h)
        return h + self.skip(x)


# ========== 軽量 U-Net ==========
class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        
        self.enc2 = ResidualBlock(base_ch * 2, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        
        self.enc3 = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1)
        
        self.enc4 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)

        # Bottleneck
        self.bot1 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch * 8, base_ch * 8, time_emb_dim)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch * 16, base_ch * 8, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch * 8, base_ch * 4, time_emb_dim)
        
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch * 4, base_ch * 2, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim)

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encode
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        
        e3 = self.enc3(d2, t_emb)
        d3 = self.down3(e3)
        
        e4 = self.enc4(d3, t_emb)
        d4 = self.down4(e4)

        # Bottleneck
        b = self.bot1(d4, t_emb)
        b = self.bot2(b, t_emb)

        # Decode
        u4 = self.up4(b)
        u4 = torch.cat([u4, e4], dim=1)
        u4 = self.dec4(u4, t_emb)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, e3], dim=1)
        u3 = self.dec3(u3, t_emb)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, t_emb)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, t_emb)

        return self.out_conv(u1)


# ========== EMA ==========
class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().cpu().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new = param.detach().cpu().clone()
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * new

    def apply(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().cpu().clone()
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}


# ========== DDPM ユーティリティ ==========
class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For sampling
        self.posterior_variance = torch.zeros_like(self.betas)
        self.posterior_variance[1:] = (
            self.betas[1:] * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        )
        self.posterior_variance[0] = 1e-8

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def p_sample(self, model, x_t, t):
        """Reverse diffusion step"""
        t_scalar = t[0].item()
        
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = (1.0 / torch.sqrt(self.alphas[t])).view(-1, 1, 1, 1)
        
        # Predict noise
        eps_pred = model(x_t, t)
        
        # Compute mean
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alpha_t * eps_pred)
        
        if t_scalar == 0:
            return model_mean
        else:
            noise = torch.randn_like(x_t)
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Full reverse process"""
        device = self.device
        batch_size = shape[0]
        
        x = torch.randn(shape, device=device)
        
        for t in tqdm(reversed(range(self.timesteps)), desc='Sampling', total=self.timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)
        
        return x

    def training_loss(self, model, x_0):
        """Compute training loss"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = model(x_t, t)
        return F.mse_loss(eps_pred, noise)


# ========== グラフ描画 ==========
def plot_losses(all_losses, epoch_losses, save_dir):
    """訓練ロスのグラフを描画"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Iteration loss
    axes[0].plot(all_losses, 'b-', alpha=0.5, linewidth=0.5)
    window = min(100, len(all_losses) // 10 + 1)
    if len(all_losses) > window:
        moving_avg = np.convolve(all_losses, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(all_losses)), moving_avg, 'r-', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss (per iteration)')
    axes[0].grid(True, alpha=0.3)
    
    # Epoch loss
    axes[1].plot(range(1, len(epoch_losses)+1), epoch_losses, 'bo-', linewidth=2, markersize=4)
    min_loss = min(epoch_losses)
    min_epoch = epoch_losses.index(min_loss) + 1
    axes[1].axhline(y=min_loss, color='r', linestyle='--', alpha=0.5)
    axes[1].scatter([min_epoch], [min_loss], color='red', s=100, zorder=5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Loss')
    axes[1].set_title(f'Epoch Loss (min: {min_loss:.6f} @ epoch {min_epoch})')
    axes[1].grid(True, alpha=0.3)
    
    # Log scale
    axes[2].semilogy(range(1, len(epoch_losses)+1), epoch_losses, 'go-', linewidth=2, markersize=4)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss (log scale)')
    axes[2].set_title('Epoch Loss (Log Scale)')
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
    
    # Model
    print(f'\nBuilding SimpleUNet (base_channels={args.base_channels})')
    model = SimpleUNet(in_ch=3, base_ch=args.base_channels, time_emb_dim=256).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Diffusion
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    
    # Optimizer & Scaler (AMP)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # EMA
    ema = EMA(model, decay=0.9995)
    
    # Resume
    start_epoch = 0
    all_losses = []
    epoch_losses = []
    best_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    
    if args.resume:
        print(f'\nResuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        all_losses = ckpt.get('all_losses', [])
        epoch_losses = ckpt.get('epoch_losses', [])
    
    # Training loop
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Image size: {args.image_size}, LR: {args.lr}')
    print(f'Early stopping patience: {args.patience} epochs')
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = diffusion.training_loss(model, images)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # EMA update
            ema.update(model)
            
            loss_val = loss.item()
            all_losses.append(loss_val)
            running_loss += loss_val
            
            pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}')
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            ema.apply(model)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'ema_state_dict': ema.shadow,
                'best_loss': best_loss,
                'all_losses': all_losses,
                'epoch_losses': epoch_losses,
                'args': vars(args),
            }, best_path)
            ema.restore(model)
            print(f'*** Best model saved! (Loss: {best_loss:.6f}) ***')
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter}/{args.patience} epochs')
            
            if patience_counter >= args.patience:
                print(f'\n*** Early stopping triggered after {epoch+1} epochs ***')
                early_stopped = True
                break
        
        # Save checkpoint and samples
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            # Save checkpoint
            ckpt_path = os.path.join(args.save_dir, f'ddpm_epoch{epoch+1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'all_losses': all_losses,
                'epoch_losses': epoch_losses,
                'args': vars(args),
            }, ckpt_path)
            print(f'Checkpoint saved: {ckpt_path}')
            
            # Generate samples with EMA
            ema.apply(model)
            model.eval()
            
            samples = diffusion.p_sample_loop(model, (16, 3, args.image_size, args.image_size))
            samples = (samples + 1.0) / 2.0  # [-1,1] -> [0,1]
            samples = samples.clamp(0.0, 1.0)
            
            sample_path = os.path.join(samples_dir, f'samples_epoch{epoch+1:04d}.png')
            save_image(samples, sample_path, nrow=4)
            print(f'Samples saved: {sample_path}')
            
            model.train()
            ema.restore(model)
    
    # Final: Plot losses
    print('\nGenerating loss plots...')
    plot_losses(all_losses, epoch_losses, args.save_dir)
    
    # Final: Load best model and generate samples
    print('\nGenerating final samples with best model...')
    best_ckpt = torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()
    
    for i in range(3):
        samples = diffusion.p_sample_loop(model, (16, 3, args.image_size, args.image_size))
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0)
        sample_path = os.path.join(samples_dir, f'final_samples_{i+1}.png')
        save_image(samples, sample_path, nrow=4)
        print(f'Final samples saved: {sample_path}')
    
    # Save training history
    history = {
        'all_losses': all_losses,
        'epoch_losses': epoch_losses,
        'args': vars(args)
    }
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print('Training completed!')
    if early_stopped:
        print(f'Early stopped at epoch {len(epoch_losses)}')
    print(f'Best epoch loss: {best_loss:.6f}')
    print(f'Models saved to: {args.save_dir}')
    print(f'Samples saved to: {samples_dir}')
    print(f'{"="*60}')


if __name__ == '__main__':
    args = get_args()
    train(args)
