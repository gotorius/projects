#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from torchvision import models

DATA_DIR = '/mnt/data1/gotou/projects/Medical/kaggledata'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')

# データ拡張（学習時のみ）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PCamDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels = labels_df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_id = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

labels_df = pd.read_csv(LABELS_CSV)
train_df, val_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df['label'])

train_dataset = PCamDataset(TRAIN_IMG_DIR, train_df, train_transform)
val_dataset = PCamDataset(TRAIN_IMG_DIR, val_df, val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

resnet50.fc = nn.Linear(resnet50.fc.in_features, 1)
model = resnet50.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル定義（学習時と同じ）
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # 1ユニット出力
model = model.to(device)

# 重みロード
ckpt_path = "/mnt/data1/gotou/projects/Medical/kaggledata/best_model_weights.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model from {ckpt_path}")



# In[ ]:


import math

# 拡散モデル用のノイズスケジューラ
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_schedule='cosine'):
        self.timesteps = timesteps
        
        if beta_schedule == 'linear':
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            self.betas = cosine_beta_schedule(timesteps)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 後方過程のパラメータ
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_0, t, noise=None):
        """前方拡散過程: x_0からx_tを生成"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

# U-Netベースのノイズ予測ネットワーク
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=32):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # 224x224用の調整されたアーキテクチャ
        # Encoder (Downsampling): 224 -> 112 -> 56 -> 28 -> 14
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)  # 224x224
        self.down1 = Block(64, 128, time_emb_dim)   # 224 -> 112
        self.down2 = Block(128, 256, time_emb_dim)  # 112 -> 56
        self.down3 = Block(256, 512, time_emb_dim)  # 56 -> 28
        
        # Bottleneck: 28 -> 14
        self.bot1 = nn.Conv2d(512, 512, 3, padding=1)
        self.bot2 = nn.Conv2d(512, 512, 4, 2, 1)  # 28 -> 14
        self.bot_time = nn.Linear(time_emb_dim, 512)
        
        # Decoder (Upsampling): 14 -> 28 -> 56 -> 112 -> 224
        self.up0 = nn.ConvTranspose2d(512, 512, 4, 2, 1)  # 14 -> 28
        self.up1 = Block(512, 256, time_emb_dim, up=True)  # 28 -> 56
        self.up2 = Block(256, 128, time_emb_dim, up=True)  # 56 -> 112
        self.up3 = Block(128, 64, time_emb_dim, up=True)   # 112 -> 224
        
        self.out = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        # Encoder
        x = self.conv0(x)           # 64 x 224 x 224
        down1 = self.down1(x, t)    # 128 x 112 x 112
        down2 = self.down2(down1, t) # 256 x 56 x 56
        down3 = self.down3(down2, t) # 512 x 28 x 28
        
        # Bottleneck
        bot = self.bot1(down3)       # 512 x 28 x 28
        time_emb = self.bot_time(t)[(..., ) + (None, ) * 2]
        bot = bot + time_emb
        bot = self.bot2(bot)         # 512 x 14 x 14
        
        # Decoder
        up0 = self.up0(bot)          # 512 x 28 x 28
        up1 = self.up1(torch.cat([up0, down3], dim=1), t)  # 256 x 56 x 56
        up2 = self.up2(torch.cat([up1, down2], dim=1), t)  # 128 x 112 x 112
        up3 = self.up3(torch.cat([up2, down1], dim=1), t)  # 64 x 224 x 224
        
        return self.out(up3)

print("拡散モデルの定義が完了しました（224x224対応）")


# In[3]:


# 拡散モデルの訓練用データローダー（画像のみ使用、ラベルは不要）
# 画像サイズは224x224のまま使用
diffusion_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]に正規化
])

class DiffusionDataset(Dataset):
    """拡散モデル訓練用データセット（ラベル不要）"""
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.img_ids = labels_df.iloc[:, 0].values  # 画像IDのみ取得
        self.transform = transform
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# 拡散モデル用データローダー作成
diffusion_train_dataset = DiffusionDataset(TRAIN_IMG_DIR, train_df, diffusion_train_transform)
diffusion_train_loader = DataLoader(diffusion_train_dataset, batch_size=16, shuffle=True, num_workers=4)

print(f"拡散モデル訓練用データセット: {len(diffusion_train_dataset)} 枚")
print(f"画像サイズ: 224x224")
print(f"バッチサイズ: 16")


# In[4]:


# 拡散モデルの訓練
def train_diffusion_model(model, diffusion, dataloader, epochs=10, device='cuda', save_dir='./ddpm_checkpoints_pcam'):
    """
    拡散モデルの訓練関数
    
    Args:
        model: ノイズ予測モデル (U-Net)
        diffusion: 拡散プロセス
        dataloader: 訓練データローダー
        epochs: エポック数
        device: デバイス
        save_dir: チェックポイント保存ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    mse_loss = nn.MSELoss()
    
    # 拡散プロセスのパラメータをデバイスに移動
    diffusion.betas = diffusion.betas.to(device)
    diffusion.alphas = diffusion.alphas.to(device)
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    diffusion.sqrt_alphas_cumprod = diffusion.sqrt_alphas_cumprod.to(device)
    diffusion.sqrt_one_minus_alphas_cumprod = diffusion.sqrt_one_minus_alphas_cumprod.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # ランダムなタイムステップを選択
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # ノイズを追加
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise)
            
            # ノイズを予測
            predicted_noise = model(x_t, t)
            
            # 損失計算
            loss = mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}')
        
        # エポックごとにチェックポイントを保存
        checkpoint_path = os.path.join(save_dir, f'ddpm_pcam_epoch{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f'チェックポイント保存: {checkpoint_path}')
    
    print('訓練完了！')
    return model

# モデルと拡散プロセスの初期化
timesteps = 1000
diffusion = DiffusionProcess(timesteps=timesteps, beta_schedule='cosine')
unet = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=32)

print(f"モデルパラメータ数: {sum(p.numel() for p in unet.parameters()):,}")
print(f"拡散ステップ数: {timesteps}")
print(f"画像サイズ: 224x224")


# In[5]:


# 訓練開始（10エポック）
trained_model = train_diffusion_model(
    model=unet,
    diffusion=diffusion,
    dataloader=diffusion_train_loader,
    epochs=10,
    device=device,
    save_dir='./ddpm_checkpoints_pcam'
)


# In[ ]:

"""
# サンプル生成関数（逆拡散過程）
@torch.no_grad()
def sample_images(model, diffusion, n_samples=8, device='cuda', image_size=224):
    
    #訓練済み拡散モデルから画像を生成
    
    Args:
        model: 訓練済みノイズ予測モデル
        diffusion: 拡散プロセス
        n_samples: 生成する画像数
        device: デバイス
        image_size: 画像サイズ（デフォルト224）
    
    Returns:
        generated_images: 生成された画像 (n_samples, 3, image_size, image_size)
    
    model.eval()
    
    # 純粋なノイズから開始
    x = torch.randn(n_samples, 3, image_size, image_size).to(device)
    
    # 逆拡散過程
    for t in tqdm(reversed(range(diffusion.timesteps)), desc='サンプリング', total=diffusion.timesteps):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        
        # ノイズ予測
        predicted_noise = model(x, t_batch)
        
        # パラメータ取得
        alpha = diffusion.alphas[t]
        alpha_cumprod = diffusion.alphas_cumprod[t]
        beta = diffusion.betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        
        # 逆拡散ステップ
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise)
        if t > 0:
            x = x + torch.sqrt(beta) * noise
    
    return x

# 生成した画像を表示する関数
def show_generated_images(images, n_rows=2):
    
    import matplotlib.pyplot as plt
    
    n_samples = images.shape[0]
    n_cols = (n_samples + n_rows - 1) // n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i, ax in enumerate(axes):
        if i < n_samples:
            img = images[i].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # [-1, 1] -> [0, 1]
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

print("サンプリング関数の準備完了")


# In[ ]:


# サンプル生成の例（訓練後に実行）
# 特定のエポックのチェックポイントから読み込んで生成も可能

# 例: エポック10のモデルから画像生成
checkpoint_path = './ddpm_checkpoints_pcam/ddpm_pcam_epoch10.pth'

# モデルの読み込み
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet_loaded = SimpleUNet(in_channels=3, out_channels=3, time_emb_dim=32)
    unet_loaded.load_state_dict(checkpoint['model_state_dict'])
    unet_loaded = unet_loaded.to(device)
    
    print(f"チェックポイント読み込み: {checkpoint_path}")
    print(f"エポック: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.6f}")
    
    # 画像生成（224x224）
    generated_images = sample_images(
        model=unet_loaded,
        diffusion=diffusion,
        n_samples=8,
        device=device,
        image_size=224
    )
    
    # 生成画像の表示
    show_generated_images(generated_images, n_rows=2)
    
    # 生成画像の保存
    save_dir = './ddpm_samples_pcam'
    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(generated_images):
        img_np = img.cpu().permute(1, 2, 0).numpy()
        img_np = (img_np + 1) / 2  # [-1, 1] -> [0, 1]
        img_np = np.clip(img_np, 0, 1)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil.save(os.path.join(save_dir, f'sample_{i}.png'))
    
    print(f"生成画像を保存: {save_dir}")
else:
    print(f"チェックポイントが見つかりません: {checkpoint_path}")
    print("先に訓練を実行してください")"""