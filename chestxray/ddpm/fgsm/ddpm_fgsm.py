"""
ChestXray (肺炎分類) - FGSM攻撃 + DDPM防御検証スクリプト
拡散モデルによる敵対的画像の浄化と防御性能評価
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        ChestXrayデータセット（ImageFolder形式）
        root_dir: NORMAL/ と PNEUMONIA/ を含むディレクトリ
        """
        from pathlib import Path
        self.transform = transform
        self.samples = []
        
        # クラスフォルダを探索
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        # クラス名とインデックスのマッピング
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 画像ファイルを収集
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} images in {root_dir}")
        print(f"Classes: {self.classes}")
        print(f"Class to index: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# データ変換（テストデータ用）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# テストデータセット
test_dataset = ChestXrayDataset(TEST_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

print(f"Test samples: {len(test_dataset)}")

# ========== 分類器の読み込み ==========
print("\n" + "="*70)
print("Loading ResNet50 classifier...")
print("="*70)

clf_ckpt = "/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth"

model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2クラス分類
model = model.to(device)

# 重みロード
checkpoint = torch.load(clf_ckpt, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded classifier from {clf_ckpt}")
print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
print(f"Classes: {checkpoint.get('class_names', ['NORMAL', 'PNEUMONIA'])}")

# ========== 正規化パラメータ ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def denormalize(x_norm, mean, std):
    """正規化された画像をピクセル空間[0,1]に戻す"""
    return x_norm * std + mean

def renormalize(x_pixel, mean, std):
    """ピクセル空間[0,1]の画像を正規化"""
    return (x_pixel - mean) / std

def rgb_to_grayscale_pixel(x_rgb_pixel):
    """
    RGB画像(3ch, ピクセル空間)をグレースケール(1ch)に変換
    Args:
        x_rgb_pixel: [B, 3, H, W] in [0, 1]
    Returns:
        x_gray_pixel: [B, 1, H, W] in [0, 1]
    """
    return x_rgb_pixel.mean(dim=1, keepdim=True)

def grayscale_to_rgb_pixel(x_gray_pixel):
    """
    グレースケール画像(1ch, ピクセル空間)をRGB(3ch)に変換
    Args:
        x_gray_pixel: [B, 1, H, W] in [0, 1]
    Returns:
        x_rgb_pixel: [B, 3, H, W] in [0, 1]
    """
    return x_gray_pixel.repeat(1, 3, 1, 1)

# ========== FGSM攻撃関数 ==========
def fgsm_attack_improved(model, images, labels, epsilon_pixel, device,
                         mean_tensor, std_tensor, return_preds=True):
    """
    正規化を考慮したFGSM攻撃
    
    Args:
        model: 分類モデル
        images: 正規化済み画像 [B, C, H, W]
        labels: ラベル [B]
        epsilon_pixel: 摂動の大きさ（ピクセルスケール 0-1）
        device: デバイス
        mean_tensor: 正規化平均 [1, C, 1, 1]
        std_tensor: 正規化標準偏差 [1, C, 1, 1]
        return_preds: 攻撃後の予測を返すか
    
    Returns:
        adv_images: 敵対的画像（正規化済み）
        adv_preds: 攻撃後の予測（オプション）
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    
    # 勾配の符号
    grad_sign = grad.sign()
    
    # εをチャネルごとに正規化空間に変換
    if not torch.is_tensor(epsilon_pixel):
        eps_pixel_tensor = torch.tensor(epsilon_pixel, dtype=images.dtype, device=device)
    else:
        eps_pixel_tensor = epsilon_pixel.to(device).to(images.dtype)
    
    eps_norm = (eps_pixel_tensor / std_tensor).view(1, -1, 1, 1)
    
    # 敵対的画像を生成（正規化空間）
    adv_images = images + eps_norm * grad_sign
    
    # ピクセル空間でクリッピング
    adv_pixel = denormalize(adv_images, mean_tensor, std_tensor)
    adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)
    
    # 正規化空間に戻す
    adv_images = renormalize(adv_pixel, mean_tensor, std_tensor).detach()
    adv_images.requires_grad = False
    
    if return_preds:
        with torch.no_grad():
            adv_outputs = model(adv_images)
            adv_preds = torch.argmax(adv_outputs, dim=1)
        
        # メモリ解放
        del grad, grad_sign, adv_pixel, eps_norm, loss
        torch.cuda.empty_cache()
        return adv_images, adv_preds
    
    del grad, grad_sign, adv_pixel, eps_norm, loss
    torch.cuda.empty_cache()
    return adv_images

# ========== DDPMモデル定義 ==========
print("\n" + "="*70)
print("Loading DDPM model...")
print("="*70)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim
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

class AttentionBlock(nn.Module):
    """Self-attention block for UNet"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.ln = nn.LayerNorm(channels)
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape for attention
        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Self-attention with residual
        x_norm = self.ln(x_flat)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # Feedforward with residual
        x_flat = x_flat + self.ff(self.ln(x_flat))
        
        # Reshape back
        x_out = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return x_out

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=256):  # 1チャンネル用
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.attn3 = AttentionBlock(base_ch*4, num_heads=4)  # Attention追加
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn4 = AttentionBlock(base_ch*8, num_heads=4)  # Attention追加
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.attn_bot = AttentionBlock(base_ch*8, num_heads=4)  # Attention追加
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        self.dec4 = ResidualBlock(base_ch*16, base_ch*8, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, stride=2, padding=1)
        self.dec3 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.dec2 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb)
        d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb)
        e3 = self.attn3(e3)  # Attention適用
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        e4 = self.attn4(e4)  # Attention適用
        d4 = self.down4(e4)
        b = self.bot1(d4, t_emb)
        b = self.attn_bot(b)  # Attention適用
        b = self.bot2(b, t_emb)
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
        out = self.out_conv(u1)
        return out

# DDPMパラメータ
T_steps = 1000
betas = torch.linspace(1e-4, 0.02, T_steps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# DDPMモデルのロード
ddpm_ckpt = "/mnt/data1/gotou/projects/chestxray/ddpm/ddpm_out3/best_model.pth"
ddpm = SimpleUNet(in_ch=1, base_ch=64, time_emb_dim=256).to(device)  # 1チャンネル用

ckpt = torch.load(ddpm_ckpt, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    ddpm.load_state_dict(ckpt['model_state_dict'])
else:
    ddpm.load_state_dict(ckpt)
ddpm.eval()

print(f"Loaded DDPM from {ddpm_ckpt}")

# ========== DDPM浄化関数 ==========
# DDPM学習時の正規化（[-1,1]スケール、1チャンネル用）
ddpm_mean = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)
ddpm_std = torch.tensor([0.5]).view(1, 1, 1, 1).to(device)

def prepare_for_diffusion(x_rgb_norm, target_size=224):
    """
    RGB正規化画像からDDPM用の1ch [-1,1]正規化画像に変換
    Args:
        x_rgb_norm: RGB ImageNet正規化画像 [B, 3, H, W]
    Returns:
        x_gray_minus1to1: グレースケール [-1,1] 正規化画像 [B, 1, H, W]
    """
    # RGB正規化を解除してピクセル空間に
    x_rgb_pixel = denormalize(x_rgb_norm, imagenet_mean, imagenet_std)
    # グレースケールに変換
    x_gray_pixel = rgb_to_grayscale_pixel(x_rgb_pixel)
    # リサイズ
    x_gray_pixel = F.interpolate(x_gray_pixel, size=(target_size, target_size), 
                                  mode="bilinear", align_corners=False)
    # [-1, 1] に正規化
    x_gray_minus1to1 = (x_gray_pixel - ddpm_mean) / ddpm_std
    return x_gray_minus1to1

def recover_from_diffusion(x_gray_minus1to1, out_size=224):
    """
    DDPM用の1ch [-1,1]正規化画像からRGB ImageNet正規化画像に変換
    Args:
        x_gray_minus1to1: グレースケール [-1,1] 正規化画像 [B, 1, H, W]
    Returns:
        x_rgb_norm: RGB ImageNet正規化画像 [B, 3, H, W]
    """
    # [-1, 1] 正規化を解除してピクセル空間に
    x_gray_pixel = x_gray_minus1to1 * ddpm_std + ddpm_mean
    x_gray_pixel = torch.clamp(x_gray_pixel, 0.0, 1.0)
    # リサイズ
    x_gray_pixel = F.interpolate(x_gray_pixel, size=(out_size, out_size), 
                                  mode="bilinear", align_corners=False)
    # RGBに変換
    x_rgb_pixel = grayscale_to_rgb_pixel(x_gray_pixel)
    # ImageNet正規化
    x_rgb_norm = renormalize(x_rgb_pixel, imagenet_mean, imagenet_std)
    return x_rgb_norm

@torch.no_grad()
def diffusion_purify(x_adv_minus1to1, model, start_t=100, T_purify=50, eta=0.0, out_mode='x0'):
    """
    拡散モデルによる画像浄化
    
    Args:
        x_adv_minus1to1: 敵対的画像（[-1,1]正規化）
        model: DDPMモデル
        start_t: 拡散開始時刻
        T_purify: 逆拡散ステップ数
        eta: DDIMパラメータ（0=deterministic）
        out_mode: 'x0'でx0推定を返す、'x_t'でx_tを返す
    
    Returns:
        浄化された画像（[-1,1]正規化）
    """
    b = x_adv_minus1to1.size(0)
    t0 = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1to1)
    
    # Forward diffusion to start_t
    sqrt_a_bar_t0 = sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
    sqrt_1m_a_bar_t0 = sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
    x_t = sqrt_a_bar_t0 * x_adv_minus1to1 + sqrt_1m_a_bar_t0 * noise
    
    # Reverse diffusion
    eps_pred_final = None
    t_final = start_t
    for t_ in range(start_t, max(start_t - T_purify, 0), -1):
        t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)
        eps_pred_final = eps_pred
        t_final = t_
        
        alpha_t = alphas[t_]
        alpha_bar_t = alphas_cumprod[t_]
        
        if t_ > 0:
            alpha_bar_prev = alphas_cumprod[t_ - 1]
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_t)
            c1 = torch.sqrt(alpha_bar_prev)
            c2 = torch.sqrt(1 - alpha_bar_prev - sigma_t**2)
            x_t = c1 * (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t) + c2 * eps_pred
            if sigma_t > 0:
                x_t = x_t + sigma_t * torch.randn_like(x_t)
        else:
            x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
    
    if out_mode == 'x0':
        # x0推定を返す
        alpha_bar_final = alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_final) * eps_pred_final) / torch.sqrt(alpha_bar_final)
        return torch.clamp(x0_hat, -1.0, 1.0)
    else:
        return x_t

# ========== 評価ループ ==========
print("\n" + "="*70)
print("Starting evaluation...")
print("="*70)

# 実験パラメータ
epsilon_pixel = 8/255.0
start_t = 80
T_purify = 50
save_examples_dir = "/mnt/data1/gotou/projects/chestxray/ddpm/fgsm/purify_examples2"
os.makedirs(save_examples_dir, exist_ok=True)

# triplets のみ出力
save_triplets_dir = os.path.join(save_examples_dir, "triplets")
os.makedirs(save_triplets_dir, exist_ok=True)

# 画像保存の設定
MAX_IMAGES_TO_SAVE = 3  # 3枚のみ保存
saved_image_count = 0

# 統計変数
total = 0
correct_clean = 0
correct_adv = 0
correct_purified = 0

all_labels = []
all_preds_clean = []
all_preds_adv = []
all_preds_purified = []

l2_norms_adv = []
linf_norms_adv = []
l2_norms_purified = []
linf_norms_purified = []

# 評価ループ
for batch_idx, (images_norm, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
    images_norm = images_norm.to(device)
    labels = labels.to(device).long()
    b = images_norm.size(0)
    
    # 1) Clean prediction
    with torch.no_grad():
        logits_clean = model(images_norm)
        preds_clean = torch.argmax(logits_clean, dim=1)
    
    # 元画像で正解した画像のみフィルタリング
    correct_mask = (preds_clean == labels)
    correct_indices = torch.where(correct_mask)[0]
    
    if len(correct_indices) == 0:
        continue
    
    images_norm_correct = images_norm[correct_indices]
    labels_correct = labels[correct_indices]
    preds_clean_correct = preds_clean[correct_indices]
    
    total += len(correct_indices)
    correct_clean += len(correct_indices)
    all_labels.extend(labels_correct.cpu().numpy())
    all_preds_clean.extend(preds_clean_correct.cpu().numpy())
    
    # 2) FGSM攻撃
    adv_images_norm, adv_preds = fgsm_attack_improved(
        model=model,
        images=images_norm_correct,
        labels=labels_correct,
        epsilon_pixel=epsilon_pixel,
        device=device,
        mean_tensor=imagenet_mean,
        std_tensor=imagenet_std,
        return_preds=True
    )
    
    correct_adv += (adv_preds == labels_correct).sum().item()
    all_preds_adv.extend(adv_preds.cpu().numpy())
    
    # L2/L∞ノルム計算
    clean_pixel = denormalize(images_norm_correct, imagenet_mean, imagenet_std)
    adv_pixel = denormalize(adv_images_norm, imagenet_mean, imagenet_std)
    diff_adv = (adv_pixel - clean_pixel).view(len(correct_indices), -1)
    l2_adv = torch.norm(diff_adv, p=2, dim=1).cpu().numpy()
    linf_adv = torch.norm(diff_adv, p=float('inf'), dim=1).cpu().numpy()
    l2_norms_adv.extend(l2_adv)
    linf_norms_adv.extend(linf_adv)
    
    # 3) DDPM浄化（RGB → 1ch グレースケールに変換してから浄化）
    x_adv_for_diff = prepare_for_diffusion(adv_images_norm, target_size=224)
    purified_minus1to1 = diffusion_purify(
        x_adv_for_diff, ddpm, 
        start_t=start_t, 
        T_purify=T_purify, 
        eta=0.0, 
        out_mode='x0'
    )
    
    # 4) 浄化画像の分類（1ch → RGB に変換してから分類）
    purified_norm = recover_from_diffusion(purified_minus1to1, out_size=224)
    with torch.no_grad():
        logits_pur = model(purified_norm)
        preds_pur = torch.argmax(logits_pur, dim=1)
        correct_purified += (preds_pur == labels_correct).sum().item()
        all_preds_purified.extend(preds_pur.cpu().numpy())
        
        # 浄化後のノルム計算
        pur_pixel = denormalize(purified_norm, imagenet_mean, imagenet_std)
        diff_pur = (pur_pixel - clean_pixel).view(len(correct_indices), -1)
        l2_pur = torch.norm(diff_pur, p=2, dim=1).cpu().numpy()
        linf_pur = torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy()
        l2_norms_purified.extend(l2_pur)
        linf_norms_purified.extend(linf_pur)
    
    # 5) Triplet画像保存（最初の3枚のみ）
    if saved_image_count < MAX_IMAGES_TO_SAVE:
        clean_pixel_save = denormalize(images_norm_correct.detach(), imagenet_mean, imagenet_std).clamp(0,1)
        adv_pixel_save = denormalize(adv_images_norm.detach(), imagenet_mean, imagenet_std).clamp(0,1)
        pur_pixel_save = denormalize(purified_norm.detach(), imagenet_mean, imagenet_std).clamp(0,1)
        
        # 224x224に統一
        clean_resized = F.interpolate(clean_pixel_save, size=(224,224), mode="bilinear", align_corners=False)
        adv_resized = F.interpolate(adv_pixel_save, size=(224,224), mode="bilinear", align_corners=False)
        pur_resized = F.interpolate(pur_pixel_save, size=(224,224), mode="bilinear", align_corners=False)
        
        for i in range(len(correct_indices)):
            if saved_image_count >= MAX_IMAGES_TO_SAVE:
                break
            
            # triplet tile のみ保存
            row = torch.cat([clean_resized[i], adv_resized[i], pur_resized[i]], dim=2)
            save_image(row, os.path.join(save_triplets_dir, f"{saved_image_count:05d}_triplet.png"))
            saved_image_count += 1

# ========== 結果表示 ==========
clean_acc = correct_clean / total if total > 0 else 0.0
adv_acc = correct_adv / total if total > 0 else 0.0
pur_acc = correct_purified / total if total > 0 else 0.0

l2_norms_adv = np.array(l2_norms_adv)
linf_norms_adv = np.array(linf_norms_adv)
l2_norms_purified = np.array(l2_norms_purified)
linf_norms_purified = np.array(linf_norms_purified)

print("\n" + "="*70)
print("==== Results (ChestXray - FGSM Attack + DDPM Defense) ====")
print("="*70)
print(f"Total samples evaluated: {total} (元画像で正解したもののみ)")
print(f"Attack: FGSM with epsilon={epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)")
print(f"Purification: DDPM start_t={start_t}, T_purify={T_purify}")
print("-"*70)
print(f"Clean accuracy:     {clean_acc:.4f} (常に1.0)")
print(f"Adv (FGSM) accuracy:{adv_acc:.4f}")
print(f"Purified accuracy:  {pur_acc:.4f}")
print(f"Defense improvement: {pur_acc - adv_acc:+.4f}")
print("-"*70)

print("\n" + "="*70)
print("==== Perturbation Norms ====")
print("="*70)
print("Adversarial Perturbations (vs Clean):")
print(f"  L2 norm:   mean={l2_norms_adv.mean():.4f}, std={l2_norms_adv.std():.4f}, "
      f"min={l2_norms_adv.min():.4f}, max={l2_norms_adv.max():.4f}")
print(f"  L∞ norm:   mean={linf_norms_adv.mean():.4f}, std={linf_norms_adv.std():.4f}, "
      f"min={linf_norms_adv.min():.4f}, max={linf_norms_adv.max():.4f}")
print("\nPurified Images (vs Clean):")
print(f"  L2 norm:   mean={l2_norms_purified.mean():.4f}, std={l2_norms_purified.std():.4f}, "
      f"min={l2_norms_purified.min():.4f}, max={l2_norms_purified.max():.4f}")
print(f"  L∞ norm:   mean={linf_norms_purified.mean():.4f}, std={linf_norms_purified.std():.4f}, "
      f"min={linf_norms_purified.min():.4f}, max={linf_norms_purified.max():.4f}")
print("="*70)

print(f"\nSaved {saved_image_count} triplet images to: {save_triplets_dir}")

# ========== 混同行列の計算(テキスト形式のみ) ==========
def print_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title}:")
    print(f"  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                NORMAL  PNEUMONIA")
    print(f"  Actual NORMAL    {tn:5d}  {fp:5d}")
    print(f"         PNEUMONIA {fn:5d}  {tp:5d}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

print("\n" + "="*70)
print("Confusion Matrix Analysis")
print("="*70)

print_confusion_matrix(all_labels, all_preds_clean, "Clean Images")
print_confusion_matrix(all_labels, all_preds_adv, "Adversarial (FGSM)")
print_confusion_matrix(all_labels, all_preds_purified, "Purified Images")

print("="*70)

# ========== 詳細統計をCSVに保存 ==========
stats_df = pd.DataFrame({
    'true_label': all_labels,
    'pred_clean': all_preds_clean,
    'pred_adv': all_preds_adv,
    'pred_purified': all_preds_purified,
    'l2_norm_adv': l2_norms_adv,
    'linf_norm_adv': linf_norms_adv,
    'l2_norm_purified': l2_norms_purified,
    'linf_norm_purified': linf_norms_purified,
})

stats_df['attack_success'] = (stats_df['pred_adv'] != stats_df['true_label']).astype(int)
stats_df['purify_success'] = (stats_df['pred_purified'] == stats_df['true_label']).astype(int)
stats_df['defense_recovery'] = ((stats_df['attack_success'] == 1) & (stats_df['purify_success'] == 1)).astype(int)

csv_path = os.path.join(save_examples_dir, 'detailed_results.csv')
stats_df.to_csv(csv_path, index=False)
print(f"\n✅ Detailed statistics saved to: {csv_path}")

# サマリー統計をテキストファイルに保存
summary_path = os.path.join(save_examples_dir, 'summary_statistics.txt')
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("ChestXray - FGSM Attack + DDPM Defense Summary\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: ChestXray (NORMAL vs PNEUMONIA)\n")
    f.write(f"Attack Parameters:\n")
    f.write(f"  Method: FGSM (Fast Gradient Sign Method)\n")
    f.write(f"  Epsilon: {epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)\n\n")
    f.write(f"Purification Parameters:\n")
    f.write(f"  Method: DDPM (Denoising Diffusion Probabilistic Model)\n")
    f.write(f"  Start timestep (t): {start_t}\n")
    f.write(f"  Purification steps: {T_purify}\n")
    f.write(f"  Checkpoint: {ddpm_ckpt}\n\n")
    f.write(f"Classifier: {clf_ckpt}\n")
    f.write(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'N/A')}\n\n")
    f.write("-"*70 + "\n")
    f.write(f"Results (evaluated on {total} correctly classified images):\n")
    f.write("-"*70 + "\n")
    f.write(f"Clean Accuracy:      {clean_acc:.4f} ({correct_clean}/{total})\n")
    f.write(f"Adversarial Accuracy:{adv_acc:.4f} ({correct_adv}/{total})\n")
    f.write(f"Purified Accuracy:   {pur_acc:.4f} ({correct_purified}/{total})\n")
    f.write(f"Defense Improvement: {pur_acc - adv_acc:+.4f}\n")
    f.write(f"Attack Success Rate: {1 - adv_acc:.4f}\n")
    if (total - correct_adv) > 0:
        defense_rate = (correct_purified - correct_adv) / (total - correct_adv)
        f.write(f"Defense Success Rate:{defense_rate:.4f} (on attacked samples)\n")
    f.write("\n" + "="*70 + "\n")
    f.write("Perturbation Norms:\n")
    f.write("="*70 + "\n")
    f.write("Adversarial Perturbations (vs Clean):\n")
    f.write(f"  L2 norm:   mean={l2_norms_adv.mean():.6f}, std={l2_norms_adv.std():.6f}\n")
    f.write(f"             min={l2_norms_adv.min():.6f}, max={l2_norms_adv.max():.6f}\n")
    f.write(f"             median={np.median(l2_norms_adv):.6f}\n")
    f.write(f"  L∞ norm:   mean={linf_norms_adv.mean():.6f}, std={linf_norms_adv.std():.6f}\n")
    f.write(f"             min={linf_norms_adv.min():.6f}, max={linf_norms_adv.max():.6f}\n")
    f.write(f"             median={np.median(linf_norms_adv):.6f}\n\n")
    f.write("Purified Images (vs Clean):\n")
    f.write(f"  L2 norm:   mean={l2_norms_purified.mean():.6f}, std={l2_norms_purified.std():.6f}\n")
    f.write(f"             min={l2_norms_purified.min():.6f}, max={l2_norms_purified.max():.6f}\n")
    f.write(f"             median={np.median(l2_norms_purified):.6f}\n")
    f.write(f"  L∞ norm:   mean={linf_norms_purified.mean():.6f}, std={linf_norms_purified.std():.6f}\n")
    f.write(f"             min={linf_norms_purified.min():.6f}, max={linf_norms_purified.max():.6f}\n")
    f.write(f"             median={np.median(linf_norms_purified):.6f}\n\n")
    f.write("="*70 + "\n")
    f.write("Confusion Matrix Statistics:\n")
    f.write("="*70 + "\n")
    for name, preds in [("Clean", all_preds_clean), ("Adversarial", all_preds_adv), ("Purified", all_preds_purified)]:
        cm = confusion_matrix(all_labels, preds)
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
        specificity = tn/(tn+fp) if (tn+fp)>0 else 0.0
        f.write(f"\n{name} Images:\n")
        f.write(f"  TN: {tn:4d}  FP: {fp:4d}  FN: {fn:4d}  TP: {tp:4d}\n")
        f.write(f"  Precision:   {precision:.4f}\n")
        f.write(f"  Recall:      {recall:.4f}\n")
        f.write(f"  F1-Score:    {f1:.4f}\n")
        f.write(f"  Specificity: {specificity:.4f}\n")

print(f"✅ Summary statistics saved to: {summary_path}")

print("\n" + "="*70)
print("All evaluations completed successfully!")
print("="*70)
print(f"\nAll results saved in: {save_examples_dir}")
print(f"  - Triplet images: {save_triplets_dir}")
print(f"  - Detailed results: {csv_path}")
print(f"  - Summary: {summary_path}")
