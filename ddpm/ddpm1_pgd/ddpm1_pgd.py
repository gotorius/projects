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
from tqdm.auto import tqdm  # これを追加

DATA_DIR = '/mnt/data1/gotou/kaggle/pcam'
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


# In[ ]:


# 全ての検証画像を使用
print(f"Using all validation images ({len(val_dataset)} samples).")


# In[3]:


import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル定義（学習時と同じ）
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)  # 1ユニット出力
model = model.to(device)

# 重みロード
ckpt_path = "/mnt/data1/gotou/kaggle/pcam/best_model_weights.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model from {ckpt_path}")


# In[4]:


import torch
import torch.nn as nn

# --- 攻撃関数 ---
import torch.nn.functional as F

import torch

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# 前提: device, mean, std が外部で定義されている（投稿の定義をそのまま使えます）
# MEAN = np.array([...]); STD = np.array([...])
# mean = torch.tensor(MEAN).view(1,3,1,1).to(device)
# std  = torch.tensor(STD).view(1,3,1,1).to(device)

def denormalize(x, mean, std):
    # normalized -> pixel [0,1]
    return x * std + mean

def renormalize(x_pixel, mean, std):
    # pixel [0,1] -> normalized
    return (x_pixel - mean) / std

def pgd_attack_improved(
    model,
    images,
    labels,
    epsilon_pixel,      # scalar e.g. 8/255 or tensor shape [C]
    alpha_pixel,        # single-step size in pixel scale (e.g. 2/255)
    steps=10,
    device=None,
    mean_tensor=None,
    std_tensor=None,
    random_start=True,
    return_preds=True,
):
    """
    PGD (L_inf) attack that respects per-channel normalization.
    - images: normalized tensor [B,C,H,W]
    - labels: tensor [B] (0/1) or shape [B,1]
    - epsilon_pixel, alpha_pixel: pixel-scale floats (0..1) or 1D tensor per channel
    - steps: number of PGD iterations
    - random_start: initialize inside L_inf-ball uniformly
    Returns:
      adv_images (normalized, detached) and adv_preds (cpu LongTensor) if return_preds True
    """
    # use provided mean/std or globals
    if mean_tensor is None or std_tensor is None:
        mean_tensor_local = mean
        std_tensor_local = std
    else:
        mean_tensor_local = mean_tensor
        std_tensor_local = std_tensor

    if device is None:
        device = images.device

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    B, C, H, W = images.shape

    # convert eps and alpha to tensors on device and pixel space shape [1,C,1,1]
    def _to_channel_tensor(x):
        if not torch.is_tensor(x):
            t = torch.tensor(x, dtype=images.dtype, device=device)
        else:
            t = x.to(device).to(images.dtype)
        if t.ndim == 0:
            t = t.view(1, 1, 1, 1)  # scalar -> broadcastable
        elif t.ndim == 1 and t.numel() == C:
            t = t.view(1, C, 1, 1)
        else:
            # assume broadcastable already
            t = t.view(1, -1, 1, 1)
        return t

    eps_pixel_t = _to_channel_tensor(epsilon_pixel)   # pixel scale
    alpha_pixel_t = _to_channel_tensor(alpha_pixel)   # pixel scale

    # convert to normalized-space epsilon/alpha
    eps_norm = eps_pixel_t / std_tensor_local.to(images.dtype)
    alpha_norm = alpha_pixel_t / std_tensor_local.to(images.dtype)

    # get pixel-space original images
    orig_pixel = denormalize(images, mean_tensor_local, std_tensor_local)  # [B,C,H,W], pixel in [0,1]

    # random start (in pixel space) then renormalize
    if random_start:
        # uniform perturbation in [-eps, eps] per channel
        uni = torch.empty_like(orig_pixel).uniform_(-1.0, 1.0)
        # scale to [-eps, eps] per channel
        rand_pert = uni * eps_pixel_t
        adv_pixel = torch.clamp(orig_pixel + rand_pert, 0.0, 1.0)
        adv_images = renormalize(adv_pixel, mean_tensor_local, std_tensor_local).detach()
    else:
        adv_images = images.clone().detach()

    # main loop
    adv_images = adv_images.to(device)
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        if outputs.ndim > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.data

        # sign update in normalized space
        adv_images = adv_images + alpha_norm * grad.sign()

        # project back to L_inf ball around original pixel image:
        # compute pixel space, clamp difference to [-eps_pixel, eps_pixel]
        adv_pixel = denormalize(adv_images, mean_tensor_local, std_tensor_local)
        eta = torch.clamp(adv_pixel - orig_pixel, min=-eps_pixel_t, max=eps_pixel_t)
        adv_pixel = torch.clamp(orig_pixel + eta, 0.0, 1.0)

        # back to normalized space and detach for next iter
        adv_images = renormalize(adv_pixel, mean_tensor_local, std_tensor_local).detach()

        # cleanup grad
        adv_images.requires_grad = False
        del grad, outputs, loss, eta, adv_pixel
        torch.cuda.empty_cache()

    # return preds if requested
    if return_preds:
        with torch.no_grad():
            adv_out = model(adv_images)
            if adv_out.ndim > 1 and adv_out.shape[1] == 1:
                adv_out = adv_out.squeeze(1)
            adv_probs = torch.sigmoid(adv_out)
            adv_preds = (adv_probs > 0.5).long().cpu()
        return adv_images, adv_preds

    return adv_images



# ddpm_purify_eval_pipeline.py
import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ---------- Basic device / paths ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 学習済みDDPM（ddpm.pyで学習したチェックポイント）
ddpm_ckpt = "/mnt/data1/gotou/kaggle/ddpm/ddpm1_fgsm/ddpm_epoch10.pth"
clf_ckpt  = "/mnt/data1/gotou/kaggle/pcam/best_model_weights.pth"

# --- モデル定義（ddpm.pyと一致） ---
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

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )
        # down
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        # bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        # up
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
        d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb)
        d4 = self.down4(e4)
        b = self.bot1(d4, t_emb)
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

# --- βスケジュール（学習時と同じ） ---
T_steps = 1000
betas = torch.linspace(1e-4, 0.02, T_steps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
# posterior_variance を使用
posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# --- load ddpm ---
ddpm = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(device)
ckpt = torch.load(ddpm_ckpt, map_location=device)
# ddpm.py の保存形式に対応
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    try:
        # EMA があれば優先的に使用
        if 'ema_state_dict' in ckpt and isinstance(ckpt['ema_state_dict'], dict):
            ddpm.load_state_dict(ckpt['ema_state_dict'], strict=False)
        else:
            ddpm.load_state_dict(ckpt['model_state_dict'])
    except Exception:
        ddpm.load_state_dict(ckpt['model_state_dict'])
else:
    ddpm.load_state_dict(ckpt)
ddpm.eval()

# --- diffusion_purify（x0再構成でノイズを抑える） ---
@torch.no_grad()
def diffusion_purify(x_adv_minus1to1, model, start_t=600, T_purify=100, eta=0.0, clamp_each_step=True, out_mode='x0'):
    """
    部分逆拡散後に x0_hat を再構成して返す（t>0で止めてもノイズを抑制）。
    out_mode: 'x0' で x0 推定を返す / 'x_t' で時刻 t のサンプルを返す
    """
    b = x_adv_minus1to1.size(0)
    t0 = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1to1)
    sqrt_a_bar_t0 = torch.sqrt(alphas_cumprod[t0]).view(-1,1,1,1)
    sqrt_1m_a_bar_t0 = torch.sqrt(1.0 - alphas_cumprod[t0]).view(-1,1,1,1)
    x_t = sqrt_a_bar_t0 * x_adv_minus1to1 + sqrt_1m_a_bar_t0 * noise

    eps_pred_final = None
    t_final = start_t
    for t_ in range(start_t, max(start_t - T_purify, 0), -1):
        t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)
        alpha_t = alphas[t_]
        alpha_bar_t = alphas_cumprod[t_]
        # DDPMの平均
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred)
        if t_ > 0:
            z = torch.randn_like(x_t)
            sigma = eta * torch.sqrt(posterior_variance[t_])
            x_t = mean + sigma * z
        else:
            x_t = mean
        if clamp_each_step:
            x_t = torch.clamp(x_t, -1.0, 1.0)
        # 最終時刻の記録
        eps_pred_final = eps_pred
        t_final = t_

    if out_mode == 'x0':
        # 最終時刻 t_final における x0 再構成
        alpha_bar_tf = alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_pred_final) / torch.sqrt(alpha_bar_tf + 1e-12)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        return x0_hat
    else:
        return x_t

# ---------- normalization utilities ----------
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
# DDPM学習時の正規化（mean=0.5, std=0.5 で [-1,1] スケール）
ddpm_mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1).to(device)
ddpm_std  = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1).to(device)

def unnormalize(x_norm):
    return x_norm * imagenet_std + imagenet_mean

def renormalize(x, mean, std):
    return (x - mean) / std

# 224x224での前後処理（DDPM学習時の正規化に合わせる）
def prepare_for_diffusion_from_norm(x_norm, target_size=224):
    x_pixel = unnormalize(x_norm)
    x_resized = F.interpolate(x_pixel, size=(target_size, target_size), mode="bilinear", align_corners=False)
    x_minus1to1 = (x_resized - ddpm_mean) / ddpm_std
    return x_minus1to1

def recover_from_diffusion_to_norm(x_minus1to1, out_size=224):
    x_pixel = x_minus1to1 * ddpm_std + ddpm_mean
    x_pixel = torch.clamp(x_pixel, 0.0, 1.0)
    x_resized = F.interpolate(x_pixel, size=(out_size, out_size), mode="bilinear", align_corners=False)
    x_norm = renormalize(x_resized, imagenet_mean, imagenet_std)
    return x_norm

# ---------- load classifier ----------
clf = models.resnet50(pretrained=False)
clf.fc = nn.Linear(clf.fc.in_features, 1)
clf.load_state_dict(torch.load(clf_ckpt, map_location=device))
clf = clf.to(device)
clf.eval()
print("Loaded classifier and ddpm (ddpm_out epoch20).")

# ---------- evaluation loop ----------
epsilon_pixel = 8/255.0
alpha_pixel = 2/255.0  # PGD step size
pgd_steps = 10         # PGD iterations
start_t = 80    # 浄化開始時刻
T_purify = 50    # 逆拡散反復数（短縮して高速化）
save_examples_dir = "purify_examples"
os.makedirs(save_examples_dir, exist_ok=True)
# per-image 出力用ディレクトリ
save_triplets_dir = os.path.join(save_examples_dir, "triplets")
save_clean_dir    = os.path.join(save_examples_dir, "clean")
save_adv_dir      = os.path.join(save_examples_dir, "adversarial")
save_pur_dir      = os.path.join(save_examples_dir, "purified")
for d in [save_triplets_dir, save_clean_dir, save_adv_dir, save_pur_dir]:
    os.makedirs(d, exist_ok=True)

# Use smaller loader if defined
# Use all validation images
EVAL_LOADER = val_loader

# 画像保存の設定
MAX_IMAGES_TO_SAVE = 20  # 最初の何枚を保存するか
saved_image_count = 0

# グローバルindex
global_idx = 0

total = 0
correct_clean = 0
correct_adv = 0
correct_purified = 0

# 混同行列用のリスト
all_labels = []
all_preds_clean = []
all_preds_adv = []
all_preds_purified = []

# ノルム計算用のリスト
l2_norms_adv = []
linf_norms_adv = []
l2_norms_purified = []
linf_norms_purified = []

# loop
for batch_idx, (images_norm, labels) in enumerate(tqdm(EVAL_LOADER, desc="Eval loop (all validation images)")):
    images_norm = images_norm.to(device)
    labels = labels.to(device).long().view(-1)
    b = images_norm.size(0)
    
    # 1) clean prediction
    with torch.no_grad():
        logits_clean = clf(images_norm)
        if logits_clean.ndim > 1 and logits_clean.shape[1] == 1:
            logits_clean = logits_clean.squeeze(1)
        probs_clean = torch.sigmoid(logits_clean)
        preds_clean = (probs_clean > 0.5).long()
    
    # 元画像で正解した画像のみフィルタリング
    correct_mask = (preds_clean.cpu() == labels.cpu())
    correct_indices = torch.where(correct_mask)[0]
    
    # 正解した画像がない場合はスキップ
    if len(correct_indices) == 0:
        continue
    
    # 正解した画像のみを選択
    images_norm_correct = images_norm[correct_indices]
    labels_correct = labels[correct_indices]
    preds_clean_correct = preds_clean[correct_indices]
    
    # 統計を更新
    total += len(correct_indices)
    correct_clean += len(correct_indices)
    all_labels.extend(labels_correct.cpu().numpy())
    all_preds_clean.extend(preds_clean_correct.cpu().numpy())

    # 2) adversarial example (PGD) - 正解した画像のみ
    adv_images_norm, adv_preds_from_attack = pgd_attack_improved(
        model=clf,
        images=images_norm_correct,
        labels=labels_correct,
        epsilon_pixel=epsilon_pixel,
        alpha_pixel=alpha_pixel,
        steps=pgd_steps,
        device=device,
        mean_tensor=imagenet_mean,
        std_tensor=imagenet_std,
        random_start=True,
        return_preds=True
    )
    adv_images_norm = adv_images_norm.to(device)
    adv_preds_from_attack = adv_preds_from_attack.to(device)
    correct_adv += (adv_preds_from_attack.cpu() == labels_correct.cpu()).sum().item()
    all_preds_adv.extend(adv_preds_from_attack.cpu().numpy())
    
    # L2 と L∞ ノルムを計算（pixel space）
    clean_pixel = unnormalize(images_norm_correct.detach())
    adv_pixel = unnormalize(adv_images_norm.detach())
    diff_adv = (adv_pixel - clean_pixel).view(len(correct_indices), -1)
    l2_adv = torch.norm(diff_adv, p=2, dim=1).cpu().numpy()
    linf_adv = torch.norm(diff_adv, p=float('inf'), dim=1).cpu().numpy()
    l2_norms_adv.extend(l2_adv)
    linf_norms_adv.extend(linf_adv)

    # 3) prepare for diffusion & purify (x0再構成で返す)
    x_adv_for_diff = prepare_for_diffusion_from_norm(adv_images_norm, target_size=224)
    purified_minus1to1 = diffusion_purify(x_adv_for_diff, ddpm, start_t=start_t, T_purify=T_purify, eta=0.0, out_mode='x0')

    # 4) recover to classifier-normalized inputs and classify
    purified_norm = recover_from_diffusion_to_norm(purified_minus1to1, out_size=224)
    with torch.no_grad():
        logits_pur = clf(purified_norm)
        if logits_pur.ndim > 1 and logits_pur.shape[1] == 1:
            logits_pur = logits_pur.squeeze(1)
        probs_pur = torch.sigmoid(logits_pur)
        preds_pur = (probs_pur > 0.5).long()
        correct_purified += (preds_pur.cpu() == labels_correct.cpu()).sum().item()
        all_preds_purified.extend(preds_pur.cpu().numpy())
        
        # 浄化後のL2/L∞ノルム計算
        pur_pixel = (purified_minus1to1.detach() + 1.0) / 2.0
        # resize to match clean_pixel size for fair comparison
        pur_pixel_resized = F.interpolate(pur_pixel, size=(224,224), mode="bilinear", align_corners=False)
        diff_pur = (pur_pixel_resized - clean_pixel).view(len(correct_indices), -1)
        l2_pur = torch.norm(diff_pur, p=2, dim=1).cpu().numpy()
        linf_pur = torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy()
        l2_norms_purified.extend(l2_pur)
        linf_norms_purified.extend(linf_pur)

    # 5) save per-image outputs (original/adv/purified) and triplet tiles - 最初の何枚かのみ
    if saved_image_count < MAX_IMAGES_TO_SAVE:
        clean_pixel = unnormalize(images_norm_correct.detach()).clamp(0,1)
        adv_pixel   = unnormalize(adv_images_norm.detach()).clamp(0,1)
        pur_pixel   = (purified_minus1to1.detach() + 1.0) / 2.0
        # unify to 224x224 for saving
        clean_resized = F.interpolate(clean_pixel, size=(224,224), mode="bilinear", align_corners=False)
        adv_resized   = F.interpolate(adv_pixel,   size=(224,224), mode="bilinear", align_corners=False)
        pur_resized   = F.interpolate(pur_pixel,   size=(224,224), mode="bilinear", align_corners=False)

        for i in range(len(correct_indices)):
            if saved_image_count >= MAX_IMAGES_TO_SAVE:
                break
            
            # 元のインデックスを取得
            original_idx = correct_indices[i].item()
            
            # 画像ID
            try:
                img_id = str(val_df.iloc[global_idx + original_idx, 0])
            except Exception:
                img_id = f"idx{saved_image_count:05d}"
            label_int = int(labels_correct[i].item())
            
            # individual saves
            save_image(clean_resized[i], os.path.join(save_clean_dir, f"{saved_image_count:05d}_{img_id}_label{label_int}_clean.png"))
            save_image(adv_resized[i],   os.path.join(save_adv_dir,   f"{saved_image_count:05d}_{img_id}_label{label_int}_adv.png"))
            save_image(pur_resized[i],   os.path.join(save_pur_dir,   f"{saved_image_count:05d}_{img_id}_label{label_int}_purified.png"))
            # triplet tile
            row = torch.cat([clean_resized[i], adv_resized[i], pur_resized[i]], dim=2)
            save_image(row, os.path.join(save_triplets_dir, f"{saved_image_count:05d}_{img_id}_triplet.png"))
            saved_image_count += 1
    
    global_idx += b

# ---------- final results ----------
clean_acc = correct_clean / total if total > 0 else 0.0
adv_acc = correct_adv / total if total > 0 else 0.0
pur_acc = correct_purified / total if total > 0 else 0.0

# ノルム統計
l2_norms_adv = np.array(l2_norms_adv)
linf_norms_adv = np.array(linf_norms_adv)
l2_norms_purified = np.array(l2_norms_purified)
linf_norms_purified = np.array(linf_norms_purified)

print("\n" + "="*70)
print("==== Results (Clean Images Only - PGD Attack) ====")
print("="*70)
print(f"Total samples evaluated: {total} (元画像で正解したもののみ)")
print(f"Attack: PGD with epsilon={epsilon_pixel:.4f}, alpha={alpha_pixel:.4f}, steps={pgd_steps}")
print(f"Purification: DDPM start_t={start_t}, T_purify={T_purify}")
print("-"*70)
print(f"Clean accuracy:     {clean_acc:.4f} (常に1.0)")
print(f"Adv (PGD) accuracy: {adv_acc:.4f}")
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

print(f"\nSaved {saved_image_count} example images to: {save_examples_dir}")
print(f"- Individual images: {save_clean_dir}, {save_adv_dir}, {save_pur_dir}")
print(f"- Triplets per image: {save_triplets_dir}")

# ---------- 混同行列の計算と可視化 ----------
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title}:")
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

print("\n" + "="*50)
print("Confusion Matrix Analysis")
print("="*50)

plot_confusion_matrix(
    all_labels, all_preds_clean,
    "Confusion Matrix - Clean Images",
    os.path.join(save_examples_dir, "confusion_matrix_clean.png")
)

plot_confusion_matrix(
    all_labels, all_preds_adv,
    "Confusion Matrix - Adversarial (PGD)",
    os.path.join(save_examples_dir, "confusion_matrix_adversarial.png")
)

plot_confusion_matrix(
    all_labels, all_preds_purified,
    "Confusion Matrix - Purified Images",
    os.path.join(save_examples_dir, "confusion_matrix_purified.png")
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cm_clean = confusion_matrix(all_labels, all_preds_clean)
cm_adv = confusion_matrix(all_labels, all_preds_adv)
cm_pur = confusion_matrix(all_labels, all_preds_purified)
titles = ['Clean Images', 'Adversarial (PGD)', 'Purified Images']
cms = [cm_clean, cm_adv, cm_pur]
accs = [clean_acc, adv_acc, pur_acc]
for ax, cm, title, acc in zip(axes, cms, titles, accs):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    ax.set_title(f'{title}\nAccuracy: {acc:.4f}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(save_examples_dir, 'confusion_matrix_comparison.png'), 
            dpi=150, bbox_inches='tight')
plt.show()
print(f"\n✅ All confusion matrices saved to: {save_examples_dir}")

# ---------- ノルム分布の可視化 ----------
print("\n" + "="*70)
print("Generating norm distribution plots...")
print("="*70)

# L2ノルムのヒストグラム
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# L2 ノルム - Adversarial
axes[0, 0].hist(l2_norms_adv, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0, 0].axvline(l2_norms_adv.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {l2_norms_adv.mean():.4f}')
axes[0, 0].set_title('L2 Norm Distribution - Adversarial Perturbations', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('L2 Norm')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# L∞ ノルム - Adversarial
axes[0, 1].hist(linf_norms_adv, bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(linf_norms_adv.mean(), color='darkorange', linestyle='--', linewidth=2, label=f'Mean: {linf_norms_adv.mean():.4f}')
axes[0, 1].set_title('L∞ Norm Distribution - Adversarial Perturbations', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('L∞ Norm')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# L2 ノルム - Purified
axes[1, 0].hist(l2_norms_purified, bins=50, color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].axvline(l2_norms_purified.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {l2_norms_purified.mean():.4f}')
axes[1, 0].set_title('L2 Norm Distribution - Purified vs Clean', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('L2 Norm')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# L∞ ノルム - Purified
axes[1, 1].hist(linf_norms_purified, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(linf_norms_purified.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {linf_norms_purified.mean():.4f}')
axes[1, 1].set_title('L∞ Norm Distribution - Purified vs Clean', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('L∞ Norm')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_examples_dir, 'norm_distributions.png'), dpi=150, bbox_inches='tight')
plt.show()

# ボックスプロット比較
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# L2 ノルムの比較
axes[0].boxplot([l2_norms_adv, l2_norms_purified], 
                labels=['Adversarial', 'Purified'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_title('L2 Norm Comparison', fontsize=12, fontweight='bold')
axes[0].set_ylabel('L2 Norm')
axes[0].grid(True, alpha=0.3, axis='y')

# L∞ ノルムの比較
axes[1].boxplot([linf_norms_adv, linf_norms_purified], 
                labels=['Adversarial', 'Purified'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('L∞ Norm Comparison', fontsize=12, fontweight='bold')
axes[1].set_ylabel('L∞ Norm')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(save_examples_dir, 'norm_boxplots.png'), dpi=150, bbox_inches='tight')
plt.show()

# ---------- 精度とノルムの散布図 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 攻撃成功/失敗とL2ノルムの関係
attack_success = (np.array(all_preds_adv) != np.array(all_labels))
axes[0].scatter(l2_norms_adv[~attack_success], np.zeros(sum(~attack_success)), 
                c='green', marker='o', alpha=0.5, label='Attack Failed')
axes[0].scatter(l2_norms_adv[attack_success], np.ones(sum(attack_success)), 
                c='red', marker='x', alpha=0.5, label='Attack Succeeded')
axes[0].set_xlabel('L2 Norm')
axes[0].set_yticks([0, 1])
axes[0].set_yticklabels(['Attack Failed', 'Attack Succeeded'])
axes[0].set_title('Attack Success vs L2 Norm', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 浄化成功/失敗とL2ノルムの関係
purify_success = (np.array(all_preds_purified) == np.array(all_labels))
axes[1].scatter(l2_norms_purified[~purify_success], np.zeros(sum(~purify_success)), 
                c='red', marker='x', alpha=0.5, label='Purify Failed')
axes[1].scatter(l2_norms_purified[purify_success], np.ones(sum(purify_success)), 
                c='green', marker='o', alpha=0.5, label='Purify Succeeded')
axes[1].set_xlabel('L2 Norm (Purified vs Clean)')
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(['Purify Failed', 'Purify Succeeded'])
axes[1].set_title('Purification Success vs L2 Norm', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_examples_dir, 'success_vs_norms.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Norm distribution plots saved to: {save_examples_dir}")
print("   - norm_distributions.png: L2/L∞ヒストグラム")
print("   - norm_boxplots.png: ボックスプロット比較")
print("   - success_vs_norms.png: 成功率とノルムの関係")

# ---------- 詳細統計をCSVに保存 ----------
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

# 攻撃成功/失敗フラグ
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
    f.write("PGD Attack + DDPM Purification - Summary Statistics\n")
    f.write("="*70 + "\n\n")
    f.write(f"Attack Parameters:\n")
    f.write(f"  Method: PGD (Projected Gradient Descent)\n")
    f.write(f"  Epsilon: {epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)\n")
    f.write(f"  Alpha: {alpha_pixel:.4f} ({alpha_pixel*255:.1f}/255)\n")
    f.write(f"  Steps: {pgd_steps}\n")
    f.write(f"  Random Start: True\n\n")
    f.write(f"Purification Parameters:\n")
    f.write(f"  Method: DDPM (Denoising Diffusion Probabilistic Model)\n")
    f.write(f"  Start timestep (t): {start_t}\n")
    f.write(f"  Purification steps: {T_purify}\n")
    f.write(f"  Checkpoint: {ddpm_ckpt}\n\n")
    f.write("-"*70 + "\n")
    f.write(f"Results (evaluated on {total} correctly classified images):\n")
    f.write("-"*70 + "\n")
    f.write(f"Clean Accuracy:      {clean_acc:.4f} ({correct_clean}/{total})\n")
    f.write(f"Adversarial Accuracy:{adv_acc:.4f} ({correct_adv}/{total})\n")
    f.write(f"Purified Accuracy:   {pur_acc:.4f} ({correct_purified}/{total})\n")
    f.write(f"Defense Improvement: {pur_acc - adv_acc:+.4f}\n")
    f.write(f"Attack Success Rate: {1 - adv_acc:.4f}\n")
    f.write(f"Defense Success Rate:{(correct_purified - correct_adv) / (total - correct_adv):.4f} (on attacked samples)\n\n")
    f.write("="*70 + "\n")
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