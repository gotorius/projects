#!/usr/bin/env python
# coding: utf-8

# # ImageNet事前訓練拡散モデルを医療画像分類で用いる

# はじめに医療データセットに対してResNet50で分類を行う。
# なお事前に得ていた事前訓練済みの分類機を用いる。データセットはkaggleのデータセットを用いる。

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
from tqdm.auto import tqdm  # これを追加

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


"""
# EarlyStopping & モデル保存
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='best_model_weights.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.best_model = None
        self.path = path

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''ベストモデルを保存'''
        torch.save(model.state_dict(), self.path)
        # deepcopy して参照を切る
        self.best_model = copy.deepcopy(model.state_dict())
        if self.verbose:
            print(f'Validation accuracy improved → saving model to {self.path}')

            """


# In[2]:


import torch
import torch.nn as nn
from torchvision import models

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


# In[3]:


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

def fgsm_attack_improved(model, images, labels, epsilon_pixel, device,
                         mean_tensor=None, std_tensor=None, return_preds=True):
    """
    FGSM attack that respects per-channel normalization.
    Args:
      model: nn.Module, expects normalized inputs (same normalization used here)
      images: normalized tensor [B,C,H,W] (already normalized by mean/std)
      labels: tensor [B] (0/1) or shape [B,1]
      epsilon_pixel: float (e.g. 8/255) or tensor broadcastable to channels (pixel-scale)
      device: torch.device
      mean_tensor, std_tensor: tensors shaped [1,C,1,1] on device (if None, must be global)
      return_preds: if True, also return predicted labels on adversarial images
    Returns:
      adv_images: tensor [B,C,H,W] (normalized, detached)
      adv_preds (optional): LongTensor [B] predicted labels on adv_images (cpu)
    """
    # use provided mean/std or expect global variables `mean`/`std`
    if mean_tensor is None or std_tensor is None:
        # these should exist in your scope (as in your snippet)
        mean_tensor_local = mean
        std_tensor_local = std
    else:
        mean_tensor_local = mean_tensor
        std_tensor_local = std_tensor

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True

    # model output: assume logits for binary classification -> shape [B] or [B,1]
    outputs = model(images)
    if outputs.ndim > 1 and outputs.shape[1] == 1:
        outputs = outputs.squeeze(1)  # [B]
    # binary cross entropy with logits
    loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
    model.zero_grad()
    loss.backward()
    grad = images.grad.data  # gradient in normalized space

    # sign of gradient in normalized space
    grad_sign = grad.sign()

    # convert epsilon_pixel (pixel scale: 0..1) -> normalized-space epsilon per channel
    # support scalar epsilon_pixel or iterable per-channel
    if not torch.is_tensor(epsilon_pixel):
        eps_pixel_tensor = torch.tensor(epsilon_pixel, dtype=images.dtype, device=device)
    else:
        eps_pixel_tensor = epsilon_pixel.to(device).to(images.dtype)

    # make eps_norm shaped [1,C,1,1]
    if eps_pixel_tensor.ndim == 0:
        eps_norm = (eps_pixel_tensor / std_tensor_local).view(1, -1, 1, 1)
    elif eps_pixel_tensor.ndim == 1 and eps_pixel_tensor.numel() == std_tensor_local.shape[1]:
        eps_norm = (eps_pixel_tensor.view(1, -1, 1, 1) / std_tensor_local)
    else:
        # fallback: try to broadcast
        eps_norm = (eps_pixel_tensor / std_tensor_local)

    # create adversarial images in normalized space
    adv_images = images + eps_norm * grad_sign

    # clamp in pixel space
    adv_pixel = denormalize(adv_images, mean_tensor_local, std_tensor_local)
    adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)

    # back to normalized space
    adv_images = renormalize(adv_pixel, mean_tensor_local, std_tensor_local).detach()
    adv_images.requires_grad = False

    if return_preds:
        with torch.no_grad():
            adv_out = model(adv_images)
            if adv_out.ndim > 1 and adv_out.shape[1] == 1:
                adv_out = adv_out.squeeze(1)
            adv_probs = torch.sigmoid(adv_out)
            adv_preds = (adv_probs > 0.5).long().cpu()
        # cleanup
        del grad, grad_sign, adv_out, adv_pixel, eps_norm, eps_pixel_tensor, loss
        torch.cuda.empty_cache()
        return adv_images, adv_preds

    # cleanup
    del grad, grad_sign, adv_pixel, eps_norm, eps_pixel_tensor, loss
    torch.cuda.empty_cache()
    return adv_images


# --- 評価関数 ---
def evaluate_clean_and_fgsm(model, val_loader, device, epsilon_pixel, mean, std):
    adv_correct, adv_total = 0, 0
    orig_correct, orig_total = 0, 0
    processed = 0
    max_samples = 100  # ← 100枚に制限

    model.eval()
    for images, labels in val_loader:
        if processed >= max_samples:
            break  # ← 100枚に達したら終了
        images, labels = images.to(device), labels.to(device)

        # 元画像での予測
        outputs = model(images)
        if outputs.ndim > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        preds = (torch.sigmoid(outputs) > 0.5).long()

        correct_mask = (preds == labels.long())
        if correct_mask.sum() == 0:
            continue

        correct_images = images[correct_mask]
        correct_labels = labels[correct_mask]

        orig_preds = preds[correct_mask]
        orig_correct += (orig_preds == correct_labels).sum().item()
        orig_total += correct_labels.size(0)

        # FGSM攻撃（正規化対応版）
        adv_images, adv_preds = fgsm_attack_improved(
            model, correct_images, correct_labels, epsilon_pixel, device,
            mean_tensor=mean, std_tensor=std, return_preds=True
        )
        adv_correct += (adv_preds == correct_labels.cpu()).sum().item()
        adv_total += correct_labels.size(0)

    orig_acc = orig_correct / orig_total * 100 if orig_total > 0 else 0
    adv_acc = adv_correct / adv_total * 100 if adv_total > 0 else 0
    print(f"Original Accuracy (on selected correct samples): {orig_acc:.2f}% ({orig_correct}/{orig_total})")
    print(f"Adversarial Accuracy (on originally correct samples): {adv_acc:.2f}% ({adv_correct}/{adv_total})")
    return orig_acc, adv_acc




# In[ ]:


# evaluate_clean_and_fgsm(model, val_loader, device, epsilon_pixel=30/255, mean=mean, std=std)
# original_acc, adversarial_acc = evaluate_clean_and_fgsm(model, val_loader, device, epsilon_pixel=0.01, mean=mean, std=std)


# In[ ]:


"""import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch

# --- 設定 ---
num_display = 10  # 先頭何枚を可視化するか

# 表示用 unnormalize（正規化 -> pixel [0,1]）
def unnormalize_tensor(x):
    return x * std.to(x.device) + mean.to(x.device)

# データを少しだけ取得
it = iter(val_loader)
collected = 0
rows = []

while collected < num_display:
    try:
        batch = next(it)
    except StopIteration:
        break
    images, labels = batch
    B = images.shape[0]
    take = min(B, num_display - collected)
    imgs = images[:take].to(device)
    labs = labels[:take].to(device)

    # FGSM攻撃
    adv_images, adv_preds = fgsm_attack_improved(
        model, imgs, labs, epsilon_pixel=8/255, device=device,
        mean_tensor=mean, std_tensor=std, return_preds=True
    )

    rows.append({
        "clean": imgs.detach().cpu(),
        "adv": adv_images.detach().cpu()
    })

    collected += take

# --- 描画 ---
plt.figure(figsize=(6, 3 * num_display))
idx = 0
for r in rows:
    B = r["clean"].shape[0]
    for i in range(B):
        # 元画像
        plt.subplot(num_display, 2, idx * 2 + 1)
        img_clean = unnormalize_tensor(r["clean"][i].to(device)).cpu()
        if img_clean.ndim == 4:
            img_clean = img_clean.squeeze(0)
        plt.imshow(TF.to_pil_image(img_clean))
        plt.title("Clean")
        plt.axis("off")

        # 敵対画像
        plt.subplot(num_display, 2, idx * 2 + 2)
        img_adv = unnormalize_tensor(r["adv"][i].to(device)).cpu()
        if img_adv.ndim == 4:
            img_adv = img_adv.squeeze(0)
        plt.imshow(TF.to_pil_image(img_adv))
        plt.title("Adversarial")
        plt.axis("off")

        idx += 1

plt.tight_layout()
plt.show()"""


# In[ ]:


# fgsm_plus_diffusion_eval.py
import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets, models
import torch.nn as nn
import torchvision
import math
import sys
sys.path.append("/mnt/data1/gotou/projects/guided-diffusion")  # クローンした場所に置き換え

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model_and_diffusion


# --- guided-diffusion imports (make sure repo is available in PYTHONPATH) ---
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model_and_diffusion

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)

# ---------------- user settings ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64 # VRAM に合わせて調整。256x256 diffusion は重いので小さめ推奨
epsilon = 0.01
num_samples = 1000 # correct_top1_images.txt の先頭何枚を使うか

# Diffusion settings
use_ddim = True
ddim_steps = 100     # 速くしたければ 25~50。品質を優先するなら 100+
real_step = 30     # reverse の深さ（実験して調整）
blend_alpha = 0.6
model_path = "/mnt/data1/gotou/projects/guided-diffusion/256x256_diffusion_uncond.pt"  # 事前チェックポイント


# ---------------- transforms (あなたの分類用) ----------------
# 既存の分類 transform と同じものを使う想定
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ImageNet 正規化パラ
imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)

# ---------------- load classifier ----------------
clf = models.resnet50(pretrained=True).to(device)
clf.eval()

# ---------------- load paths ----------------
with open("correct_image_paths.txt") as f:
    all_paths = [p.strip() for p in f]
paths = all_paths[:num_samples]

# ---------------- guided-diffusion model load ----------------
defaults = model_and_diffusion_defaults()
defaults.update({
    "image_size": 256,
    "num_channels": 256,
    "num_res_blocks": 2,
    "num_heads": 4,
    "num_head_channels": -1,
    "learn_sigma": True,
    "class_cond": False,
    "use_fp16": False,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "attention_resolutions": "32,16,8",
    "diffusion_steps": 1000,
    "timestep_respacing": f"ddim{ddim_steps}" if use_ddim else "1000",  # ★追加
})

print("Creating guided-diffusion model...")
diff_model, diffusion = create_model_and_diffusion(**defaults)

print("Loading checkpoint:", model_path)
ckpt = torch.load(model_path, map_location="cpu")
diff_model.load_state_dict(ckpt)

diff_model.to(device)
diff_model.eval()
print("✅ Model loaded successfully!")


# In[7]:


import torch
import torch.nn.functional as F

# ---------------- normalization ----------------
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)

def unnormalize(x):
    return x * imagenet_std + imagenet_mean

def normalize(x):
    return (x - imagenet_mean) / imagenet_std

# --- resize & range conversion ---
def prepare_for_diffusion(x):
    # [0,1] -> [-1,1], resize 224->256
    x = F.interpolate(x, size=(256,256), mode="bilinear", align_corners=False)
    return (x - 0.5) * 2.0

def recover_from_diffusion(x):
    # [-1,1] -> [0,1], resize 256->224
    x = (x + 1) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
    return x

def diffusion_denoise(x_noisy, steps=50):
    """
    x_noisy: [-1,1], (B,3,256,256)
    steps: denoise の反復回数 (少なめでもOK)
    """
    with torch.no_grad():
        sample = diffusion.p_sample_loop(
            diff_model,
            (x_noisy.shape[0], 3, 256, 256),
            noise=x_noisy,
            clip_denoised=True,
            model_kwargs={},
            progress=False,
            device=device,
            cond_fn=None,
        )
    return sample

dataloader = val_loader  # または train_loader



# In[8]:


# Purify function using diffusion (DDIM if specified)
@torch.no_grad()
def purify_with_diffusion(x_neg1_1, diffusion, diff_model, device,
                          use_ddim=True, real_step=500, blend_alpha=0.5,
                          save_debug=False, debug_prefix="dbg"):
    """
    x_neg1_1: (B,3,256,256) in [-1,1]
    returns recon in [-1,1]
    blend_alpha: weight for recon: output = blend_alpha * recon + (1-blend_alpha) * x_neg1_1
    save_debug: if True save example images to /mnt/data1/gotou/debug/
    """
    B, C, H, W = x_neg1_1.shape
    model_kwargs = {}

    try:
        T = diffusion.num_timesteps
    except Exception:
        T = 1000
    t_val = max(1, min(real_step, T-1))
    t = torch.full((B,), t_val, dtype=torch.long, device=device)

    noise = torch.randn_like(x_neg1_1)
    # forward: x_t
    try:
        x_t = diffusion.q_sample(x_neg1_1, t, noise=noise)
    except TypeError:
        x_t = diffusion.q_sample(x_neg1_1, t)

    # reverse: use available sample loop (many impls accept noise=start state)
    if use_ddim:
        recon = diffusion.ddim_sample_loop(
            diff_model,
            (B, C, H, W),
            noise=x_t,
            model_kwargs=model_kwargs,
            clip_denoised=True,
        )
    else:
        recon = diffusion.p_sample_loop(
            diff_model,
            (B, C, H, W),
            noise=x_t,
            model_kwargs=model_kwargs,
            clip_denoised=True,
        )

    recon = torch.clamp(recon, -1.0, 1.0)

    # Blend with original (x_neg1_1) to avoid over-distortion (tunable)
    if blend_alpha < 1.0:
        recon = blend_alpha * recon + (1.0 - blend_alpha) * x_neg1_1
        recon = torch.clamp(recon, -1.0, 1.0)

    # Optional debug save: save first image in batch (clean/adv/x_t/recon)
    if save_debug:
        debug_dir = "/mnt/data1/gotou/debug"
        os.makedirs(debug_dir, exist_ok=True)
        import torchvision.utils as vutils
        # x_neg1_1: original input to diffusion ([-1,1]) -> to [0,1]
        unnorm_in = (x_neg1_1[0].cpu() + 1.0) / 2.0
        unnorm_x_t = (x_t[0].cpu() + 1.0) / 2.0
        unnorm_recon = (recon[0].cpu() + 1.0) / 2.0
        vutils.save_image(unnorm_in, os.path.join(debug_dir, f"{debug_prefix}_in.png"))
        vutils.save_image(unnorm_x_t, os.path.join(debug_dir, f"{debug_prefix}_xt.png"))
        vutils.save_image(unnorm_recon, os.path.join(debug_dir, f"{debug_prefix}_recon.png"))

    return recon


# In[10]:


from tqdm import tqdm
import pandas as pd
from PIL import Image
import os

# ラベルCSVを読み込み
labels_df = pd.read_csv('/mnt/data1/gotou/projects/Medical/kaggledata/train_labels.csv')
labels_dict = dict(zip(labels_df['id'], labels_df['label']))

with open("correct_image_paths.txt") as f:
    all_paths = [p.strip() for p in f]
paths = all_paths[:num_samples]

clean_correct = 0
adv_correct = 0
purified_correct = 0
total = 0


# In[ ]:


"""import matplotlib.pyplot as plt

for p in tqdm(paths[:5], desc="Processing images"):  # 最初の5枚だけ表示してみる
    img_id = os.path.splitext(os.path.basename(p))[0]
    label = labels_dict.get(img_id, None)
    if label is None:
        continue

    img = Image.open(p).convert("RGB")
    t = val_transform(img).unsqueeze(0).to(device)
    label_tensor = torch.tensor([label]).to(device)

    # --- 元画像で分類 ---
    outputs_clean = model(t)
    pred_clean = (torch.sigmoid(outputs_clean) > 0.5).long().cpu().item()

    # --- 敵対的画像で分類 ---
    adv_img, adv_pred = fgsm_attack_improved(
        model, t, label_tensor, epsilon_pixel=0.3, device=device,
        mean_tensor=mean, std_tensor=std, return_preds=True
    )
    pred_adv = adv_pred.item()

    # --- 浄化 ---
    x_01 = unnormalize(adv_img)
    x_diff_in = prepare_for_diffusion(x_01)
    x_purified = purify_with_diffusion(
        x_diff_in, diffusion, diff_model, device,
        use_ddim=True, real_step=30, blend_alpha=0.6, save_debug=False
    )
    x_rec_01 = recover_from_diffusion(x_purified)
    x_rec_norm = normalize(x_rec_01)
    outputs_purified = model(x_rec_norm)
    pred_purified = (torch.sigmoid(outputs_purified) > 0.5).long().cpu().item()

    # --- 画像を比較表示 ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title(f"Clean\nPred: {pred_clean}\nLabel: {label}")
    axes[0].axis('off')

    axes[1].imshow(unnormalize(adv_img[0]).permute(1,2,0).cpu().numpy())
    axes[1].set_title(f"Adversarial\nPred: {pred_adv}\nLabel: {label}")
    axes[1].axis('off')

    axes[2].imshow(x_rec_01[0].permute(1,2,0).cpu().numpy())
    axes[2].set_title(f"Purified\nPred: {pred_purified}\nLabel: {label}")
    axes[2].axis('off')


    plt.show()"""


# In[11]:


for p in tqdm(paths, desc="Processing images"):
    img_id = os.path.splitext(os.path.basename(p))[0]
    label = labels_dict.get(img_id, None)
    if label is None:
        continue  # ラベルが見つからない場合はスキップ

    img = Image.open(p).convert("RGB")
    t = val_transform(img).unsqueeze(0).to(device)
    label_tensor = torch.tensor([label]).to(device)

    # --- 元画像で分類 ---
    outputs_clean = model(t)
    pred_clean = (torch.sigmoid(outputs_clean) > 0.5).long().cpu().item()
    if pred_clean == label:
        clean_correct += 1

    # --- 敵対的画像で分類 ---
    adv_img, adv_pred = fgsm_attack_improved(
    model, t, label_tensor, epsilon_pixel=0.01, device=device,
    mean_tensor=mean, std_tensor=std, return_preds=True
    )
    outputs_adv = model(adv_img)
    pred_adv = adv_pred.item()  # 1枚の場合
    if pred_adv == label:
        adv_correct += 1
        
    # --- guided-diffusionで浄化後分類 ---
    x_01 = unnormalize(adv_img)
    x_diff_in = prepare_for_diffusion(x_01)
    x_purified = purify_with_diffusion(
        x_diff_in, diffusion, diff_model, device,
        use_ddim=True, real_step=30, blend_alpha=0.6, save_debug=False
    )
    x_rec_01 = recover_from_diffusion(x_purified)
    x_rec_norm = normalize(x_rec_01)
    outputs_purified = model(x_rec_norm)
    pred_purified = (torch.sigmoid(outputs_purified) > 0.5).long().cpu().item()
    if pred_purified == label:
        purified_correct += 1

    total += 1

print(f"Clean Accuracy: {clean_correct/total*100:.2f}% ({clean_correct}/{total})")
print(f"Adversarial Accuracy: {adv_correct/total*100:.2f}% ({adv_correct}/{total})")
print(f"Purified Accuracy: {purified_correct/total*100:.2f}% ({purified_correct}/{total})")

