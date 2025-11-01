#!/usr/bin/env python
# coding: utf-8

# # ImageNet事前訓練拡散モデルを医療画像分類で用いる

# はじめに医療データセットに対してResNet50で分類を行う。
# なお事前に得ていた事前訓練済みの分類機を用いる。データセットはkaggleのデータセットを用いる。

# In[2]:


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
ckpt_path = "/mnt/data1/gotou/projects/Medical/kaggledata/best_model_weights.pth"
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model from {ckpt_path}")


# In[4]:


import torch
import torch.nn as nn

# --- 攻撃関数 ---
def fgsm_attack(model, images, labels, epsilon, criterion, device):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True
    outputs = model(images)
    outputs = outputs.view(-1)  # ここを追加
    loss = criterion(outputs, labels.float())
    model.zero_grad()
    loss.backward()
    grad = images.grad.data

    adv_images = images + epsilon * grad.sign()
    adv_images = torch.clamp(adv_images, 0, 1)
    return adv_images.detach()

# --- 評価関数 ---
def evaluate_clean_and_fgsm(model, val_loader, criterion, device, epsilon):
    adv_correct, adv_total = 0, 0
    orig_correct, orig_total = 0, 0

    model.eval()
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        # 元画像での予測
        outputs = model(images).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).long()

        # 正解サンプルの mask
        correct_mask = (preds == labels.long())
        if correct_mask.sum() == 0:
            continue

        # 正解したデータだけ取り出し
        correct_images = images[correct_mask]
        correct_labels = labels[correct_mask]

        # 元画像の精度
        orig_preds = preds[correct_mask]
        orig_correct += (orig_preds == correct_labels).sum().item()
        orig_total += correct_labels.size(0)

        # FGSM攻撃
        adv_images = fgsm_attack(model, correct_images, correct_labels,
                                 epsilon, criterion, device)

        # 攻撃後の精度
        with torch.no_grad():
            adv_outputs = model(adv_images).squeeze()
            adv_preds = (torch.sigmoid(adv_outputs) > 0.5).long()
            adv_correct += (adv_preds == correct_labels).sum().item()
            adv_total += correct_labels.size(0)

    orig_acc = orig_correct / orig_total * 100 if orig_total > 0 else 0
    adv_acc = adv_correct / adv_total * 100 if adv_total > 0 else 0
    print(f"Original Accuracy (on selected correct samples): {orig_acc:.2f}% ({orig_correct}/{orig_total})")
    print(f"Adversarial Accuracy (on originally correct samples): {adv_acc:.2f}% ({adv_correct}/{adv_total})")

    return orig_acc, adv_acc




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
batch_size = 8 # VRAM に合わせて調整。256x256 diffusion は重いので小さめ推奨
epsilon = 0.03
num_samples = 2000 # correct_top1_images.txt の先頭何枚を使うか

# Diffusion settings
use_ddim = True
ddim_steps = 100     # 速くしたければ 25~50。品質を優先するなら 100+
real_step = 30     # reverse の深さ（実験して調整）
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


# In[10]:


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



# In[11]:


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


# In[ ]:


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
    adv_img = fgsm_attack(model, t, label_tensor, epsilon, criterion, device)
    outputs_adv = model(adv_img)
    pred_adv = (torch.sigmoid(outputs_adv) > 0.5).long().cpu().item()
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

