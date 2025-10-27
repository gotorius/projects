#!/usr/bin/env python
# coding: utf-8

# # 敵対的攻撃をPGDで行ってみる。

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
import torch.nn.functional as F

# ===== 設定 (学習時の正規化と一致させる) =====
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
mean = torch.tensor(MEAN).view(1,3,1,1).to(device)
std  = torch.tensor(STD).view(1,3,1,1).to(device)

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


# --- 評価ループ（PGD版） ---
def evaluate_clean_and_pgd(model, val_loader, device, epsilon_pixel=8/255, alpha_pixel=2/255, steps=10,
                           mean=mean, std=std, attack_only_correct=False, random_start=True):
    """
    attack_only_correct: if True, only attack samples that the model originally classifies correctly
    returns: (orig_acc, adv_acc) percentages (on evaluated subset)
    """
    adv_correct, adv_total = 0, 0
    orig_correct, orig_total = 0, 0

    model.eval()

    # tqdmで進捗バーを表示
    pbar = tqdm(val_loader, desc=f"PGD attack ({steps} steps)", ncols=100)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # original predictions
        outputs = model(images)
        if outputs.ndim > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        preds = (torch.sigmoid(outputs) > 0.5).long()

        if attack_only_correct:
            correct_mask = (preds == labels.long())
            if correct_mask.sum() == 0:
                continue
            images_to_attack = images[correct_mask]
            labels_to_attack = labels[correct_mask]
            # accumulate orig stats on selected subset
            orig_preds = preds[correct_mask]
            orig_correct += (orig_preds == labels_to_attack.long()).sum().item()
            orig_total += labels_to_attack.size(0)
        else:
            images_to_attack = images
            labels_to_attack = labels
            orig_correct += (preds == labels.long()).sum().item()
            orig_total += labels.size(0)

        # run PGD
        adv_images, adv_preds = pgd_attack_improved(
            model,
            images_to_attack,
            labels_to_attack,
            epsilon_pixel=epsilon_pixel,
            alpha_pixel=alpha_pixel,
            steps=steps,
            device=device,
            mean_tensor=mean,
            std_tensor=std,
            random_start=random_start,
            return_preds=True
        )

        adv_correct += (adv_preds == labels_to_attack.cpu()).sum().item()
        adv_total += labels_to_attack.size(0)

    orig_acc = orig_correct / orig_total * 100 if orig_total > 0 else 0.0
    adv_acc = adv_correct / adv_total * 100 if adv_total > 0 else 0.0
    print(f"Original Accuracy (evaluated subset): {orig_acc:.2f}% ({orig_correct}/{orig_total})")
    print(f"PGD Adversarial Accuracy: {adv_acc:.2f}% ({adv_correct}/{adv_total})")
    return orig_acc, adv_acc


# In[ ]:


"""import torch
from torch.utils.data import Subset, DataLoader

# 先頭10サンプルのインデックス
n = 100
subset_idx = list(range(n))

# 元のデータセットから Subset を作る
subset_dataset = Subset(val_loader.dataset, subset_idx)

# 元の loader の設定を引き継いで DataLoader を作成
subset_loader = DataLoader(
    subset_dataset,
    batch_size=val_loader.batch_size,
    shuffle=False,                # 評価なら False が普通
    num_workers=val_loader.num_workers,
    pin_memory=getattr(val_loader, 'pin_memory', False)
)

# 先頭10枚だけで評価（全データ・攻撃のみ・両方に使える）
evaluate_clean_and_pgd(model, subset_loader, device,
                       epsilon_pixel=8/255, alpha_pixel=2/255, steps=10,
                       mean=mean, std=std, attack_only_correct=False, random_start=True)

evaluate_clean_and_pgd(model, subset_loader, device,
                       epsilon_pixel=8/255, alpha_pixel=2/255, steps=10,
                       mean=mean, std=std, attack_only_correct=True, random_start=True)"""


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
epsilon = 0.3
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
with open("/mnt/data1/gotou/kaggle/path/correct_image_paths.txt") as f:
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


# In[6]:


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



# In[7]:


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


from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
import torch
import numpy as np

# --- ラベルCSVの読み込み ---
labels_df = pd.read_csv('/mnt/data1/gotou/projects/Medical/kaggledata/train_labels.csv')
labels_dict = dict(zip(labels_df['id'], labels_df['label']))

# --- 攻撃対象画像リスト ---
with open("/mnt/data1/gotou/kaggle/path/correct_image_paths.txt") as f:
    all_paths = [p.strip() for p in f]
paths = all_paths[:num_samples]

clean_correct = adv_correct = purified_correct = total = 0

# 混同行列用に保存
y_true = []
y_pred_clean = []
y_pred_adv = []
y_pred_purified = []

# === ループ ===
for p in tqdm(paths, desc="Processing images"):
    img_id = os.path.splitext(os.path.basename(p))[0]
    label = labels_dict.get(img_id, None)
    if label is None:
        continue

    img = Image.open(p).convert("RGB")
    t = val_transform(img).unsqueeze(0).to(device)
    label_tensor = torch.tensor([label]).to(device)

    # --- 元画像 ---
    outputs_clean = model(t)
    pred_clean = (torch.sigmoid(outputs_clean) > 0.5).long().cpu().item()

    # --- PGD攻撃 ---
    adv_img, adv_pred = pgd_attack_improved(
        model=model,
        images=t,
        labels=label_tensor,
        epsilon_pixel=8/255,
        alpha_pixel=2/255,
        steps=10,
        device=device,
        mean_tensor=mean,
        std_tensor=std,
        random_start=True,
        return_preds=True
    )
    pred_adv = adv_pred.item()

    # --- guided-diffusionで浄化 ---
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

    # --- 正解数カウント ---
    if pred_clean == label:
        clean_correct += 1
    if pred_adv == label:
        adv_correct += 1
    if pred_purified == label:
        purified_correct += 1

    total += 1

    # --- 混同行列用データ保存 ---
    y_true.append(label)
    y_pred_clean.append(pred_clean)
    y_pred_adv.append(pred_adv)
    y_pred_purified.append(pred_purified)

# --- 精度出力 ---
print(f"\nClean Accuracy: {clean_correct/total*100:.2f}% ({clean_correct}/{total})")
print(f"Adversarial Accuracy (PGD): {adv_correct/total*100:.2f}% ({adv_correct}/{total})")
print(f"Purified Accuracy: {purified_correct/total*100:.2f}% ({purified_correct}/{total})")

# --- 混同行列をテキスト出力 ---
def print_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}")
    print(pd.DataFrame(
        cm,
        index=["True Normal (0)", "True Tumor (1)"],
        columns=["Pred Normal (0)", "Pred Tumor (1)"]
    ))

print_confusion_matrix(y_true, y_pred_clean, "Confusion Matrix - Clean Images")
print_confusion_matrix(y_true, y_pred_adv, "Confusion Matrix - Adversarial (PGD) Images")
print_confusion_matrix(y_true, y_pred_purified, "Confusion Matrix - Purified Images")


# In[ ]:


"""import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm
import os

# === 表示設定 ===
num_visualize = 5  # 表示するサンプル数

# --- 攻撃対象画像リスト ---
with open("correct_image_paths.txt") as f:
    all_paths = [p.strip() for p in f]
paths = all_paths[:num_visualize]

# === Figure全体を準備 ===
fig, axes = plt.subplots(num_visualize, 3, figsize=(9, 3 * num_visualize))
if num_visualize == 1:
    axes = [axes]  # 1枚だけの時でもループできるように

for row_idx, p in enumerate(tqdm(paths, desc="Visualizing samples")):
    img_id = os.path.splitext(os.path.basename(p))[0]
    label = labels_dict.get(img_id, None)
    if label is None:
        continue

    # === Clean ===
    img = Image.open(p).convert("RGB")
    t = val_transform(img).unsqueeze(0).to(device)
    outputs_clean = model(t)
    pred_clean = (torch.sigmoid(outputs_clean) > 0.5).long().cpu().item()
    clean_img_vis = unnormalize(t)

    # === Adversarial ===
    adv_img, adv_pred = pgd_attack_improved(
        model=model,
        images=t,
        labels=torch.tensor([label]).to(device),
        epsilon_pixel=8/255,
        alpha_pixel=2/255,
        steps=10,
        device=device,
        mean_tensor=mean,
        std_tensor=std,
        random_start=True,
        return_preds=True
    )
    pred_adv = adv_pred.item()
    adv_img_vis = unnormalize(adv_img)

    # === Purified ===
    x_01 = unnormalize(adv_img)
    x_diff_in = prepare_for_diffusion(x_01)
    x_purified = purify_with_diffusion(
        x_diff_in, diffusion, diff_model, device,
        use_ddim=True, real_step=30, blend_alpha=0.6, save_debug=False
    )
    x_rec_01 = recover_from_diffusion(x_purified)
    pur_img_vis = x_rec_01
    outputs_pur = model(normalize(x_rec_01))
    pred_pur = (torch.sigmoid(outputs_pur) > 0.5).long().cpu().item()

    # === 各列に表示 ===
    titles = [
        f"Clean\npred={pred_clean}",
        f"Adversarial\npred={pred_adv}",
        f"Purified\npred={pred_pur}"
    ]
    imgs = [clean_img_vis, adv_img_vis, pur_img_vis]

    for col_idx in range(3):
        ax = axes[row_idx][col_idx] if num_visualize > 1 else axes[col_idx]
        ax.imshow(TF.to_pil_image(imgs[col_idx][0].cpu()))
        ax.set_title(titles[col_idx], fontsize=10)
        ax.axis("off")

    axes[row_idx][0].set_ylabel(f"Sample {row_idx+1}\nLabel={label}", fontsize=10)

plt.tight_layout()
plt.show()"""

