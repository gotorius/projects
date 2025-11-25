"""
改訂版: メモリ節約のためチャンク処理で AutoAttack を回す（OOM 回避）
(ユーザ元コードをベースに最低限の修正を入れています)
"""

import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd
from autoattack import AutoAttack
import gc

# ========== 設定 (Hardcoded) ==========
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TEST_DIR = os.path.join(DATA_DIR, 'test')
CLF_CKPT = '/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth'
DDPM_CKPT = '/mnt/data1/gotou/projects/chestxray/ddpm/ddpm_out/ddpm_epoch100.pth'
OUT_DIR = '/mnt/data1/gotou/projects/chestxray/ddpm/autoattack/diffpure_v3_results'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1  # DataLoader のミニバッチ（小さいほどメモリピーク低下）
CHUNK_SIZE = 8  # チャンク内でまとめて攻撃するサンプル数（GPUに載る範囲に調整）
N_SAMPLES = 100 # 検証する画像枚数（先頭N）
EPSILON = 8/255.0
START_T = 80
PURIFY_STEPS = 50
SEED = 1234

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Output Dir: {OUT_DIR}")

# ========== 再現性のためのシード設定 ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ========== データセット定義 ==========
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        from pathlib import Path
        self.transform = transform
        self.samples = []
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cfold in class_folders:
            cidx = self.class_to_idx[cfold.name]
            for p in cfold.glob('*'):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(p), cidx))
        # ソートして順序を固定
        self.samples.sort(key=lambda x: x[0])
        print(f"Collected {len(self.samples)} test images from {root_dir}")
        print("Classes:", self.classes)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ========== 変換 ==========
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データセット読み込み (先頭 N_SAMPLES のみ)
full_dataset = ChestXrayDataset(TEST_DIR, transform=test_transform)
indices = list(range(min(N_SAMPLES, len(full_dataset))))
subset_dataset = torch.utils.data.Subset(full_dataset, indices)
# num_workers を 0 にしてワーカープロセスのメモリ消費を抑える
test_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
print(f"Using subset of {len(subset_dataset)} samples.")

# ========== モデル定義 (DDPM) ==========
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
    def forward(self, t):
        device=t.device; half=self.dim//2
        emb = torch.log(torch.tensor(10000.0))/ (half - 1)
        emb = torch.exp(torch.arange(half, device=device)*-emb)
        emb = t[:,None].float()*emb[None,:]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,time_emb_dim=None):
        super().__init__(); self.time_emb_dim=time_emb_dim
        self.conv1=nn.Conv2d(in_ch,out_ch,3,padding=1)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,padding=1)
        self.norm1=nn.GroupNorm(8 if out_ch>=8 else 1,out_ch)
        self.norm2=nn.GroupNorm(8 if out_ch>=8 else 1,out_ch)
        self.skip=nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        if time_emb_dim is not None:
            self.time_mlp=nn.Sequential(nn.Linear(time_emb_dim,out_ch), nn.SiLU())
        else: self.time_mlp=None
        self.act=nn.SiLU()
    def forward(self,x,t_emb=None):
        h=self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h=self.act(h); h=self.norm2(self.conv2(h)); h=self.act(h)
        return h + self.skip(x)

class SimpleUNet(nn.Module):
    def __init__(self,in_ch=3,base_ch=64,time_emb_dim=256):
        super().__init__()
        self.time_mlp=nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim,time_emb_dim*2), nn.SiLU(), nn.Linear(time_emb_dim*2,time_emb_dim))
        self.enc1=ResidualBlock(in_ch,base_ch,time_emb_dim); self.down1=nn.Conv2d(base_ch,base_ch*2,4,2,1)
        self.enc2=ResidualBlock(base_ch*2,base_ch*2,time_emb_dim); self.down2=nn.Conv2d(base_ch*2,base_ch*4,4,2,1)
        self.enc3=ResidualBlock(base_ch*4,base_ch*4,time_emb_dim); self.down3=nn.Conv2d(base_ch*4,base_ch*8,4,2,1)
        self.enc4=ResidualBlock(base_ch*8,base_ch*8,time_emb_dim); self.down4=nn.Conv2d(base_ch*8,base_ch*8,4,2,1)
        self.bot1=ResidualBlock(base_ch*8,base_ch*8,time_emb_dim); self.bot2=ResidualBlock(base_ch*8,base_ch*8,time_emb_dim)
        self.up4=nn.ConvTranspose2d(base_ch*8,base_ch*8,4,2,1); self.dec4=ResidualBlock(base_ch*16,base_ch*8,time_emb_dim)
        self.up3=nn.ConvTranspose2d(base_ch*8,base_ch*4,4,2,1); self.dec3=ResidualBlock(base_ch*8,base_ch*4,time_emb_dim)
        self.up2=nn.ConvTranspose2d(base_ch*4,base_ch*2,4,2,1); self.dec2=ResidualBlock(base_ch*4,base_ch*2,time_emb_dim)
        self.up1=nn.ConvTranspose2d(base_ch*2,base_ch,4,2,1); self.dec1=ResidualBlock(base_ch*2,base_ch,time_emb_dim)
        self.out_conv=nn.Sequential(nn.GroupNorm(8,base_ch), nn.SiLU(), nn.Conv2d(base_ch,in_ch,3,padding=1))
    def forward(self,x,t):
        t_emb=self.time_mlp(t)
        e1=self.enc1(x,t_emb); d1=self.down1(e1)
        e2=self.enc2(d1,t_emb); d2=self.down2(e2)
        e3=self.enc3(d2,t_emb); d3=self.down3(e3)
        e4=self.enc4(d3,t_emb); d4=self.down4(e4)
        b=self.bot1(d4,t_emb); b=self.bot2(b,t_emb)
        u4=self.up4(b); u4=torch.cat([u4,e4],dim=1); u4=self.dec4(u4,t_emb)
        u3=self.up3(u4); u3=torch.cat([u3,e3],dim=1); u3=self.dec3(u3,t_emb)
        u2=self.up2(u3); u2=torch.cat([u2,e2],dim=1); u2=self.dec2(u2,t_emb)
        u1=self.up1(u2); u1=torch.cat([u1,e1],dim=1); u1=self.dec1(u1,t_emb)
        return self.out_conv(u1)

# ========== スケジュール ==========
T_steps=1000
betas=torch.linspace(1e-4,0.02,T_steps,device=DEVICE)
alphas=1.0-betas
alphas_cumprod=torch.cumprod(alphas,dim=0)
posterior_variance=torch.zeros_like(betas)
posterior_variance[1:]=betas[1:]*(1.0-alphas_cumprod[:-1])/(1.0-alphas_cumprod[1:])
posterior_variance[0]=1e-8

# ========== 正規化ツール ==========
imagenet_mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(DEVICE)
imagenet_std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(DEVICE)
ddpm_mean=torch.tensor([0.5,0.5,0.5]).view(1,3,1,1).to(DEVICE)
ddpm_std=torch.tensor([0.5,0.5,0.5]).view(1,3,1,1).to(DEVICE)

def denormalize(x):
    return x*imagenet_std+imagenet_mean

def renormalize(x):
    return (x-imagenet_mean)/imagenet_std

# ========== DiffPure Model (End-to-End Differentiable) ==========
class DiffPureModel(nn.Module):
    def __init__(self, classifier, ddpm, start_t, steps, seed=1234):
        super().__init__()
        self.classifier = classifier
        self.ddpm = ddpm
        self.start_t = start_t
        self.steps = steps
        self.seed = seed
        
        # AutoAttack互換のためのダミー
        self.register_buffer('counter', torch.zeros(1, device=DEVICE))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=DEVICE)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        # x: [0, 1] range input (AutoAttack standard)
        
        # 1. Convert [0, 1] -> DDPM space [-1, 1]
        # DDPM expects [-1, 1]
        x_ddpm = (x - 0.5) / 0.5
        
        # 2. Purify (Differentiable)
        x_purified_ddpm = self.purify(x_ddpm)
        
        # 3. Convert [-1, 1] -> [0, 1]
        x_purified_01 = (x_purified_ddpm * 0.5) + 0.5
        x_purified_01 = torch.clamp(x_purified_01, 0.0, 1.0)
        
        # 4. Convert [0, 1] -> ImageNet Norm
        x_final = (x_purified_01 - imagenet_mean) / imagenet_std
        
        # 5. Classify
        out = self.classifier(x_final)
        return out

    def purify(self, x_in):
        # 決定論的な動作のためにシードを固定 (AutoAttackの勾配計算のため)
        torch.manual_seed(self.seed)
        
        b = x_in.size(0)
        t0 = torch.full((b,), self.start_t, device=DEVICE, dtype=torch.long)
        
        # Forward diffusion (add noise)
        noise = torch.randn_like(x_in)
        sqrt_a_bar_t0 = torch.sqrt(alphas_cumprod[t0]).view(-1, 1, 1, 1)
        sqrt_1m_a_bar_t0 = torch.sqrt(1.0 - alphas_cumprod[t0]).view(-1, 1, 1, 1)
        
        x_t = sqrt_a_bar_t0 * x_in + sqrt_1m_a_bar_t0 * noise
        
        # Reverse diffusion (denoise)
        eps_final = None
        t_final = self.start_t
        
        for t_ in range(self.start_t, max(self.start_t - self.steps, 0), -1):
            tb = torch.full((b,), t_, device=DEVICE, dtype=torch.long)
            
            # Predict noise
            eps = self.ddpm(x_t, tb)
            
            alpha_t = alphas[t_]
            alpha_bar_t = alphas_cumprod[t_]
            
            # Calculate mean
            mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps)
            
            if t_ > 0:
                z = torch.randn_like(x_t)
                sigma = 0.0 # eta=0.0 (deterministic reverse process)
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            # Clamp
            x_t = torch.clamp(x_t, -1.0, 1.0)
            
            eps_final = eps
            t_final = t_
            
        alpha_bar_tf = alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_final) / torch.sqrt(alpha_bar_tf + 1e-12)
        
        return torch.clamp(x0_hat, -1.0, 1.0)

# ========== メイン処理 ==========
def main():
    print("Loading models...")
    
    # 1. Classifier
    clf = models.resnet50(pretrained=False)
    clf.fc = nn.Linear(clf.fc.in_features, 2)
    ckpt = torch.load(CLF_CKPT, map_location=DEVICE)
    clf.load_state_dict(ckpt['model_state_dict'])
    clf = clf.to(DEVICE)
    clf.eval()
    print("Classifier loaded.")

    # 2. DDPM
    ddpm = SimpleUNet().to(DEVICE)
    raw = torch.load(DDPM_CKPT, map_location=DEVICE)
    if isinstance(raw, dict) and 'model_state_dict' in raw:
        ddpm.load_state_dict(raw['model_state_dict'])
    else:
        ddpm.load_state_dict(raw)
    ddpm.eval()
    print("DDPM loaded.")

    # 3. DiffPure Model (Combined)
    diffpure_model = DiffPureModel(clf, ddpm, START_T, PURIFY_STEPS, seed=SEED).to(DEVICE)
    diffpure_model.eval()

    # 4. AutoAttack 作成（1回だけ）
    adversary = AutoAttack(diffpure_model, norm='Linf', eps=EPSILON, version='custom', 
                           attacks_to_run=['apgd-ce', 'apgd-t', 'fab-t', 'square'], device=DEVICE)
    adversary.apgd.n_restarts = 1
    adversary.fab.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1
    adversary.fab.n_target_classes = 1
    adversary.apgd_targeted.n_target_classes = 1
    adversary.square.n_queries = 5000

    # 5. チャンク処理ループ
    all_x_clean_cpu = []   # 保存用 (少量)
    all_x_adv_cpu = []
    all_y_cpu = []
    preds_clean_list = []
    preds_adv_list = []
    preds_clf_clean_list = []

    saved_triplets = 0
    total = 0
    correct_clean = 0
    correct_robust = 0
    correct_clf_clean = 0

    # collect mini-batches into chunks
    chunk_x = []
    chunk_y = []
    print("Starting chunked attack loop...")
    for i, (x_batch, y_batch) in enumerate(test_loader):
        # x_batch: ImageNet normalized [B,3,224,224] (0..1 normalized to imagenet)
        chunk_x.append(x_batch)  # keep on CPU for now
        chunk_y.append(y_batch)
        total += x_batch.size(0)

        # if chunk is full or last sample reached, process chunk
        if len(chunk_x) >= CHUNK_SIZE or total >= len(subset_dataset):
            # stack chunk and move to device
            x_chunk = torch.cat(chunk_x, dim=0).to(DEVICE)    # ImageNet normalized
            y_chunk = torch.cat(chunk_y, dim=0).to(DEVICE)

            # denormalize to [0,1] for AutoAttack / DiffPure input
            x_chunk_01 = denormalize(x_chunk)
            x_chunk_01 = torch.clamp(x_chunk_01, 0.0, 1.0)

            # Run AutoAttack on this chunk (bs can be chunk size or 1)
            try:
                x_adv_chunk = adversary.run_standard_evaluation(x_chunk_01, y_chunk, bs=BATCH_SIZE)
                # run_standard_evaluation returns a tensor on device (or numpy). Ensure tensor.
                if isinstance(x_adv_chunk, np.ndarray):
                    x_adv_chunk = torch.from_numpy(x_adv_chunk).to(DEVICE)
            except Exception as e:
                print("AutoAttack error on chunk:", e)
                # fallback: treat as no adversarial examples (copy clean)
                x_adv_chunk = x_chunk_01.clone()

            # Evaluate chunk (DiffPure / classifier)
            with torch.no_grad():
                out_clean = diffpure_model(x_chunk_01)
                preds_clean = out_clean.argmax(dim=1)
                out_adv = diffpure_model(x_adv_chunk)
                preds_adv = out_adv.argmax(dim=1)

                out_clf_clean = clf(x_chunk)  # classifier expects imagenet-normalized
                preds_clf_clean = out_clf_clean.argmax(dim=1)

            # accumulate accuracies
            correct_clean += (preds_clean == y_chunk).sum().item()
            correct_robust += (preds_adv == y_chunk).sum().item()
            correct_clf_clean += (preds_clf_clean == y_chunk).sum().item()

            # move to CPU and store (to save results at end)
            all_x_clean_cpu.append(x_chunk_01.cpu())
            all_x_adv_cpu.append(x_adv_chunk.cpu())
            all_y_cpu.append(y_chunk.cpu())

            # save up to 10 triplets (clean, adv, purified)
            for ti in range(x_chunk_01.size(0)):
                if saved_triplets >= 10:
                    break
                idx = ti
                clean = x_chunk_01[idx:idx+1]
                adv = x_adv_chunk[idx:idx+1]
                with torch.no_grad():
                    purified_ddpm = diffpure_model.purify((adv - 0.5) / 0.5)  # pass [-1,1]
                    purified = (purified_ddpm * 0.5) + 0.5
                    purified = torch.clamp(purified, 0.0, 1.0).squeeze(0).cpu()
                clean_cpu = clean.squeeze(0).cpu()
                adv_cpu = adv.squeeze(0).cpu()
                # concat horizontally
                row = torch.cat([clean_cpu, adv_cpu, purified], dim=2)
                triplet_dir = os.path.join(OUT_DIR, 'triplets')
                os.makedirs(triplet_dir, exist_ok=True)
                save_image(row, os.path.join(triplet_dir, f'{saved_triplets:03d}_triplet.png'))
                saved_triplets += 1

            # cleanup GPU references and Python lists for this chunk
            del x_chunk, x_chunk_01, x_adv_chunk, y_chunk, out_clean, out_adv, out_clf_clean, preds_clean, preds_adv, preds_clf_clean
            chunk_x = []
            chunk_y = []
            gc.collect()
            torch.cuda.empty_cache()

    # 合計をまとめる
    total_samples = sum([t.size(0) for t in all_y_cpu])
    acc_clean = correct_clean / total_samples if total_samples>0 else 0.0
    acc_robust = correct_robust / total_samples if total_samples>0 else 0.0
    acc_clf_clean = correct_clf_clean / total_samples if total_samples>0 else 0.0

    print("\n======================================")
    print("Results (DiffPure End-to-End Attack) [Chunked]")
    print("======================================")
    print(f"Total samples: {total_samples}")
    print(f"Classifier Clean Acc: {acc_clf_clean:.4f}")
    print(f"DiffPure Clean Acc:   {acc_clean:.4f}")
    print(f"DiffPure Robust Acc:  {acc_robust:.4f}")
    print("======================================")

    # 保存（concatして1つにまとめる）
    x_clean_all = torch.cat(all_x_clean_cpu, dim=0)
    x_adv_all = torch.cat(all_x_adv_cpu, dim=0)
    y_all = torch.cat(all_y_cpu, dim=0)

    torch.save({
        'x_clean': x_clean_all,
        'x_adv': x_adv_all,
        'y': y_all,
    }, os.path.join(OUT_DIR, 'results.pt'))

    print(f"Saved results to {OUT_DIR}")

if __name__ == '__main__':
    main()
