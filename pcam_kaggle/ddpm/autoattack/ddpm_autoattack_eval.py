"""
PCam Dataset - AutoAttack + DDPM Purification Defense (Non-adaptive版)
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
Non-adaptive Attack: 分類器のみを攻撃 → DDPM浄化 → 分類

DiffPureに従い、4つの攻撃（apgd-ce, apgd-t, fab-t, square）で検証
"""

"""# 基本実行（デフォルト設定: standard版 = 4つの攻撃）
python ddpm_autoattack_eval.py

# パラメータ指定
python ddpm_autoattack_eval.py \
    --lp_norm Linf \
    --adv_eps 0.03137 \
    --start_t 80 \
    --T_purify 50 \
    --num_samples 500 \
    --gpu 0

# カスタム攻撃（特定の攻撃のみ）
python ddpm_autoattack_eval.py \
    --attack_version custom \
    --attack_type apgd-ce,square \
    --num_samples 100
    """

import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import time
import random

from autoattack import AutoAttack


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='PCam AutoAttack + DDPM Defense (DiffPure Style)')
    
    # 攻撃設定
    # DiffPureに従い、デフォルトはstandard版（4種類の攻撃: apgd-ce, apgd-t, fab-t, square）
    parser.add_argument('--attack_version', type=str, default='standard',
                        choices=['standard', 'rand', 'custom'],
                        help='Attack version: standard (4 attacks), rand (2 attacks + EOT), or custom')
    parser.add_argument('--attack_type', type=str, default='apgd-ce,apgd-t,fab-t,square',
                        help='Attack type for custom version (comma-separated, e.g., apgd-ce,square)')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Lp norm for attack')
    parser.add_argument('--adv_eps', type=float, default=8/255,
                        help='Adversarial perturbation epsilon')
    parser.add_argument('--eot_iter', type=int, default=20,
                        help='EOT iterations for rand version')
    
    # DDPM浄化設定
    parser.add_argument('--start_t', type=int, default=80,
                        help='Diffusion start timestep')
    parser.add_argument('--T_purify', type=int, default=50,
                        help='Number of purification steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDPM sampling eta (0=DDIM, 1=DDPM)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation (small for memory efficiency)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0,
                        help='Data random seed')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to evaluate (0 for all)')
    
    # パス設定
    parser.add_argument('--data_dir', type=str, default='/mnt/data1/gotou/projects/data',
                        help='Data directory')
    parser.add_argument('--ddpm_ckpt', type=str, 
                        default='/mnt/data1/gotou/projects/path/ddpm_out/ddpm1_epoch10.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/data/best_model_weights.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DDPM_MEAN = [0.5, 0.5, 0.5]
DDPM_STD = [0.5, 0.5, 0.5]


# ========== データセット ==========
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


# ========== DDPMモデル定義 ==========
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
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_ch), nn.SiLU()) if time_emb_dim else None
        self.act = nn.SiLU()
    
    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
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
        # Encoder
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, stride=2, padding=1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim)
        self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, stride=2, padding=1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, stride=2, padding=1)
        # Bottleneck
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
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
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        e4 = self.enc4(self.down3(e3), t_emb)
        b = self.bot2(self.bot1(self.down4(e4), t_emb), t_emb)
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), t_emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out_conv(d1)


# ========== DDPM浄化クラス ==========
class DDPMPurifier(nn.Module):
    """DDPM浄化処理をカプセル化"""
    def __init__(self, ddpm_model, device, T_steps=1000, start_t=80, T_purify=50, eta=0.0):
        super().__init__()
        self.ddpm = ddpm_model
        self.device = device
        self.T_steps = T_steps
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
        
        # βスケジュール
        betas = torch.linspace(1e-4, 0.02, T_steps, device=device)
        alphas = 1.0 - betas
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        posterior_variance = torch.zeros_like(betas)
        alphas_cumprod = self.alphas_cumprod
        posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
        posterior_variance[0] = 1e-8
        self.register_buffer('posterior_variance', posterior_variance)
        
        # 正規化パラメータ
        self.register_buffer('imagenet_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer('ddpm_mean', torch.tensor(DDPM_MEAN).view(1, 3, 1, 1))
        self.register_buffer('ddpm_std', torch.tensor(DDPM_STD).view(1, 3, 1, 1))
    
    def pixel_to_ddpm(self, x_pixel):
        """[0,1] → [-1,1]"""
        return (x_pixel - self.ddpm_mean) / self.ddpm_std
    
    def ddpm_to_pixel(self, x_ddpm):
        """[-1,1] → [0,1]"""
        return torch.clamp(x_ddpm * self.ddpm_std + self.ddpm_mean, 0, 1)
    
    def purify(self, x_pixel, return_grad=True):
        """
        ピクセル空間[0,1]の画像を浄化
        return_grad=True: 勾配を流す（Adaptive攻撃用）
        return_grad=False: 勾配を切る（高速評価用）
        """
        b = x_pixel.size(0)
        device = x_pixel.device  # 入力のデバイスを使用（DataParallel対応）
        
        # DDPM空間に変換
        x_ddpm = self.pixel_to_ddpm(x_pixel)
        
        # Forward diffusion
        t0 = torch.full((b,), self.start_t, device=device, dtype=torch.long)
        noise = torch.randn_like(x_ddpm)
        sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t0]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t0]).view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_ddpm + sqrt_one_minus_alpha_bar * noise
        
        # Reverse diffusion
        eps_pred_final = None
        t_final = self.start_t
        
        for t_ in range(self.start_t, max(self.start_t - self.T_purify, 0), -1):
            t_batch = torch.full((b,), t_, device=device, dtype=torch.long)
            eps_pred = self.ddpm(x_t, t_batch)
            
            alpha_t = self.alphas[t_]
            alpha_bar_t = self.alphas_cumprod[t_]
            
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
            )
            
            if t_ > 0:
                z = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt(self.posterior_variance[t_])
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            x_t = torch.clamp(x_t, -1.0, 1.0)
            eps_pred_final = eps_pred
            t_final = t_
        
        # x0再構成
        alpha_bar_final = self.alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_final) * eps_pred_final) / torch.sqrt(alpha_bar_final + 1e-12)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        # ピクセル空間に戻す
        return self.ddpm_to_pixel(x0_hat)
    
    def forward(self, x_pixel):
        return self.purify(x_pixel)


# ========== AutoAttack用モデルラッパー ==========
class ClassifierWrapper(nn.Module):
    """分類器のみのラッパー（Non-adaptive用）"""
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 2クラスロジット"""
        x_norm = (x - self.mean) / self.std
        logits = self.classifier(x_norm)
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        # 2クラス分類用に変換
        return torch.stack([-logits, logits], dim=1)


class DDPMDefenseWrapper(nn.Module):
    """DDPM浄化 + 分類器のラッパー（Adaptive用）"""
    def __init__(self, purifier, classifier, mean, std):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.counter = 0
        self.print_freq = 50  # print頻度を減らす
    
    def reset_counter(self):
        self.counter = 0
    
    def forward(self, x):
        """x: [0,1]の画像 → DDPM浄化 → 2クラスロジット"""
        if self.counter % self.print_freq == 0:
            print(f'  [DDPMDefense] Forward pass #{self.counter}')
        self.counter += 1
        
        # DDPM浄化
        x_purified = self.purifier(x)
        
        # 正規化して分類
        x_norm = (x_purified - self.mean) / self.std
        logits = self.classifier(x_norm)
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        
        return torch.stack([-logits, logits], dim=1)


def reset_defense_counter(model):
    """DataParallelでラップされていても動作するreset_counter"""
    if hasattr(model, 'module'):
        model.module.reset_counter()
    else:
        model.reset_counter()


# ========== データ読み込み ==========
def load_data(args):
    """検証データを読み込み"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    labels_csv = os.path.join(args.data_dir, 'train_labels.csv')
    train_img_dir = os.path.join(args.data_dir, 'train')
    
    labels_df = pd.read_csv(labels_csv)
    _, val_df = train_test_split(labels_df, test_size=0.1, 
                                  random_state=42, stratify=labels_df['label'])
    
    val_dataset = PCamDataset(train_img_dir, val_df, transform)
    
    # サンプル数制限
    if args.num_samples > 0 and args.num_samples < len(val_dataset):
        np.random.seed(args.data_seed)
        indices = np.random.choice(len(val_dataset), args.num_samples, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    # 全データをテンソルに変換
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    x_list, y_list = [], []
    for images, labels in tqdm(loader, desc="Loading data"):
        x_list.append(images)
        y_list.append(labels)
    
    x_val = torch.cat(x_list, dim=0)
    y_val = torch.cat(y_list, dim=0)
    
    print(f"Loaded {len(x_val)} samples")
    return x_val, y_val


# ========== モデル読み込み ==========
def load_models(args, device):
    """分類器とDDPMを読み込み"""
    # 分類器
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 1)
    classifier.load_state_dict(torch.load(args.clf_ckpt, map_location=device))
    classifier = classifier.to(device).eval()
    
    # DDPM
    ddpm = SimpleUNet(in_ch=3, base_ch=64, time_emb_dim=256).to(device)
    ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        if 'ema_state_dict' in ckpt and isinstance(ckpt['ema_state_dict'], dict):
            ddpm.load_state_dict(ckpt['ema_state_dict'], strict=False)
        else:
            ddpm.load_state_dict(ckpt['model_state_dict'])
    else:
        ddpm.load_state_dict(ckpt)
    ddpm.eval()
    
    print(f"Loaded classifier from {args.clf_ckpt}")
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return classifier, ddpm


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    """モデルの精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * bs
            end = min(start + bs, len(x))
            x_batch = x[start:end].to(device)
            y_batch = y[start:end].to(device)
            
            out = model(x_batch)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
    
    return correct / len(x)


# ========== AutoAttack評価 ==========
def eval_autoattack(args, classifier_model, defense_model, x_val, y_val, device, log_dir):
    """
    DiffPureスタイルのAutoAttack評価（Non-adaptive版）
    Non-adaptive: 分類器のみを攻撃 → DDPM防御を適用 → 精度測定
    
    DiffPureに従い、4つの攻撃で検証:
    - apgd-ce: Auto-PGD with cross-entropy loss
    - apgd-t: Auto-PGD with targeted attack
    - fab-t: FAB targeted attack
    - square: Square attack (gradient-free)
    """
    # 攻撃設定
    attack_version = args.attack_version
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    
    print(f"\n{'='*70}")
    print(f"Attack Configuration")
    print(f"{'='*70}")
    print(f"Version: {attack_version}")
    print(f"Attacks: {attack_list}")
    print(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps:.4f}")
    print(f"{'='*70}")
    
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    results = {}
    
    # ==================== Non-adaptive Attack ====================
    print(f"\n{'='*70}")
    print("NON-ADAPTIVE ATTACK (Classifier only → DDPM Defense)")
    print(f"{'='*70}")
    print("Attack targets classifier only, then DDPM purification is applied.")
    
    # 初期精度（クリーン画像）
    init_acc_clf = get_accuracy(classifier_model, x_val, y_val, bs=args.batch_size, device=device)
    print(f"Initial classifier accuracy (clean): {init_acc_clf:.4f}")
    
    # DDPM防御適用時のクリーン精度
    reset_defense_counter(defense_model)
    init_acc_def = get_accuracy(defense_model, x_val, y_val, bs=args.batch_size, device=device)
    print(f"Initial defense accuracy (clean + DDPM): {init_acc_def:.4f}")
    
    # AutoAttack（分類器のみを攻撃）
    print(f"\nRunning AutoAttack on classifier...")
    if attack_version == 'custom':
        adversary_clf = AutoAttack(
            classifier_model, norm=args.lp_norm, eps=args.adv_eps,
            version='custom', attacks_to_run=attack_list,
            log_path=os.path.join(log_dir, 'log_autoattack.txt'),
            device=device
        )
        adversary_clf.apgd.n_restarts = 1
    else:
        adversary_clf = AutoAttack(
            classifier_model, norm=args.lp_norm, eps=args.adv_eps,
            version=attack_version,
            log_path=os.path.join(log_dir, 'log_autoattack.txt'),
            device=device
        )
    
    if attack_version == 'rand':
        adversary_clf.apgd.eot_iter = args.eot_iter
    
    start_time = time.time()
    x_adv = adversary_clf.run_standard_evaluation(x_val, y_val, bs=args.batch_size)
    attack_time = time.time() - start_time
    
    # 敵対的精度（防御なし）
    robust_acc_no_defense = get_accuracy(classifier_model, x_adv, y_val, bs=args.batch_size, device=device)
    print(f"\nRobust accuracy (no defense): {robust_acc_no_defense:.4f}")
    
    # DDPM防御を適用した精度
    reset_defense_counter(defense_model)
    defended_acc = get_accuracy(defense_model, x_adv, y_val, bs=args.batch_size, device=device)
    print(f"Defended accuracy (DDPM purification): {defended_acc:.4f}")
    print(f"Defense improvement: {defended_acc - robust_acc_no_defense:+.4f}")
    print(f"Attack time: {attack_time:.2f}s")
    
    results['non_adaptive'] = {
        'clean_acc_classifier': init_acc_clf,
        'clean_acc_defense': init_acc_def,
        'robust_acc_no_defense': robust_acc_no_defense,
        'defended_acc': defended_acc,
        'improvement': defended_acc - robust_acc_no_defense,
        'attack_time': attack_time,
        'attacks': attack_list
    }
    
    # 敵対的サンプル保存
    torch.save({'x_adv': x_adv.cpu(), 'y': y_val.cpu()},
               os.path.join(log_dir, f'x_adv_sd{args.seed}.pt'))
    
    return results, x_adv


# ========== 混同行列出力 ==========
def print_confusion_matrix(y_true, y_pred, title):
    """混同行列をテキスト出力"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        
        print(f"\n{title}:")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'precision': precision, 'recall': recall, 'f1': f1}
    return {}


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, defense_model, y_true, 
                       save_dir, device, max_samples=10):
    """サンプル画像を保存（クリーン、敵対的、浄化後）"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    purifier = defense_model.purifier
    
    for i in range(n):
        # 浄化画像を取得
        with torch.no_grad():
            x_purified = purifier(x_adv[i:i+1].to(device))
        
        label = int(y_true[i])
        
        # clean, adv, purified の3枚を並べて保存
        triplet = torch.cat([
            x_clean[i:i+1],
            x_adv[i:i+1],
            x_purified.cpu()
        ], dim=0)
        grid = make_grid(triplet, nrow=3, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_label{label}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Adversarial | Purified]")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing GPU: {args.gpu}")
    print(f"Device: {device}")
    
    # 出力ディレクトリ
    attack_name = args.attack_version if args.attack_version != 'custom' else args.attack_type
    log_dir = os.path.join(
        args.output_dir,
        f'{attack_name}_{args.lp_norm}_eps{int(args.adv_eps*255)}',
        f'start{args.start_t}_purify{args.T_purify}',
        f'seed{args.seed}_data{args.data_seed}'
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, ddpm = load_models(args, device)
    
    # 浄化器
    purifier = DDPMPurifier(
        ddpm, device,
        start_t=args.start_t,
        T_purify=args.T_purify,
        eta=args.eta
    ).to(device)
    
    # ラッパー作成
    classifier_model = ClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = DDPMDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_val, y_val = load_data(args)
    
    # AutoAttack評価
    results, x_adv = eval_autoattack(
        args, classifier_model, defense_model, x_val, y_val, device, log_dir
    )
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS (Non-adaptive Attack)")
    print(f"{'='*70}")
    print(f"Attack: {args.attack_version}, Norm: {args.lp_norm}, Eps: {args.adv_eps:.4f}")
    print(f"Attacks used: {results['non_adaptive']['attacks']}")
    print(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:          {results['non_adaptive']['clean_acc_classifier']:.4f}")
    print(f"  With DDPM purification:   {results['non_adaptive']['clean_acc_defense']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy:")
    print(f"  Without defense:          {results['non_adaptive']['robust_acc_no_defense']:.4f}")
    print(f"  With DDPM purification:   {results['non_adaptive']['defended_acc']:.4f}")
    print(f"  Defense improvement:      {results['non_adaptive']['improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['non_adaptive']['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # 混同行列
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # バッチ処理で予測を取得（OOM回避）
    def get_predictions_batched(model, x, batch_size=32):
        preds = []
        n_batches = (len(x) + batch_size - 1) // batch_size
        with torch.no_grad():
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, len(x))
                x_batch = x[start:end].to(device)
                pred = model(x_batch).argmax(dim=1).cpu()
                preds.append(pred)
                # メモリ解放
                del x_batch
                torch.cuda.empty_cache()
        return torch.cat(preds).numpy()
    
    # Clean (classifier only)
    clean_pred = get_predictions_batched(classifier_model, x_val, batch_size=args.batch_size)
    # Adversarial (no defense)
    adv_pred_no_def = get_predictions_batched(classifier_model, x_adv, batch_size=args.batch_size)
    # Adversarial (with DDPM defense)
    reset_defense_counter(defense_model)
    adv_pred_defended = get_predictions_batched(defense_model, x_adv, batch_size=args.batch_size)
    
    y_true = y_val.cpu().numpy()
    print_confusion_matrix(y_true, clean_pred, "Clean Images (Classifier)")
    print_confusion_matrix(y_true, adv_pred_no_def, "Adversarial Images (No Defense)")
    print_confusion_matrix(y_true, adv_pred_defended, "Adversarial Images (DDPM Defense)")
    
    # サンプル画像保存
    save_sample_images(
        x_val[:10].cpu(), x_adv[:10].cpu(),
        defense_model, y_val[:10].cpu().numpy(),
        os.path.join(log_dir, 'samples'), device
    )
    
    # 結果をファイルに保存
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PCam - AutoAttack + DDPM Defense (Non-adaptive)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: {args.attack_version}\n")
        f.write(f"Attacks used: {results['non_adaptive']['attacks']}\n")
        f.write(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps:.4f}\n")
        f.write(f"DDPM: start_t={args.start_t}, T_purify={args.T_purify}\n")
        f.write(f"Samples: {len(x_val)}\n\n")
        f.write("-"*70 + "\n")
        f.write("Clean Accuracy:\n")
        f.write(f"  Classifier only:          {results['non_adaptive']['clean_acc_classifier']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['non_adaptive']['clean_acc_defense']:.4f}\n\n")
        f.write("Adversarial Accuracy:\n")
        f.write(f"  Without defense:          {results['non_adaptive']['robust_acc_no_defense']:.4f}\n")
        f.write(f"  With DDPM purification:   {results['non_adaptive']['defended_acc']:.4f}\n")
        f.write(f"  Defense improvement:      {results['non_adaptive']['improvement']:+.4f}\n\n")
        f.write("-"*70 + "\n")
        f.write(f"Attack time: {results['non_adaptive']['attack_time']:.2f}s\n")
        f.write("="*70 + "\n")
    
    print(f"\n✅ Results saved to: {log_dir}")
    print(f"✅ Summary: {summary_path}")
    
    return results


if __name__ == '__main__':
    main()
