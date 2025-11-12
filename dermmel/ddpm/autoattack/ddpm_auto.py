"""
DermMel (メラノーマ分類) - AutoAttack攻撃 + DDPM浄化防御検証スクリプト
ChestXray版 ddpm_auto.py を皮膚病変データセットに適用
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/DermMel'
VALID_DIR = os.path.join(DATA_DIR, 'valid')  # Melanoma/ と NotMelanoma/

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class DermMelDataset(Dataset):
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
        print(f"Collected {len(self.samples)} validation images from {root_dir}")
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
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_dataset = DermMelDataset(VALID_DIR, transform=valid_transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
print(f"Validation samples: {len(valid_dataset)}")

# ========== 分類器ロード ==========
clf_ckpt = '/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth'
print("Loading classifier:", clf_ckpt)
clf = models.resnet50(pretrained=False)
clf.fc = nn.Linear(clf.fc.in_features, 2)
ckpt = torch.load(clf_ckpt, map_location=device)
clf.load_state_dict(ckpt['model_state_dict'])
clf = clf.to(device)
clf.eval()
print("Classifier loaded. Best val acc:", ckpt.get('best_val_acc', 'N/A'))

# ========== DDPMロード ==========
ddpm_ckpt = '/mnt/data1/gotou/projects/dermmel/ddpm/ddpm_out/ddpm_epoch100.pth'
print("Loading DDPM:", ddpm_ckpt)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):
        device = t.device; half = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__(); self.time_emb_dim = time_emb_dim
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_ch), nn.SiLU())
        else: self.time_mlp = None
        self.act = nn.SiLU()
    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(h); h = self.norm2(self.conv2(h)); h = self.act(h)
        return h + self.skip(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim*2), nn.SiLU(), nn.Linear(time_emb_dim*2, time_emb_dim))
        self.enc1 = ResidualBlock(in_ch, base_ch, time_emb_dim); self.down1 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)
        self.enc2 = ResidualBlock(base_ch*2, base_ch*2, time_emb_dim); self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)
        self.enc3 = ResidualBlock(base_ch*4, base_ch*4, time_emb_dim); self.down3 = nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1)
        self.enc4 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim); self.down4 = nn.Conv2d(base_ch*8, base_ch*8, 4, 2, 1)
        self.bot1 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim); self.bot2 = ResidualBlock(base_ch*8, base_ch*8, time_emb_dim)
        self.up4 = nn.ConvTranspose2d(base_ch*8, base_ch*8, 4, 2, 1); self.dec4 = ResidualBlock(base_ch*16, base_ch*8, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, 4, 2, 1); self.dec3 = ResidualBlock(base_ch*8, base_ch*4, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1); self.dec2 = ResidualBlock(base_ch*4, base_ch*2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1); self.dec1 = ResidualBlock(base_ch*2, base_ch, time_emb_dim)
        self.out_conv = nn.Sequential(nn.GroupNorm(8, base_ch), nn.SiLU(), nn.Conv2d(base_ch, in_ch, 3, padding=1))
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb); d1 = self.down1(e1)
        e2 = self.enc2(d1, t_emb); d2 = self.down2(e2)
        e3 = self.enc3(d2, t_emb); d3 = self.down3(e3)
        e4 = self.enc4(d3, t_emb); d4 = self.down4(e4)
        b = self.bot1(d4, t_emb); b = self.bot2(b, t_emb)
        u4 = self.up4(b); u4 = torch.cat([u4, e4], dim=1); u4 = self.dec4(u4, t_emb)
        u3 = self.up3(u4); u3 = torch.cat([u3, e3], dim=1); u3 = self.dec3(u3, t_emb)
        u2 = self.up2(u3); u2 = torch.cat([u2, e2], dim=1); u2 = self.dec2(u2, t_emb)
        u1 = self.up1(u2); u1 = torch.cat([u1, e1], dim=1); u1 = self.dec1(u1, t_emb)
        return self.out_conv(u1)

ddpm = SimpleUNet().to(device)
raw = torch.load(ddpm_ckpt, map_location=device)
if isinstance(raw, dict) and 'model_state_dict' in raw:
    ddpm.load_state_dict(raw['model_state_dict'])
else:
    ddpm.load_state_dict(raw)
ddpm.eval()
print("DDPM loaded.")

# ========== スケジュール ==========
T_steps = 1000
betas = torch.linspace(1e-4, 0.02, T_steps, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
posterior_variance = torch.zeros_like(betas)
posterior_variance[1:] = betas[1:] * (1.0 - alphas_cumprod[:-1]) / (1.0 - alphas_cumprod[1:])
posterior_variance[0] = 1e-8

# ========== 正規化ツール ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
ddpm_mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
ddpm_std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)

def denormalize(x):
    return x * imagenet_std + imagenet_mean

def renormalize(x):
    return (x - imagenet_mean) / imagenet_std

def to_ddpm_space(x_norm):
    x_pix = denormalize(x_norm)
    return (x_pix - ddpm_mean) / ddpm_std

def from_ddpm_space(x_minus1):
    x_pix = x_minus1 * ddpm_std + ddpm_mean
    x_pix = torch.clamp(x_pix, 0.0, 1.0)
    return renormalize(x_pix)

# ========== AutoAttack ラッパー ==========
from autoattack import AutoAttack

class NormalizedModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        x_norm = (x - imagenet_mean) / imagenet_std
        out = self.base(x_norm)
        return out

def run_autoattack(base_model, x_norm, y, eps):
    x_pix = denormalize(x_norm)  # [0,1]
    wrapped = NormalizedModel(base_model).eval()
    aa = AutoAttack(wrapped, norm='Linf', eps=eps, version='custom', attacks_to_run=['apgd-ce'], device=device, verbose=False)
    adv_pix = aa.run_standard_evaluation(x_pix, y, bs=len(x_pix))
    adv_norm = renormalize(adv_pix)
    return adv_norm

# ========== DDPM浄化 ==========
@torch.no_grad()
def diffusion_purify(x_adv_minus1, model, start_t=80, steps=50, eta=0.0, clamp_each=True):
    b = x_adv_minus1.size(0)
    t0 = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1)
    sqrt_a_bar_t0 = torch.sqrt(alphas_cumprod[t0]).view(-1, 1, 1, 1)
    sqrt_1m_a_bar_t0 = torch.sqrt(1.0 - alphas_cumprod[t0]).view(-1, 1, 1, 1)
    x_t = sqrt_a_bar_t0 * x_adv_minus1 + sqrt_1m_a_bar_t0 * noise
    eps_final = None; t_final = start_t
    for t_ in range(start_t, max(start_t - steps, 0), -1):
        tb = torch.full((b,), t_, device=device, dtype=torch.long)
        eps = model(x_t, tb)
        alpha_t = alphas[t_]; alpha_bar_t = alphas_cumprod[t_]
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps)
        if t_ > 0:
            z = torch.randn_like(x_t)
            sigma = eta * torch.sqrt(posterior_variance[t_])
            x_t = mean + sigma * z
        else:
            x_t = mean
        if clamp_each:
            x_t = torch.clamp(x_t, -1.0, 1.0)
        eps_final = eps; t_final = t_
    alpha_bar_tf = alphas_cumprod[t_final]
    x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_final) / torch.sqrt(alpha_bar_tf + 1e-12)
    return torch.clamp(x0_hat, -1.0, 1.0)

# ========== 評価設定 ==========
EPSILON_PIXEL = 8 / 255.0
START_T = 80
PURIFY_STEPS = 50

out_dir = f'/mnt/data1/gotou/projects/dermmel/ddpm/autoattack/results_autoattack_t{START_T}_s{PURIFY_STEPS}'
os.makedirs(out_dir, exist_ok=True)
triplet_dir = os.path.join(out_dir, 'triplets'); os.makedirs(triplet_dir, exist_ok=True)
MAX_SAVE = 3
saved = 0

# 統計
all_labels = []
all_clean = []
all_adv = []
all_pur = []
correct_clean = 0
correct_adv = 0
correct_pur = 0
total = 0
l2_adv = []
linf_adv = []
l2_pur = []
linf_pur = []

print(f"\n======================================")
print("Starting AutoAttack + DDPM purification evaluation (DermMel)")
print("[Evaluation policy] Use only samples correctly classified by the clean model")
print("======================================")

for batch_idx, (x_norm, y) in enumerate(tqdm(valid_loader, desc='Eval (AutoAttack->DDPM)')):
    x_norm = x_norm.to(device); y = y.to(device)
    
    # Clean preds
    with torch.no_grad():
        logits_clean = clf(x_norm)
        preds_clean = torch.argmax(logits_clean, dim=1)
    
    # Filter to only correctly classified clean samples
    correct_mask = (preds_clean == y)
    num_correct = int(correct_mask.sum().item())
    if num_correct == 0:
        continue
    
    x_norm = x_norm[correct_mask]
    y = y[correct_mask]
    preds_clean = preds_clean[correct_mask]
    
    # Update totals (clean subset only)
    total += x_norm.size(0)
    correct_clean += x_norm.size(0)  # all are correct by construction
    
    # AutoAttack adversarial (on filtered subset)
    adv_norm = run_autoattack(clf, x_norm, y, EPSILON_PIXEL)
    with torch.no_grad():
        adv_logits = clf(adv_norm)
        adv_preds = torch.argmax(adv_logits, dim=1)
    correct_adv += (adv_preds == y).sum().item()
    
    # Purify via DDPM
    x_minus1 = to_ddpm_space(adv_norm)
    pur_minus1 = diffusion_purify(x_minus1, ddpm, start_t=START_T, steps=PURIFY_STEPS, eta=0.0)
    pur_norm = from_ddpm_space(pur_minus1)
    with torch.no_grad():
        pur_logits = clf(pur_norm)
        pur_preds = torch.argmax(pur_logits, dim=1)
    correct_pur += (pur_preds == y).sum().item()
    
    # Norms (pixel space)
    clean_pix = denormalize(x_norm)
    adv_pix = denormalize(adv_norm)
    pur_pix = (pur_minus1 + 1.0) / 2.0  # already [0,1]
    diff_adv = (adv_pix - clean_pix).view(x_norm.size(0), -1)
    diff_pur = (pur_pix - clean_pix).view(x_norm.size(0), -1)
    l2_adv.extend(torch.norm(diff_adv, p=2, dim=1).cpu().numpy())
    linf_adv.extend(torch.norm(diff_adv, p=float('inf'), dim=1).cpu().numpy())
    l2_pur.extend(torch.norm(diff_pur, p=2, dim=1).cpu().numpy())
    linf_pur.extend(torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy())
    
    # accumulate labels/preds (filtered subset only)
    all_labels.extend(y.cpu().numpy())
    all_clean.extend(preds_clean.cpu().numpy())
    all_adv.extend(adv_preds.cpu().numpy())
    all_pur.extend(pur_preds.cpu().numpy())
    
    # save triplets (filtered subset only)
    if saved < MAX_SAVE:
        for i in range(x_norm.size(0)):
            if saved >= MAX_SAVE: break
            row = torch.cat([clean_pix[i], adv_pix[i], pur_pix[i]], dim=2)
            save_image(row, os.path.join(triplet_dir, f'{saved:05d}_triplet.png'))
            saved += 1

# 結果集計
if total == 0:
    print("No samples were correctly classified by the clean model. Evaluation aborted.")
else:
    clean_acc = correct_clean / total
    adv_acc = correct_adv / total
    pur_acc = correct_pur / total
    l2_adv = np.array(l2_adv); linf_adv = np.array(linf_adv); l2_pur = np.array(l2_pur); linf_pur = np.array(linf_pur)
    
    print("\n======================================")
    print("Results (DermMel AutoAttack + DDPM)")
    print("======================================")
    print(f'Total images (clean-correct only): {total}')
    print(f'Clean accuracy:     {clean_acc:.4f}')
    print(f'Adversarial accuracy:{adv_acc:.4f}')
    print(f'Purified accuracy:  {pur_acc:.4f}')
    print(f'Defense improvement:{(pur_acc - adv_acc):+.4f}')
    print("-" * 40)
    print("Perturbation Norms (Adv vs Clean):")
    print(f'  L2 mean={l2_adv.mean():.4f} std={l2_adv.std():.4f}')
    print(f'  Linf mean={linf_adv.mean():.4f} std={linf_adv.std():.4f}')
    print("Purified (vs Clean):")
    print(f'  L2 mean={l2_pur.mean():.4f} std={l2_pur.std():.4f}')
    print(f'  Linf mean={linf_pur.mean():.4f} std={linf_pur.std():.4f}')
    
    # 混同行列（テキスト）
    def print_cm(y_true, y_pred, title, labels=None):
        if labels is None:
            labels = valid_dataset.classes
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        print(f"\n{title}:")
        print("  Confusion Matrix:")
        print(f"                Predicted")
        print(f"                {labels[0]:10s} {labels[1]:12s}")
        print(f"  Actual {labels[0]:10s}  {tn:5d}  {fp:5d}")
        print(f"         {labels[1]:10s}  {fn:5d}  {tp:5d}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")
        print(f"  F1-Score:    {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")
    
    print_cm(all_labels, all_clean, 'Clean Images')
    print_cm(all_labels, all_adv, 'Adversarial (AutoAttack)')
    print_cm(all_labels, all_pur, 'Purified Images')
    
    # CSV / summary保存
    summary_txt = os.path.join(out_dir, 'summary_statistics.txt')
    df = pd.DataFrame({
        'true_label': all_labels,
        'pred_clean': all_clean,
        'pred_adv': all_adv,
        'pred_purified': all_pur,
        'l2_norm_adv': l2_adv,
        'linf_norm_adv': linf_adv,
        'l2_norm_purified': l2_pur,
        'linf_norm_purified': linf_pur,
    })
    df['attack_success'] = (df['pred_adv'] != df['true_label']).astype(int)
    df['purify_success'] = (df['pred_purified'] == df['true_label']).astype(int)
    df['defense_recovery'] = ((df['attack_success'] == 1) & (df['purify_success'] == 1)).astype(int)
    csv_path = os.path.join(out_dir, 'detailed_results.csv')
    df.to_csv(csv_path, index=False)
    
    with open(summary_txt, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write('DermMel - AutoAttack + DDPM Purification Summary\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Dataset: DermMel validation set (Melanoma vs NotMelanoma)\n')
        f.write(f'Attack: AutoAttack (APGD-CE), epsilon={EPSILON_PIXEL:.4f} ({EPSILON_PIXEL * 255:.1f}/255)\n')
        f.write(f'Purification: DDPM start_t={START_T}, steps={PURIFY_STEPS}\n')
        f.write(f'Classifier ckpt: {clf_ckpt}\n')
        f.write(f'DDPM ckpt: {ddpm_ckpt}\n\n')
        f.write('-' * 70 + '\n')
        f.write(f'Total images (clean-correct only): {total}\n')
        f.write(f'Clean Acc:      {clean_acc:.4f}\n')
        f.write(f'Adversarial Acc:{adv_acc:.4f}\n')
        f.write(f'Purified Acc:   {pur_acc:.4f}\n')
        f.write(f'Defense Improvement: {pur_acc - adv_acc:+.4f}\n')
        f.write('-' * 70 + '\n')
        f.write('Perturbation Norms (Adv vs Clean):\n')
        f.write(f'  L2 mean={l2_adv.mean():.6f} std={l2_adv.std():.6f}\n')
        f.write(f'  Linf mean={linf_adv.mean():.6f} std={linf_adv.std():.6f}\n')
        f.write('Purified (vs Clean):\n')
        f.write(f'  L2 mean={l2_pur.mean():.6f} std={l2_pur.std():.6f}\n')
        f.write(f'  Linf mean={linf_pur.mean():.6f} std={linf_pur.std():.6f}\n')
        f.write('-' * 70 + '\n')
        for name, preds in [('Clean', all_clean), ('Adversarial', all_adv), ('Purified', all_pur)]:
            cm = confusion_matrix(all_labels, preds); tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f.write(f'\n{name} Images:\n')
            f.write(f'  TN:{tn:4d} FP:{fp:4d} FN:{fn:4d} TP:{tp:4d}\n')
            f.write(f'  Precision:{precision:.4f} Recall:{recall:.4f} F1:{f1:.4f} Specificity:{specificity:.4f}\n')
    
    print(f"Saved triplets -> {triplet_dir}")
    print(f"Saved stats CSV -> {csv_path}")
    print(f"Saved summary -> {summary_txt}")
    print("Evaluation complete.")
