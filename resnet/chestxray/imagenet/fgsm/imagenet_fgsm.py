"""
ChestXray (肺炎分類) - FGSM攻撃 + Guided-Diffusion (ImageNet事前学習) 防御検証スクリプト
ImageNet事前学習済み拡散モデルによる敵対的画像の浄化と防御性能評価
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
from PIL import Image
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix

# Guided-diffusionモジュールのインポート
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

# ========== 設定 ==========
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TEST_DIR = os.path.join(DATA_DIR, 'test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ========== データセット定義 ==========
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        from pathlib import Path
        self.transform = transform
        self.samples = []
        
        root_path = Path(root_dir)
        class_folders = sorted([d for d in root_path.iterdir() if d.is_dir()])
        
        self.classes = [d.name for d in class_folders]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for class_folder in class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} images in {root_dir}")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# データ変換（256x256にリサイズ）
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = ChestXrayDataset(TEST_DIR, transform=test_transform)

# 先頭100枚のみ使用
from torch.utils.data import Subset
test_dataset_subset = Subset(test_dataset, range(min(100, len(test_dataset))))
test_loader = DataLoader(test_dataset_subset, batch_size=8, shuffle=False, num_workers=4)

print(f"Test samples: {len(test_dataset)} (using first {len(test_dataset_subset)} images)")

# ========== 分類器の読み込み ==========
print("\n" + "="*70)
print("Loading ResNet50 classifier...")
print("="*70)

clf_ckpt = "/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth"

classifier = models.resnet50(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)
classifier = classifier.to(device)

checkpoint = torch.load(clf_ckpt, map_location=device)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

print(f"Loaded classifier from {clf_ckpt}")

# ========== 正規化パラメータ ==========
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def denormalize(x_norm, mean, std):
    return x_norm * std + mean

def renormalize(x_pixel, mean, std):
    return (x_pixel - mean) / std

# ========== FGSM攻撃関数 ==========
def fgsm_attack(model, images, labels, epsilon_pixel, device, mean_tensor, std_tensor):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    images.requires_grad = True
    
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    grad = images.grad.data
    grad_sign = grad.sign()
    
    eps_pixel_tensor = torch.tensor(epsilon_pixel, dtype=images.dtype, device=device)
    eps_norm = (eps_pixel_tensor / std_tensor).view(1, -1, 1, 1)
    
    adv_images = images + eps_norm * grad_sign
    
    adv_pixel = denormalize(adv_images, mean_tensor, std_tensor)
    adv_pixel = torch.clamp(adv_pixel, 0.0, 1.0)
    
    adv_images = renormalize(adv_pixel, mean_tensor, std_tensor).detach()
    
    with torch.no_grad():
        adv_outputs = model(adv_images)
        adv_preds = torch.argmax(adv_outputs, dim=1)
    
    return adv_images, adv_preds

# ========== Guided-Diffusionモデルのロード ==========
print("\n" + "="*70)
print("Loading Guided-Diffusion model (ImageNet pretrained)...")
print("="*70)

# 256x256 ImageNet unconditionalモデルの設定
model_config = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'image_size': 256,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': False,
    'use_scale_shift_norm': True,
}

# モデルと拡散プロセスを作成
diffusion_model, diffusion = create_model_and_diffusion(
    **model_config,
    timestep_respacing='',
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    use_checkpoint=False,
    use_new_attention_order=False,
    dropout=0.0,
    channel_mult='',
    num_heads=4,
    num_heads_upsample=-1,
)

# チェックポイントのロード
model_path = '/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt'
state_dict = torch.load(model_path, map_location=device)
diffusion_model.load_state_dict(state_dict)
diffusion_model.to(device)
diffusion_model.eval()

print(f"Loaded Guided-Diffusion model from {model_path}")
print(f"Model image size: 256x256")
print(f"Diffusion steps: {diffusion.num_timesteps}")

# ========== DDPM浄化関数 ==========
def prepare_for_diffusion(x_norm):
    """ImageNet正規化からDDPM用の[-1,1]正規化に変換"""
    x_pixel = denormalize(x_norm, imagenet_mean, imagenet_std)
    # Guided-diffusionは[-1, 1]スケールを使用
    x_minus1to1 = x_pixel * 2.0 - 1.0
    return x_minus1to1

def recover_from_diffusion(x_minus1to1):
    """DDPM用の[-1,1]正規化からImageNet正規化に変換"""
    x_pixel = (x_minus1to1 + 1.0) / 2.0
    x_pixel = torch.clamp(x_pixel, 0.0, 1.0)
    x_norm = renormalize(x_pixel, imagenet_mean, imagenet_std)
    return x_norm

@torch.no_grad()
def diffusion_purify_guided(x_adv_minus1to1, model, diffusion_obj, start_t=250, T_purify=250, eta=0.0):
    """
    Guided-diffusionを使った画像浄化
    
    Args:
        x_adv_minus1to1: 敵対的画像（[-1,1]正規化）
        model: Guided-diffusion UNetモデル
        diffusion_obj: GaussianDiffusionオブジェクト
        start_t: 拡散開始時刻
        T_purify: 逆拡散ステップ数
        eta: DDIM eta (0=deterministic)
    
    Returns:
        浄化された画像（[-1,1]正規化）
    """
    b = x_adv_minus1to1.size(0)
    
    # Forward diffusion to start_t
    t = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1to1)
    x_t = diffusion_obj.q_sample(x_adv_minus1to1, t, noise=noise)
    
    # Reverse diffusion (DDIM sampling)
    # start_tからmax(start_t - T_purify, 0)まで逆拡散
    end_t = max(start_t - T_purify, 0)
    indices = list(range(start_t, end_t, -1))
    
    for i in indices:
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        # モデル予測
        out = diffusion_obj.p_mean_variance(
            model, x_t, t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs={}
        )
        
        # DDIM step
        if i > 0:
            nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
            x_t = out["mean"] + nonzero_mask * (eta * torch.sqrt(out["variance"])) * torch.randn_like(x_t)
        else:
            x_t = out["mean"]
    
    return torch.clamp(x_t, -1.0, 1.0)

# ========== 評価ループ ==========
print("\n" + "="*70)
print("Starting evaluation...")
print("="*70)

# 実験パラメータ
epsilon_pixel = 8/255.0
start_t = 80  # 自前DDPMと同じ設定
T_purify = 50  # 逆拡散ステップ数
save_examples_dir = "/mnt/data1/gotou/projects/chestxray/imagenet/fgsm/guided_diffusion_examples3"
os.makedirs(save_examples_dir, exist_ok=True)

save_triplets_dir = os.path.join(save_examples_dir, "triplets")
os.makedirs(save_triplets_dir, exist_ok=True)

MAX_IMAGES_TO_SAVE = 3
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
    
    # 1) Clean prediction
    with torch.no_grad():
        logits_clean = classifier(images_norm)
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
    adv_images_norm, adv_preds = fgsm_attack(
        model=classifier,
        images=images_norm_correct,
        labels=labels_correct,
        epsilon_pixel=epsilon_pixel,
        device=device,
        mean_tensor=imagenet_mean,
        std_tensor=imagenet_std
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
    
    # 3) Guided-Diffusion浄化
    x_adv_for_diff = prepare_for_diffusion(adv_images_norm)
    purified_minus1to1 = diffusion_purify_guided(
        x_adv_for_diff, 
        diffusion_model, 
        diffusion,
        start_t=start_t,
        T_purify=T_purify,
        eta=0.0
    )
    
    # 4) 浄化画像の分類
    purified_norm = recover_from_diffusion(purified_minus1to1)
    
    # 分類器は224x224を期待するのでリサイズ
    purified_norm_224 = F.interpolate(purified_norm, size=(224, 224), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        logits_pur = classifier(purified_norm_224)
        preds_pur = torch.argmax(logits_pur, dim=1)
        correct_purified += (preds_pur == labels_correct).sum().item()
        all_preds_purified.extend(preds_pur.cpu().numpy())
        
        # 浄化後のノルム計算
        pur_pixel = denormalize(purified_norm, imagenet_mean, imagenet_std)
        # clean_pixelは256x256, pur_pixelも256x256なので直接比較
        diff_pur = (pur_pixel - clean_pixel).view(len(correct_indices), -1)
        l2_pur = torch.norm(diff_pur, p=2, dim=1).cpu().numpy()
        linf_pur = torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy()
        l2_norms_purified.extend(l2_pur)
        linf_norms_purified.extend(linf_pur)
    
    # 5) Triplet画像保存（最初の3枚のみ）
    if saved_image_count < MAX_IMAGES_TO_SAVE:
        clean_pixel_save = clean_pixel.detach().clamp(0,1)
        adv_pixel_save = adv_pixel.detach().clamp(0,1)
        pur_pixel_save = pur_pixel.detach().clamp(0,1)
        
        for i in range(len(correct_indices)):
            if saved_image_count >= MAX_IMAGES_TO_SAVE:
                break
            
            # triplet tile のみ保存（256x256）
            row = torch.cat([clean_pixel_save[i], adv_pixel_save[i], pur_pixel_save[i]], dim=2)
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
print("==== Results (ChestXray - FGSM + Guided-Diffusion Defense) ====")
print("="*70)
print(f"Total samples evaluated: {total} (元画像で正解したもののみ)")
print(f"Attack: FGSM with epsilon={epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)")
print(f"Purification: Guided-Diffusion (ImageNet) start_t={start_t}, T_purify={T_purify}")
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

# ========== 混同行列 ==========
def print_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title}:")
    print(f"  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                NORMAL  PNEUMONIA")
    print(f"  Actual NORMAL    {tn:5d}  {fp:5d}")
    print(f"         PNEUMONIA {fn:5d}  {tp:5d}")
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

# サマリー統計
summary_path = os.path.join(save_examples_dir, 'summary_statistics.txt')
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("ChestXray - FGSM Attack + Guided-Diffusion (ImageNet) Defense\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: ChestXray (NORMAL vs PNEUMONIA)\n")
    f.write(f"Attack: FGSM, epsilon={epsilon_pixel:.4f} ({epsilon_pixel*255:.1f}/255)\n")
    f.write(f"Defense: Guided-Diffusion (ImageNet pretrained), start_t={start_t}, T_purify={T_purify}\n")
    f.write(f"Model: {model_path}\n\n")
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
        f.write(f"Defense Success Rate:{defense_rate:.4f}\n")

print(f"✅ Summary saved to: {summary_path}")

print("\n" + "="*70)
print("Evaluation completed successfully!")
print("="*70)
