import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Add DiffPure to path for guided_diffusion
sys.path.append('/mnt/data1/gotou/DiffPure')
try:
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
except ImportError:
    print("Error: Could not import guided_diffusion. Make sure DiffPure is in the path.")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = '/mnt/data1/gotou/projects/data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
CLF_CKPT_PATH = "/mnt/data1/gotou/projects/data/best_model_weights.pth"
GUIDED_DIFFUSION_PATH = "/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Dataset ---
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

# Transforms (Classifier expects 224x224, ImageNet Norm)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load Data ---
if os.path.exists(LABELS_CSV):
    labels_df = pd.read_csv(LABELS_CSV)
    # Use same split as training if possible, or just a subset for validation
    _, val_df = train_test_split(labels_df, test_size=0.1, random_state=42, stratify=labels_df['label'])
    val_dataset = PCamDataset(TRAIN_IMG_DIR, val_df, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4) # Reduced batch size for VRAM
else:
    print(f"Warning: {LABELS_CSV} not found. Cannot load dataset.")
    sys.exit(1)

# --- Load Classifier ---
print("Loading classifier...")
clf = models.resnet50(pretrained=False)
clf.fc = nn.Linear(clf.fc.in_features, 1)
if os.path.exists(CLF_CKPT_PATH):
    clf.load_state_dict(torch.load(CLF_CKPT_PATH, map_location=DEVICE))
else:
    print(f"Warning: {CLF_CKPT_PATH} not found.")
    sys.exit(1)
clf = clf.to(DEVICE)
clf.eval()

# --- Load Guided Diffusion ---
print("Loading guided diffusion model...")
def load_guided_diffusion(model_path, device):
    model_config = model_and_diffusion_defaults()
    # Config from DiffPure/configs/imagenet.yml
    model_config.update({
        'attention_resolutions': '32,16,8',
        'class_cond': False,
        'diffusion_steps': 1000,
        'rescale_timesteps': True,
        'timestep_respacing': '1000', 
        'image_size': 256,
        'learn_sigma': True,
        'noise_schedule': 'linear',
        'num_channels': 256,
        'num_head_channels': 64,
        'num_res_blocks': 2,
        'resblock_updown': True,
        'use_fp16': True,
        'use_scale_shift_norm': True
    })
    
    model, diffusion = create_model_and_diffusion(**model_config)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Warning: {model_path} not found.")
        sys.exit(1)
        
    model.to(device)
    if model_config['use_fp16']:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion

gd_model, gd_diffusion = load_guided_diffusion(GUIDED_DIFFUSION_PATH, DEVICE)

# --- Helper Functions ---
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)

def unnormalize(x):
    return x * imagenet_std + imagenet_mean

def normalize(x):
    return (x - imagenet_mean) / imagenet_std

def fgsm_attack(model, images, labels, epsilon):
    images = images.clone().detach()
    images.requires_grad = True
    outputs = model(images)
    
    # Binary Cross Entropy Loss
    # labels are 0 or 1. outputs are logits.
    loss = F.binary_cross_entropy_with_logits(outputs.squeeze(1), labels.float())
    
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    
    # FGSM in pixel space
    x_pixel = unnormalize(images)
    
    # Gradient w.r.t pixel space: grad_pixel = grad_norm / std
    grad_pixel = data_grad / imagenet_std
    
    perturbed_image = x_pixel + epsilon * grad_pixel.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return normalize(perturbed_image).detach()

@torch.no_grad()
def purify_with_guided_diffusion(images_norm, model, diffusion, start_t=80, purify_steps=50):
    # 1. Resize to 256x256
    images_pixel = unnormalize(images_norm)
    images_256 = F.interpolate(images_pixel, size=(256, 256), mode='bilinear', align_corners=False)
    
    # 2. Scale to [-1, 1]
    x0 = images_256 * 2.0 - 1.0
    
    # 3. Add noise
    t = torch.tensor([start_t] * x0.shape[0], device=DEVICE)
    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise=noise)
    
    # 4. Denoise (Reverse process)
    x = x_t
    # Iterate from start_t down to start_t - purify_steps
    # e.g. start_t=80, purify_steps=50 -> range(80, 30, -1) -> 80, 79, ..., 31
    indices = list(range(start_t, max(start_t - purify_steps, 0), -1))
    
    x0_hat = x # Default if no steps
    
    for i in indices:
        t_tensor = torch.tensor([i] * x0.shape[0], device=DEVICE)
        out = diffusion.p_sample(
            model,
            x,
            t_tensor,
            clip_denoised=True
        )
        x = out["sample"]
        x0_hat = out["pred_xstart"]
        
    # 5. Scale back to [0, 1]
    # Use x0_hat (predicted x0) from the last step for better quality if stopping early
    x_purified_256 = (x0_hat + 1.0) * 0.5
    x_purified_256 = torch.clamp(x_purified_256, 0, 1)
    
    # 6. Resize back to 224x224
    x_purified_224 = F.interpolate(x_purified_256, size=(224, 224), mode='bilinear', align_corners=False)
    
    # 7. Normalize
    return normalize(x_purified_224)

# --- Main Loop ---
EPSILON = 8/255.0
START_T = 80
PURIFY_STEPS = 50

SAVE_DIR = "guided_fgsm_results"
os.makedirs(SAVE_DIR, exist_ok=True)
TRIPLET_DIR = os.path.join(SAVE_DIR, "triplets")
os.makedirs(TRIPLET_DIR, exist_ok=True)

total = 0
correct_clean = 0
correct_adv = 0
correct_purified = 0

l2_norms_adv = []
linf_norms_adv = []
l2_norms_purified = []
linf_norms_purified = []

all_labels = []
all_preds_clean = []
all_preds_adv = []
all_preds_purified = []

MAX_IMAGES_TO_SAVE = 5
saved_image_count = 0

print(f"Starting evaluation with FGSM (eps={EPSILON:.4f}) and Guided Diffusion Purification (start_t={START_T}, steps={PURIFY_STEPS})...")

# Use a subset for quick testing if needed, but here we iterate over val_loader
# To match ddpm_auto.py, we iterate over all validation images
for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE).float().view(-1)
    
    # 1. Clean
    with torch.no_grad():
        logits_clean = clf(images).squeeze(1)
        preds_clean = (torch.sigmoid(logits_clean) > 0.5).float()
    
    # Filter correct only
    correct_mask = (preds_clean == labels)
    if correct_mask.sum() == 0:
        continue
        
    images_correct = images[correct_mask]
    labels_correct = labels[correct_mask]
    
    total += len(images_correct)
    correct_clean += len(images_correct)
    all_labels.extend(labels_correct.cpu().numpy())
    all_preds_clean.extend(preds_clean[correct_mask].cpu().numpy())
    
    # 2. FGSM Attack
    adv_images = fgsm_attack(clf, images_correct, labels_correct, EPSILON)
    
    with torch.no_grad():
        logits_adv = clf(adv_images).squeeze(1)
        preds_adv = (torch.sigmoid(logits_adv) > 0.5).float()
        
    correct_adv += (preds_adv == labels_correct).sum().item()
    all_preds_adv.extend(preds_adv.cpu().numpy())
    
    # Norms
    clean_pixel = unnormalize(images_correct)
    adv_pixel = unnormalize(adv_images)
    diff = (adv_pixel - clean_pixel).view(len(images_correct), -1)
    l2_norms_adv.extend(torch.norm(diff, p=2, dim=1).cpu().numpy())
    linf_norms_adv.extend(torch.norm(diff, p=float('inf'), dim=1).cpu().numpy())
    
    # 3. Purification
    purified_images = purify_with_guided_diffusion(adv_images, gd_model, gd_diffusion, start_t=START_T, purify_steps=PURIFY_STEPS)
    
    with torch.no_grad():
        logits_purified = clf(purified_images).squeeze(1)
        preds_purified = (torch.sigmoid(logits_purified) > 0.5).float()
        
    correct_purified += (preds_purified == labels_correct).sum().item()
    all_preds_purified.extend(preds_purified.cpu().numpy())
    
    # Norms Purified
    purified_pixel = unnormalize(purified_images)
    diff_pur = (purified_pixel - clean_pixel).view(len(images_correct), -1)
    l2_norms_purified.extend(torch.norm(diff_pur, p=2, dim=1).cpu().numpy())
    linf_norms_purified.extend(torch.norm(diff_pur, p=float('inf'), dim=1).cpu().numpy())
    
    # Save Images
    if saved_image_count < MAX_IMAGES_TO_SAVE:
        for i in range(len(images_correct)):
            if saved_image_count >= MAX_IMAGES_TO_SAVE:
                break
            
            # Create grid: Clean | Adv | Purified
            # Resize to same size for grid if needed, but they are all 224x224 now (purified is resized back)
            grid = torch.cat([
                clean_pixel[i],
                adv_pixel[i],
                purified_pixel[i]
            ], dim=2) # Concatenate horizontally
            
            save_image(grid, os.path.join(TRIPLET_DIR, f"triplet_{saved_image_count}.png"))
            saved_image_count += 1

# --- Statistics ---
clean_acc = correct_clean / total if total > 0 else 0
adv_acc = correct_adv / total if total > 0 else 0
pur_acc = correct_purified / total if total > 0 else 0

l2_norms_adv = np.array(l2_norms_adv)
linf_norms_adv = np.array(linf_norms_adv)
l2_norms_purified = np.array(l2_norms_purified)
linf_norms_purified = np.array(linf_norms_purified)

print("\n" + "="*70)
print("==== Results (Clean Images Only - FGSM) ====")
print("="*70)
print(f"Total samples evaluated: {total}")
print(f"Attack: FGSM with epsilon={EPSILON:.4f}")
print(f"Purification: Guided Diffusion start_t={START_T}, steps={PURIFY_STEPS}")
print("-"*70)
print(f"Clean accuracy:     {clean_acc:.4f}")
print(f"Adv (FGSM) accuracy:{adv_acc:.4f}")
print(f"Purified accuracy:  {pur_acc:.4f}")
print(f"Defense improvement: {pur_acc - adv_acc:+.4f}")
print("-"*70)

print("\n" + "="*70)
print("==== Perturbation Norms ====")
print("="*70)
print("Adversarial Perturbations (vs Clean):")
if len(l2_norms_adv) > 0:
    print(f"  L2 norm:   mean={l2_norms_adv.mean():.4f}, std={l2_norms_adv.std():.4f}, "
          f"min={l2_norms_adv.min():.4f}, max={l2_norms_adv.max():.4f}")
    print(f"  L∞ norm:   mean={linf_norms_adv.mean():.4f}, std={linf_norms_adv.std():.4f}, "
          f"min={linf_norms_adv.min():.4f}, max={linf_norms_adv.max():.4f}")

print("\nPurified Images (vs Clean):")
if len(l2_norms_purified) > 0:
    print(f"  L2 norm:   mean={l2_norms_purified.mean():.4f}, std={l2_norms_purified.std():.4f}, "
          f"min={l2_norms_purified.min():.4f}, max={l2_norms_purified.max():.4f}")
    print(f"  L∞ norm:   mean={linf_norms_purified.mean():.4f}, std={linf_norms_purified.std():.4f}, "
          f"min={linf_norms_purified.min():.4f}, max={linf_norms_purified.max():.4f}")
print("="*70)

# Save detailed results
stats_df = pd.DataFrame({
    'true_label': all_labels,
    'pred_clean': all_preds_clean,
    'pred_adv': all_preds_adv,
    'pred_purified': all_preds_purified,
})
stats_df.to_csv(os.path.join(SAVE_DIR, 'detailed_results.csv'), index=False)

# Confusion Matrix
def print_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{title} Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Normal  Tumor")
    print(f"  Actual Normal   {tn:5d}  {fp:5d}")
    print(f"         Tumor    {fn:5d}  {tp:5d}")
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

print_cm(all_labels, all_preds_clean, "Clean")
print_cm(all_labels, all_preds_adv, "Adversarial")
print_cm(all_labels, all_preds_purified, "Purified")

# Summary Text
summary_path = os.path.join(SAVE_DIR, 'summary_statistics.txt')
with open(summary_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("FGSM + Guided Diffusion Purification - Summary Statistics\n")
    f.write("="*70 + "\n\n")
    f.write(f"Clean Accuracy:      {clean_acc:.4f}\n")
    f.write(f"Adversarial Accuracy:{adv_acc:.4f}\n")
    f.write(f"Purified Accuracy:   {pur_acc:.4f}\n")
    f.write(f"Defense Improvement: {pur_acc - adv_acc:+.4f}\n")
print(f"✅ Summary statistics saved to: {summary_path}")
