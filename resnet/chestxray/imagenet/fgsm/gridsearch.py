"""
Grid search for Guided-Diffusion purification hyperparams on ChestXray FGSM defense.

- Sweep start_t in [0, 100] step=10
- Sweep T_purify in [50, 150] step=10
- Use only the first 100 images (same as imagenet_fgsm.py subset)
- Reuse reverse diffusion trajectory per start_t to evaluate all T_purify efficiently (eta=0.0)
- Output CSV and a heatmap; print best params by purified accuracy

Usage (example):
  python gridsearch.py \
    --data-dir /mnt/data1/Public/MedImages/CellData/chest_xray/test \
    --clf-ckpt /mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth \
    --gd-ckpt  /mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt \
    --out-dir  /mnt/data1/gotou/projects/chestxray/imagenet/fgsm/gridsearch_out \
    --epsilon  0.031372549 \
    --batch-size 8 --device cuda
"""

import os
import sys
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Guided-diffusion import path
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import create_model_and_diffusion


def denormalize(x_norm, mean, std):
    return x_norm * mean.new_tensor(std) + mean

def renormalize(x_pixel, mean, std):
    return (x_pixel - mean) / mean.new_tensor(std)


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


def prepare_for_diffusion(x_norm, mean, std):
    x_pixel = denormalize(x_norm, mean, std)
    x_minus1to1 = x_pixel * 2.0 - 1.0
    return x_minus1to1


def recover_from_diffusion(x_minus1to1, mean, std):
    x_pixel = (x_minus1to1 + 1.0) / 2.0
    x_pixel = torch.clamp(x_pixel, 0.0, 1.0)
    x_norm = renormalize(x_pixel, mean, std)
    return x_norm


@torch.no_grad()
def reverse_diffuse_grid(x_adv_minus1to1, model, diffusion_obj, start_t, end_indices, device, eta=0.0):
    """
    Compute reverse diffusion once from start_t down to 0, and capture x_t snapshots
    at timesteps listed in end_indices (set of ints). Returns dict: end_t -> x_end.
    """
    b = x_adv_minus1to1.size(0)
    t = torch.full((b,), start_t, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv_minus1to1)
    x_t = diffusion_obj.q_sample(x_adv_minus1to1, t, noise=noise)

    targets = set(end_indices)
    captured = {}

    for i in range(start_t, -1, -1):
        t_i = torch.full((b,), i, device=device, dtype=torch.long)
        out = diffusion_obj.p_mean_variance(
            model, x_t, t_i,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs={}
        )
        if i > 0:
            if eta == 0.0:
                x_t = out["mean"]
            else:
                nonzero_mask = (t_i != 0).float().view(-1, 1, 1, 1)
                x_t = out["mean"] + nonzero_mask * (eta * torch.sqrt(out["variance"])) * torch.randn_like(x_t)
        else:
            x_t = out["mean"]

        if i in targets and i not in captured:
            captured[i] = x_t.clone()
        # Small early exit if we've captured all
        if len(captured) == len(targets):
            break

    # Ensure requested indices are present (if end_t < 0 truncated to 0 by caller)
    for e in targets:
        if e not in captured:
            captured[e] = x_t.clone()
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/data1/Public/MedImages/CellData/chest_xray/test')
    parser.add_argument('--clf-ckpt', type=str, default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth')
    parser.add_argument('--gd-ckpt', type=str, default='/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt')
    parser.add_argument('--out-dir', type=str, default='/mnt/data1/gotou/projects/chestxray/imagenet/fgsm/gridsearch_out')

    parser.add_argument('--epsilon', type=float, default=8/255.0)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])

    parser.add_argument('--start-min', type=int, default=0)
    parser.add_argument('--start-max', type=int, default=100)
    parser.add_argument('--start-step', type=int, default=10)
    parser.add_argument('--T-min', type=int, default=50)
    parser.add_argument('--T-max', type=int, default=150)
    parser.add_argument('--T-step', type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Transforms (256x256, ImageNet norm) like imagenet_fgsm.py
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset first 100
    full_dataset = ChestXrayDataset(args.data_dir, transform=test_transform)
    subset = Subset(full_dataset, range(min(100, len(full_dataset))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Classifier (ResNet50 2-class)
    classifier = models.resnet50(pretrained=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier = classifier.to(device)
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    classifier.load_state_dict(ckpt['model_state_dict'])
    classifier.eval()

    # Guided-diffusion model 256x256
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
    gd_model, diffusion = create_model_and_diffusion(
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
    gd_state = torch.load(args.gd_ckpt, map_location=device)
    gd_model.load_state_dict(gd_state)
    gd_model.to(device)
    gd_model.eval()

    # Constants
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    start_values = list(range(args.start_min, args.start_max + 1, args.start_step))
    T_values = list(range(args.T_min, args.T_max + 1, args.T_step))

    # Accumulators
    total = 0
    correct_adv_total = 0
    correct_clean_total = 0
    correct_map = defaultdict(int)  # (start_t, T) -> correct count

    for images_norm, labels in tqdm(loader, desc='Batches'):
        images_norm = images_norm.to(device)
        labels = labels.to(device).long()

        # Clean predictions
        with torch.no_grad():
            logits_clean = classifier(images_norm)
            preds_clean = torch.argmax(logits_clean, dim=1)
        correct_mask = (preds_clean == labels)
        idx = torch.where(correct_mask)[0]
        if len(idx) == 0:
            continue

        images_norm_c = images_norm[idx]
        labels_c = labels[idx]
        preds_clean_c = preds_clean[idx]

        total += len(idx)
        correct_clean_total += len(idx)

        # FGSM once
        adv_images_norm, adv_preds = fgsm_attack(
            model=classifier,
            images=images_norm_c,
            labels=labels_c,
            epsilon_pixel=args.epsilon,
            device=device,
            mean_tensor=imagenet_mean,
            std_tensor=imagenet_std
        )
        correct_adv_total += (adv_preds == labels_c).sum().item()

        # Prepare once
        x_adv_for_diff = prepare_for_diffusion(adv_images_norm, imagenet_mean, imagenet_std)

        # For each start_t, compute reverse trajectory and capture endpoints needed by T grid
        for start_t in start_values:
            end_indices = set(max(start_t - T, 0) for T in T_values)
            captured = reverse_diffuse_grid(
                x_adv_for_diff, gd_model, diffusion,
                start_t=start_t,
                end_indices=end_indices,
                device=device,
                eta=0.0
            )
            # For each T, classify purified image from captured state
            for T in T_values:
                end_t = max(start_t - T, 0)
                x_end = captured[end_t]
                purified_norm = recover_from_diffusion(x_end, imagenet_mean, imagenet_std)
                # Classifier expects 224x224 in their setup when classifying purified
                purified_norm_224 = F.interpolate(purified_norm, size=(224,224), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    logits_pur = classifier(purified_norm_224)
                    preds_pur = torch.argmax(logits_pur, dim=1)
                    correct_map[(start_t, T)] += (preds_pur == labels_c).sum().item()

        # free memory per batch
        torch.cuda.empty_cache()

    # Summarize results
    rows = []
    for start_t in start_values:
        for T in T_values:
            corr = correct_map[(start_t, T)]
            acc = corr / total if total > 0 else 0.0
            rows.append({
                'start_t': start_t,
                'T_purify': T,
                'purified_acc': acc,
                'correct_purified': corr,
                'total': total,
            })
    df = pd.DataFrame(rows)
    df.sort_values(by='purified_acc', ascending=False, inplace=True)

    out_csv = os.path.join(args.out_dir, 'grid_results.csv')
    df.to_csv(out_csv, index=False)

    # Print best
    if len(df) > 0:
        best = df.iloc[0]
        print("\nBest params:")
        print(f"  start_t={int(best['start_t'])}, T_purify={int(best['T_purify'])}, acc={best['purified_acc']:.4f} ({int(best['correct_purified'])}/{int(best['total'])})")
    print(f"\nClean acc (on included set): {correct_clean_total/total if total>0 else 0.0:.4f}")
    print(f"Adv   acc (FGSM):           {correct_adv_total/total if total>0 else 0.0:.4f}")
    print(f"Saved CSV: {out_csv}")


if __name__ == '__main__':
    main()
