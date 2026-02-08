"""
DDPM-based Adversarial Defense Evaluation for PCam Dataset - PGD Attack

実行例:
python ddpm_pgd_eval.py --epsilon 0.031 --alpha 0.01 --steps 10
"""

import os
import sys
sys.path.insert(0, '/mnt/data1/gotou/projects/pcam/ddpm')
sys.path.insert(0, '/mnt/data1/gotou/projects/pcam/ddpm/fgsm')

import argparse
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm

from ddpm_train_pcam import SimpleUNet, GaussianDiffusion
from ddpm_fgsm_eval import (DDPMPurifier, load_cached_samples, evaluate,
                            evaluate_with_purification, compute_l2_norm,
                            print_confusion_matrix, save_sample_images,
                            IMAGENET_MEAN, IMAGENET_STD)


def parse_args():
    parser = argparse.ArgumentParser(description='DDPM Defense - PGD Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255, help='PGD perturbation bound')
    parser.add_argument('--alpha', type=float, default=2/255, help='PGD step size')
    parser.add_argument('--steps', type=int, default=10, help='PGD steps')
    
    # DDPM浄化設定
    parser.add_argument('--t_purify', type=int, default=50, help='Purification steps')
    parser.add_argument('--start_t', type=int, default=80, help='Starting timestep')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_500.pt')
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/checkpoints/best_model.pth')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/pgd/results')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def pgd_attack(model, x, y, epsilon, alpha, steps, device):
    """PGD攻撃"""
    x = x.clone().to(device)
    x_adv = x.clone()
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        x_norm = (x_adv - mean) / std
        outputs = model(x_norm)
        loss = F.cross_entropy(outputs, y.to(device))
        
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        x_adv = x_adv.detach()
    
    return x_adv


def load_models(args, device):
    """モデル読み込み"""
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
    # 分類器
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    # DDPM
    ddpm_ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    ddpm_args = ddpm_ckpt.get('args', {})
    base_ch = ddpm_args.get('base_channels', 64)
    timesteps = ddpm_args.get('timesteps', 1000)
    
    unet = SimpleUNet(in_ch=3, base_ch=base_ch, time_emb_dim=256).to(device)
    unet.load_state_dict(ddpm_ckpt['model_state_dict'])
    unet.eval()
    
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    
    return classifier, unet, diffusion


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"pgd_eps{args.epsilon:.4f}_steps{args.steps}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    classifier, unet, diffusion = load_models(args, device)
    purifier = DDPMPurifier(unet, diffusion, device, t_purify=args.t_purify, start_t=args.start_t)
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    print(f"\n{'='*70}")
    print("PGD Attack + DDPM Defense Evaluation")
    print(f"{'='*70}")
    print(f"Attack: PGD, Epsilon: {args.epsilon:.4f}, Alpha: {args.alpha:.4f}, Steps: {args.steps}")
    print(f"DDPM: t_purify={args.t_purify}, start_t={args.start_t}")
    print(f"Samples: {len(x_test)}, Classes: {classes}")
    print(f"{'='*70}")
    
    results = {}
    
    print("\n[1/4] Clean images...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    print("\n[2/4] Clean images with DDPM...")
    clean_purified_acc, pred_clean_purified, x_purified_clean = evaluate_with_purification(
        purifier, classifier, x_test, y_test, device, args.batch_size, "Purifying clean"
    )
    l2_clean = compute_l2_norm(x_test, x_purified_clean)
    print(f"Clean accuracy (DDPM): {clean_purified_acc:.4f}, L2: {l2_clean:.4f}")
    results['clean_acc_with_ddpm'] = clean_purified_acc
    results['l2_clean_vs_purified'] = l2_clean
    
    print("\n[3/4] PGD attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="PGD Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = pgd_attack(classifier, x_batch, y_batch, args.epsilon, args.alpha, args.steps, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start_time
    
    l2_adv = compute_l2_norm(x_test, x_adv)
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    print(f"Adversarial accuracy: {adv_acc:.4f}, L2: {l2_adv:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_adv
    results['attack_time'] = attack_time
    
    print("\n[4/4] Adversarial images with DDPM...")
    adv_purified_acc, pred_adv_purified, x_purified_adv = evaluate_with_purification(
        purifier, classifier, x_adv, y_test, device, args.batch_size, "Purifying adversarial"
    )
    l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
    print(f"Adversarial accuracy (DDPM): {adv_purified_acc:.4f}, L2: {l2_adv_purified:.4f}")
    results['adv_acc_with_ddpm'] = adv_purified_acc
    results['l2_adv_vs_purified'] = l2_adv_purified
    results['defense_improvement'] = adv_purified_acc - adv_acc
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: PGD (ε={args.epsilon:.4f}, α={args.alpha:.4f}, steps={args.steps})")
    print(f"-"*70)
    print(f"Clean: {results['clean_acc']:.4f} → DDPM: {results['clean_acc_with_ddpm']:.4f}")
    print(f"Adversarial: {results['adv_acc_no_defense']:.4f} → DDPM: {results['adv_acc_with_ddpm']:.4f}")
    print(f"Defense improvement: {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"L2 norms: Clean→Purified: {results['l2_clean_vs_purified']:.4f}, "
          f"Clean→Adv: {results['l2_clean_vs_adv']:.4f}, Adv→Purified: {results['l2_adv_vs_purified']:.4f}")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    y_true = y_test.numpy()
    cm1 = print_confusion_matrix(y_true, pred_clean, "1. Clean", classes)
    cm2 = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean (DDPM)", classes)
    cm3 = print_confusion_matrix(y_true, pred_adv, "3. Adversarial", classes)
    cm4 = print_confusion_matrix(y_true, pred_adv_purified, "4. Adversarial (DDPM)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm1, 'clean_purified': cm2,
        'adv_no_defense': cm3, 'adv_purified': cm4
    }
    
    print("\nSaving samples...")
    save_sample_images(x_test[:10], x_adv[:10], x_purified_clean[:10], x_purified_adv[:10],
                       y_test[:10], classes, os.path.join(log_dir, 'samples'))
    
    # 保存
    results_save = {}
    for k, v in results.items():
        if k == 'confusion_matrices':
            results_save[k] = {}
            for kk, vv in v.items():
                results_save[k][kk] = {
                    kkk: (int(vvv) if isinstance(vvv, (int, np.integer)) else
                          vvv.tolist() if isinstance(vvv, np.ndarray) else float(vvv))
                    for kkk, vvv in vv.items()
                }
        else:
            results_save[k] = float(v) if isinstance(v, (float, np.floating)) else v
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
