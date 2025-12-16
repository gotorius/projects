"""
DermMel - PGD Attack + Guided-Diffusion (ImageNet Pretrained) Defense

ImageNet事前学習済み拡散モデルを用いた敵対的防御評価 (PGD攻撃)

実行例:
python imagenet_pgd_eval.py --epsilon 0.031 --start_t 80 --t_purify 50 --num_steps 20 --step_size 0.003
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm

# Guided-diffusionモジュールのインポート
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel PGD Attack + Guided-Diffusion Defense')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD perturbation epsilon')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of PGD steps')
    parser.add_argument('--step_size', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='Use random start for PGD')
    
    # 拡散モデル浄化設定
    parser.add_argument('--start_t', type=int, default=80,
                        help='Diffusion start timestep')
    parser.add_argument('--t_purify', type=int, default=50,
                        help='Number of purification steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM sampling eta (0=deterministic)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--diffusion_ckpt', type=str,
                        default='/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt',
                        help='Guided-Diffusion checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/imagenet/pgd/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== Guided-Diffusion浄化クラス ==========
class GuidedDiffusionPurifier(nn.Module):
    """Guided-Diffusion (ImageNet Pretrained) による浄化"""
    def __init__(self, diffusion_model, diffusion, device, start_t=80, t_purify=50, eta=0.0):
        super().__init__()
        self.model = diffusion_model
        self.diffusion = diffusion
        self.device = device
        self.start_t = start_t
        self.t_purify = t_purify
        self.eta = eta
    
    @torch.no_grad()
    def forward(self, x):
        """
        x: (B, 3, H, W), [0, 1]
        return: purified image (B, 3, H, W), [0, 1]
        """
        b = x.size(0)
        original_size = x.shape[-2:]
        
        # 256x256にリサイズ
        if original_size != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        # [0,1] → [-1,1]
        x_diff = x * 2.0 - 1.0
        
        # Forward diffusion to start_t
        t = torch.full((b,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_diff)
        x_t = self.diffusion.q_sample(x_diff, t, noise=noise)
        
        # Reverse diffusion
        end_t = max(self.start_t - self.t_purify, 0)
        indices = list(range(self.start_t, end_t, -1))
        
        for i in indices:
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            out = self.diffusion.p_mean_variance(
                self.model, x_t, t,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs={}
            )
            
            if i > 0:
                nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
                x_t = out["mean"] + nonzero_mask * (self.eta * torch.sqrt(out["variance"])) * torch.randn_like(x_t)
            else:
                x_t = out["mean"]
        
        x_purified = torch.clamp(x_t, -1.0, 1.0)
        x_purified = (x_purified + 1.0) / 2.0
        
        if original_size != (256, 256):
            x_purified = F.interpolate(x_purified, size=original_size, mode='bilinear', align_corners=False)
        
        return torch.clamp(x_purified, 0, 1)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """分類器を読み込み (DermMel用 - Dropoutなし)"""
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    
    return classifier


def load_guided_diffusion(args, device):
    """Guided-Diffusionモデルを読み込み"""
    print("\nLoading Guided-Diffusion model (ImageNet pretrained)...")
    
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
    
    state_dict = torch.load(args.diffusion_ckpt, map_location=device)
    diffusion_model.load_state_dict(state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()
    
    print(f"Loaded Guided-Diffusion model from {args.diffusion_ckpt}")
    print(f"  Model image size: 256x256")
    print(f"  Diffusion steps: {diffusion.num_timesteps}")
    
    return diffusion_model, diffusion


# ========== データ読み込み ==========
def load_cached_samples(path):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    print(f"Loaded {len(x_test)} samples from {path}")
    print(f"Classes: {classes}")
    return x_test, y_test, classes


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, step_size, num_steps, random_start, device):
    """PGD攻撃"""
    x = x.clone().to(device)
    y = y.to(device)
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    if random_start:
        x_adv = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0, 1)
    else:
        x_adv = x.clone()
    
    for _ in range(num_steps):
        x_adv.requires_grad = True
        x_norm = (x_adv - mean) / std
        outputs = model(x_norm)
        loss = F.cross_entropy(outputs, y)
        loss.backward()
        
        grad = x_adv.grad.sign()
        x_adv = x_adv.detach() + step_size * grad
        
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv.detach()


# ========== 評価関数 ==========
def evaluate(model, x_test, y_test, device, batch_size=16):
    """精度を計算"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            x_norm = (x_batch - mean) / std
            outputs = model(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    return correct / total, np.array(predictions)


def evaluate_with_purification(purifier, classifier, x_test, y_test, device, batch_size=8, desc="Purifying"):
    """拡散浄化後の精度を計算"""
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_purified_all = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            x_purified = purifier(x_batch)
            x_purified_all.append(x_purified.cpu())
            
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    x_purified_all = torch.cat(x_purified_all, dim=0)
    return correct / total, np.array(predictions), x_purified_all


# ========== ユーティリティ ==========
def compute_l2_norm(x1, x2):
    """L2ノルムを計算"""
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def print_confusion_matrix(y_true, y_pred, title, classes, file=None):
    """混同行列を出力"""
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    def write_and_print(text):
        print(text)
        if file:
            file.write(text + '\n')
    
    write_and_print(f"\n{title}")
    write_and_print("-" * 60)
    
    header = f"{'':>15}" + "".join([f"Pred {c:>8}" for c in classes])
    write_and_print(header)
    
    for i, true_class in enumerate(classes):
        row = f"{'True ' + true_class:>15}" + "".join([f"{cm[i, j]:>12}" for j in range(len(classes))])
        write_and_print(row)
    
    write_and_print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {'cm': cm, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, labels, classes, save_dir):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    n = min(len(x_clean), 10)
    
    for i in range(n):
        label = classes[labels[i]]
        images = [x_clean[i], x_adv[i], x_purified_clean[i], x_purified_adv[i]]
        grid = make_grid(images, nrow=4, padding=2, normalize=False)
        save_image(grid, os.path.join(save_dir, f'sample_{i}_{label}.png'))
    
    print(f"Saved {n} sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    classifier = load_classifier(args, device)
    diffusion_model, diffusion = load_guided_diffusion(args, device)
    
    purifier = GuidedDiffusionPurifier(
        diffusion_model, diffusion, device,
        start_t=args.start_t, t_purify=args.t_purify, eta=args.eta
    )
    
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("PGD Attack + Guided-Diffusion (ImageNet) Defense Evaluation (DermMel)")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"        Steps: {args.num_steps}, Step Size: {args.step_size:.4f}, Random Start: {args.random_start}")
    write_and_print(f"Diffusion: start_t={args.start_t}, t_purify={args.t_purify}, eta={args.eta}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    write_and_print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    write_and_print("\n[2/4] Evaluating clean images with Guided-Diffusion purification...")
    clean_purified_acc, pred_clean_purified, x_purified_clean = evaluate_with_purification(
        purifier, classifier, x_test, y_test, device, args.batch_size, "Purifying clean images"
    )
    l2_clean_purified = compute_l2_norm(x_test, x_purified_clean)
    write_and_print(f"Clean accuracy (with Diffusion): {clean_purified_acc:.4f}")
    write_and_print(f"L2 norm (clean vs purified): {l2_clean_purified:.4f}")
    results['clean_acc_with_diffusion'] = clean_purified_acc
    results['l2_clean_vs_purified'] = l2_clean_purified
    
    write_and_print("\n[3/4] Running PGD attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="PGD Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = pgd_attack(classifier, x_batch, y_batch, args.epsilon,
                                 args.step_size, args.num_steps, args.random_start, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start_time
    
    l2_clean_adv = compute_l2_norm(x_test, x_adv)
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    write_and_print(f"L2 norm (clean vs adversarial): {l2_clean_adv:.4f}")
    write_and_print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_clean_adv
    results['attack_time'] = attack_time
    
    write_and_print("\n[4/4] Evaluating adversarial images with Guided-Diffusion purification...")
    adv_purified_acc, pred_adv_purified, x_purified_adv = evaluate_with_purification(
        purifier, classifier, x_adv, y_test, device, args.batch_size, "Purifying adversarial images"
    )
    l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
    write_and_print(f"Adversarial accuracy (with Diffusion): {adv_purified_acc:.4f}")
    write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
    results['adv_acc_with_diffusion'] = adv_purified_acc
    results['l2_adv_vs_purified'] = l2_adv_purified
    results['defense_improvement'] = adv_purified_acc - adv_acc
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"        Steps: {args.num_steps}, Step Size: {args.step_size:.4f}")
    write_and_print(f"Defense: Guided-Diffusion (ImageNet), start_t={args.start_t}, t_purify={args.t_purify}")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With Diffusion:              {results['clean_acc_with_diffusion']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (PGD):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With Diffusion:              {results['adv_acc_with_diffusion']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    write_and_print(f"  Clean vs Purified:           {results['l2_clean_vs_purified']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    write_and_print(f"  Adversarial vs Purified:     {results['l2_adv_vs_purified']:.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Attack time: {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes, results_file)
    cm_results['clean_purified'] = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with Diffusion)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_purified'] = print_confusion_matrix(y_true, pred_adv_purified, "4. Adversarial Images (with Diffusion)", classes, results_file)
    
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(x_test[:10], x_adv[:10], x_purified_clean[:10], x_purified_adv[:10],
                       y_test[:10], classes, samples_dir)
    
    results_file.close()
    
    results_save = {
        'config': vars(args),
        'results': {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in results.items()},
        'confusion_matrices': {
            k: {
                'cm': v['cm'].tolist(),
                'accuracy': float(v['accuracy']),
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1'])
            } for k, v in cm_results.items()
        }
    }
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print(f"\nResults saved to {log_dir}")


if __name__ == '__main__':
    main()
