"""
DDPM-based Adversarial Defense Evaluation for PCam Dataset - FGSM Attack

評価内容:
1. クリーン画像の分類精度
2. クリーン画像をDDPMで浄化した後の分類精度
3. FGSM敵対的画像の分類精度(防御なし)
4. FGSM敵対的画像をDDPMで浄化した後の分類精度(防御あり)

実行例:
python ddpm_fgsm_eval.py --epsilon 0.031 --t_purify 50 --start_t 80
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

# プロジェクトのルートを追加
sys.path.insert(0, '/mnt/data1/gotou/projects/pcam/ddpm')
from ddpm_train_pcam import SimpleUNet, GaussianDiffusion


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DDPM Defense Evaluation - FGSM Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # DDPM浄化設定
    parser.add_argument('--t_purify', type=int, default=50,
                        help='Number of diffusion steps for purification')
    parser.add_argument('--start_t', type=int, default=80,
                        help='Starting timestep for reverse diffusion')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/checkpoints/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/fgsm/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== DDPM Purifier ==========
class DDPMPurifier(nn.Module):
    """DDPM-based purification"""
    def __init__(self, unet, diffusion, device, t_purify=50, start_t=80):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.device = device
        self.t_purify = t_purify
        self.start_t = start_t
    
    def forward(self, x):
        """
        x: (B, 3, H, W), [0, 1]
        return: purified image (B, 3, H, W), [0, 1]
        """
        # [0,1] → [-1,1] (DDPM訓練時の正規化)
        x = x * 2.0 - 1.0
        
        # Forward: ノイズを加える (start_t まで)
        batch_size = x.size(0)
        t = torch.full((batch_size,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.diffusion.q_sample(x, t, noise)
        
        # Reverse: ノイズ除去 (start_t → start_t - t_purify)
        x_purified = x_noisy
        for i in range(self.t_purify):
            curr_t = self.start_t - i
            if curr_t < 0:
                break
            t_batch = torch.full((batch_size,), curr_t, device=self.device, dtype=torch.long)
            x_purified = self.diffusion.p_sample(self.unet, x_purified, t_batch)
        
        # [-1,1] → [0,1]
        x_purified = torch.clamp((x_purified + 1.0) / 2.0, 0, 1)
        return x_purified


# ========== モデル読み込み ==========
def load_models(args, device):
    """分類器とDDPMを読み込み"""
    # 分類器
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
    
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
    image_size = ddpm_args.get('image_size', 224)
    
    unet = SimpleUNet(in_ch=3, base_ch=base_ch, time_emb_dim=256).to(device)
    unet.load_state_dict(ddpm_ckpt['model_state_dict'])
    unet.eval()
    
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device)
    
    print(f"Loaded DDPM from {args.ddpm_ckpt}")
    print(f"  Image size: {image_size}, Base channels: {base_ch}, Timesteps: {timesteps}")
    
    return classifier, unet, diffusion


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


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """FGSM攻撃"""
    x = x.clone().to(device)
    x.requires_grad = True
    
    # 正規化
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std
    
    outputs = model(x_norm)
    loss = F.cross_entropy(outputs, y.to(device))
    loss.backward()
    
    x_adv = x + epsilon * x.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)
    
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


def evaluate_with_purification(purifier, classifier, x_test, y_test, device, batch_size=16, desc="Purifying"):
    """DDPM浄化後の精度を計算"""
    purifier.eval()
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
            
            # 浄化
            x_purified = purifier(x_batch)
            x_purified_all.append(x_purified.cpu())
            
            # 分類
            x_norm = (x_purified - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    x_purified_all = torch.cat(x_purified_all, dim=0)
    return correct / total, np.array(predictions), x_purified_all


def compute_l2_norm(x1, x2):
    """L2ノルムを計算"""
    diff = (x1 - x2).view(len(x1), -1)
    l2 = torch.norm(diff, p=2, dim=1).mean().item()
    return l2


def print_confusion_matrix(y_true, y_pred, title, classes):
    """混同行列を表示"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    print(f"\n{title}")
    print("-" * 60)
    
    # ヘッダー
    header = f"{'':>15}"
    for cls in classes:
        header += f" {'Pred ' + cls:>12}"
    print(header)
    
    # 各行
    for i, cls in enumerate(classes):
        row = f"{'True ' + cls:>15}"
        for j in range(len(classes)):
            row += f" {cm[i,j]:>12}"
        print(row)
    
    # メトリクス
    if len(classes) == 2:
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return {'cm': cm, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    else:
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"Accuracy: {accuracy:.4f}")
        return {'cm': cm, 'accuracy': accuracy}


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv,
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label]
        
        # 4枚を横に並べる
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"sample_{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # シード
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # デバイス
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"fgsm_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, unet, diffusion = load_models(args, device)
    
    # DDPM Purifier
    purifier = DDPMPurifier(unet, diffusion, device, t_purify=args.t_purify, start_t=args.start_t)
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("FGSM Attack + DDPM Defense Evaluation")
    print(f"{'='*70}")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"DDPM Purification: t_purify={args.t_purify}, start_t={args.start_t}")
    print(f"Samples: {len(x_test)}")
    print(f"Classes: {classes}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # ========== 2. クリーン画像をDDPMで浄化 ==========
    print("\n[2/4] Evaluating clean images with DDPM purification...")
    clean_purified_acc, pred_clean_purified, x_purified_clean = evaluate_with_purification(
        purifier, classifier, x_test, y_test, device, args.batch_size, "Purifying clean images"
    )
    print(f"Clean accuracy (with DDPM): {clean_purified_acc:.4f}")
    
    # L2ノルム
    l2_clean = compute_l2_norm(x_test, x_purified_clean)
    print(f"L2 norm (clean vs purified): {l2_clean:.4f}")
    results['clean_acc_with_ddpm'] = clean_purified_acc
    results['l2_clean_vs_purified'] = l2_clean
    
    # ========== 3. FGSM攻撃 ==========
    print("\n[3/4] Running FGSM attack...")
    start_time = time.time()
    
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start_time
    
    # L2ノルム
    l2_adv = compute_l2_norm(x_test, x_adv)
    print(f"L2 norm (clean vs adversarial): {l2_adv:.4f}")
    results['l2_clean_vs_adv'] = l2_adv
    
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像をDDPMで浄化 ==========
    print("\n[4/4] Evaluating adversarial images with DDPM purification...")
    adv_purified_acc, pred_adv_purified, x_purified_adv = evaluate_with_purification(
        purifier, classifier, x_adv, y_test, device, args.batch_size, "Purifying adversarial images"
    )
    print(f"Adversarial accuracy (with DDPM): {adv_purified_acc:.4f}")
    
    # L2ノルム
    l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
    print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
    results['adv_acc_with_ddpm'] = adv_purified_acc
    results['l2_adv_vs_purified'] = l2_adv_purified
    
    # 防御効果
    defense_improvement = adv_purified_acc - adv_acc
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"DDPM: t_purify={args.t_purify}, start_t={args.start_t}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:             {results['clean_acc']:.4f}")
    print(f"  With DDPM purification:      {results['clean_acc_with_ddpm']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (FGSM):")
    print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    print(f"  With DDPM purification:      {results['adv_acc_with_ddpm']:.4f}")
    print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"L2 Norms:")
    print(f"  Clean vs Purified:           {results['l2_clean_vs_purified']:.4f}")
    print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    print(f"  Adversarial vs Purified:     {results['l2_adv_vs_purified']:.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    y_true = y_test.numpy()
    
    cm1 = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes)
    cm2 = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with DDPM)", classes)
    cm3 = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes)
    cm4 = print_confusion_matrix(y_true, pred_adv_purified, "4. Adversarial Images (with DDPM)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm1,
        'clean_purified': cm2,
        'adv_no_defense': cm3,
        'adv_purified': cm4
    }
    
    # ==================== サンプル画像保存 ====================
    print("\nSaving sample images...")
    save_sample_images(
        x_test[:10],
        x_adv[:10],
        x_purified_clean[:10],
        x_purified_adv[:10],
        y_test[:10],
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # 結果保存
    results_save = {}
    for k, v in results.items():
        if k == 'confusion_matrices':
            results_save[k] = {}
            for kk, vv in v.items():
                if 'cm' in vv:
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
