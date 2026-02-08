"""
DDPM-based Adversarial Defense Evaluation for DermMel Dataset - FGSM Attack (ViT Classifier)

ViT分類器を使用したDDPM防御評価スクリプト

実行例:
python ddpm_fgsm_eval_vit.py --epsilon 0.031 --t_purify 50 --start_t 80
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
sys.path.insert(0, '/mnt/data1/gotou/projects/resnet/dermmel/ddpm')
from ddpm_train_v2 import SimpleUNet


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DDPM Defense Evaluation - FGSM Attack (ViT) for DermMel')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # DDPM浄化設定
    parser.add_argument('--t_purify', type=int, default=50,
                        help='Number of diffusion steps for purification')
    parser.add_argument('--start_t', type=int, default=270,
                        help='Starting timestep for reverse diffusion')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='Stochasticity parameter for DDIM')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--ddpm_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/resnet/dermmel/ddpm/ddpm_out2/best_model.pth',
                        help='DDPM checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth',
                        help='ViT classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/ddpm/fgsm/results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== ViTモデル構築 ==========
def get_vit_model(model_name='vit_b_16', num_classes=2, dropout=0.1):
    """
    Vision Transformer モデルの構築（推論用）
    """
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 分類ヘッドを置き換え（訓練時と同じ構造）
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    
    return model


# ========== GaussianDiffusion クラス ==========
class GaussianDiffusion:
    """DDPM用のGaussian Diffusionクラス"""
    def __init__(self, timesteps=1000, device='cuda', beta_schedule='cosine'):
        self.timesteps = timesteps
        self.device = device
        
        # βスケジュール
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == 'cosine':
            import math
            s = 0.008
            t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
            f = (t / timesteps + s) / (1 + s)
            alphas_bar = torch.cos(f * math.pi / 2) ** 2
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
            betas = betas.clamp(min=1e-8, max=0.999).to(torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas.to(device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(device)
        
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-8)
        self.posterior_variance[0] = 1e-8


# ========== DDPM Purifier (Improved) ==========
class DDPMPurifierImproved(nn.Module):
    """DDPM-based purification with x0 reconstruction"""
    def __init__(self, unet, diffusion, device, t_purify=50, start_t=80, eta=0.0):
        super().__init__()
        self.unet = unet
        self.diffusion = diffusion
        self.device = device
        self.t_purify = t_purify
        self.start_t = start_t
        self.eta = eta
    
    def forward(self, x):
        """
        x: (B, 3, H, W), [0, 1] (unnormalized pixel values)
        return: purified image (B, 3, H, W), [0, 1]
        """
        # [0,1] → [-1,1] (DDPM訓練時の正規化: x * 2 - 1)
        x_minus1to1 = x * 2.0 - 1.0
        
        # Forward: ノイズを加える (start_t まで)
        batch_size = x.size(0)
        t0 = torch.full((batch_size,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_minus1to1)
        
        sqrt_alpha_bar_t0 = self.diffusion.sqrt_alphas_cumprod[t0].view(-1, 1, 1, 1)
        sqrt_1m_alpha_bar_t0 = self.diffusion.sqrt_one_minus_alphas_cumprod[t0].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t0 * x_minus1to1 + sqrt_1m_alpha_bar_t0 * noise
        
        # Reverse: ノイズ除去 (start_t → start_t - t_purify)
        eps_pred_final = None
        t_final = self.start_t
        
        for i in range(self.t_purify):
            curr_t = self.start_t - i
            if curr_t < 0:
                break
            
            t_batch = torch.full((batch_size,), curr_t, device=self.device, dtype=torch.long)
            eps_pred = self.unet(x_t, t_batch)
            
            alpha_t = self.diffusion.alphas[curr_t]
            alpha_bar_t = self.diffusion.alphas_cumprod[curr_t]
            
            # DDPMの平均計算
            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * eps_pred
            )
            
            if curr_t > 0:
                z = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt(self.diffusion.posterior_variance[curr_t])
                x_t = mean + sigma * z
            else:
                x_t = mean
            
            # 各ステップでクランプ
            x_t = torch.clamp(x_t, -1.0, 1.0)
            
            # 最終ステップの記録
            eps_pred_final = eps_pred
            t_final = curr_t
        
        # x0再構成 (ノイズ抑制)
        alpha_bar_tf = self.diffusion.alphas_cumprod[t_final]
        x0_hat = (x_t - torch.sqrt(1 - alpha_bar_tf) * eps_pred_final) / torch.sqrt(alpha_bar_tf + 1e-12)
        x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
        
        # [-1,1] → [0,1] (正しい逆変換: (x + 1) / 2)
        x_purified = (x0_hat + 1.0) / 2.0
        x_purified = torch.clamp(x_purified, 0, 1)
        
        return x_purified


# ========== モデル読み込み ==========
def load_models(args, device):
    """ViT分類器とDDPMを読み込み"""
    # 分類器 (ViT)
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
    classifier = get_vit_model(model_name='vit_b_16', num_classes=num_classes, dropout=0.1)
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded ViT classifier from {args.clf_ckpt}")
    
    # DDPM
    ddpm_ckpt = torch.load(args.ddpm_ckpt, map_location=device)
    ddpm_args = ddpm_ckpt.get('args', {})
    
    base_ch = ddpm_args.get('base_channels', 128)
    timesteps = ddpm_args.get('timesteps', 1000)
    image_size = ddpm_args.get('image_size', 224)
    
    unet = SimpleUNet(in_ch=3, base_ch=base_ch, time_emb_dim=256, attn_heads=4).to(device)
    
    # EMAがあれば優先的に使用
    if 'ema_state_dict' in ddpm_ckpt and ddpm_ckpt['ema_state_dict'] is not None:
        print("Using EMA weights")
        unet.load_state_dict(ddpm_ckpt['ema_state_dict'])
    elif 'model_state_dict' in ddpm_ckpt:
        unet.load_state_dict(ddpm_ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ddpm_ckpt)
    
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


# ========== ユーティリティ ==========
def compute_l2_norm(x1, x2):
    """L2ノルムを計算"""
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def print_confusion_matrix(y_true, y_pred, title, classes, file=None):
    """混同行列を出力"""
    cm = confusion_matrix(y_true, y_pred)
    
    # メトリクス計算
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
    
    return {
        'cm': cm,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


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
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # デバイス
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ (MMDDHHMM形式)
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # 結果ファイル
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
    
    # モデル読み込み
    classifier, unet, diffusion = load_models(args, device)
    
    # Purifier作成
    purifier = DDPMPurifierImproved(
        unet, diffusion, device,
        t_purify=args.t_purify, start_t=args.start_t, eta=args.eta
    )
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FGSM Attack + DDPM Defense Evaluation (ViT Classifier) - DermMel")
    write_and_print(f"{'='*70}")
    write_and_print(f"Classifier: ViT-B/16")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"DDPM Purification: t_purify={args.t_purify}, start_t={args.start_t}, eta={args.eta}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    # 1. クリーン画像の評価
    write_and_print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # 2. クリーン画像 + DDPM浄化
    write_and_print("\n[2/4] Evaluating clean images with DDPM purification...")
    clean_purified_acc, pred_clean_purified, x_purified_clean = evaluate_with_purification(
        purifier, classifier, x_test, y_test, device, args.batch_size, "Purifying clean images"
    )
    l2_clean_purified = compute_l2_norm(x_test, x_purified_clean)
    write_and_print(f"Clean accuracy (with DDPM): {clean_purified_acc:.4f}")
    write_and_print(f"L2 norm (clean vs purified): {l2_clean_purified:.4f}")
    results['clean_acc_with_ddpm'] = clean_purified_acc
    results['l2_clean_vs_purified'] = l2_clean_purified
    
    # 3. FGSM攻撃
    write_and_print("\n[3/4] Running FGSM attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
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
    
    # 4. 敵対的画像 + DDPM浄化
    write_and_print("\n[4/4] Evaluating adversarial images with DDPM purification...")
    adv_purified_acc, pred_adv_purified, x_purified_adv = evaluate_with_purification(
        purifier, classifier, x_adv, y_test, device, args.batch_size, "Purifying adversarial images"
    )
    l2_adv_purified = compute_l2_norm(x_adv, x_purified_adv)
    write_and_print(f"Adversarial accuracy (with DDPM): {adv_purified_acc:.4f}")
    write_and_print(f"L2 norm (adversarial vs purified): {l2_adv_purified:.4f}")
    results['adv_acc_with_ddpm'] = adv_purified_acc
    results['l2_adv_vs_purified'] = l2_adv_purified
    results['defense_improvement'] = adv_purified_acc - adv_acc
    
    # 最終結果
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS (ViT Classifier) - DermMel")
    write_and_print(f"{'='*70}")
    write_and_print(f"Classifier: ViT-B/16")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"DDPM: t_purify={args.t_purify}, start_t={args.start_t}, eta={args.eta}")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With DDPM purification:      {results['clean_acc_with_ddpm']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (FGSM):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With DDPM purification:      {results['adv_acc_with_ddpm']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    write_and_print(f"  Clean vs Purified:           {results['l2_clean_vs_purified']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    write_and_print(f"  Adversarial vs Purified:     {results['l2_adv_vs_purified']:.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Attack time: {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    # 混同行列
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes, results_file)
    cm_results['clean_purified'] = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with DDPM)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_purified'] = print_confusion_matrix(y_true, pred_adv_purified, "4. Adversarial Images (with DDPM)", classes, results_file)
    
    # サンプル画像保存
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(x_test[:10], x_adv[:10], x_purified_clean[:10], x_purified_adv[:10],
                       y_test[:10], classes, samples_dir)
    
    results_file.close()
    
    # JSON形式でも保存
    results_save = {
        'config': vars(args),
        'classifier': 'vit_b_16',
        'dataset': 'dermmel',
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
    print(f"Text results: {os.path.join(log_dir, 'results.txt')}")


if __name__ == '__main__':
    main()
