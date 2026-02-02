"""
PCam Dataset - AutoAttack + Guided-Diffusion (ImageNet Pretrained) Defense (ViT Classifier)
AutoAttackによる強力な敵対的攻撃に対するGuided-Diffusion防御の検証

AutoAttack:
- APGD-CE: Auto-PGD with cross-entropy loss
- APGD-DLR: Auto-PGD with difference of logits ratio loss  
- FAB: Fast Adaptive Boundary attack
- Square: Square attack (query-based)

実行例:
python imagenet_autoattack_eval.py --epsilon 0.031 --start_t 80 --T_purify 50 --gpu 0
"""

import os
import sys
import argparse
import random
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Guided-diffusionモジュールのインポート
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

# AutoAttackのインポート
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not found. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='PCam AutoAttack + Guided-Diffusion Defense (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Attack norm')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # 拡散モデル浄化設定
    parser.add_argument('--start_t', type=int, default=80,
                        help='Diffusion start timestep')
    parser.add_argument('--T_purify', type=int, default=50,
                        help='Number of purification steps')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM sampling eta (0=deterministic)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--diffusion_ckpt', type=str, 
                        default='/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt',
                        help='Guided-Diffusion checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/imagenet/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
    """ViT分類器のラッパー
    入力: [0,1]のRGB画像 (任意サイズ)
    出力: 2クラスロジット
    """
    def __init__(self, classifier, mean, std, input_size=224):
        super().__init__()
        self.classifier = classifier
        self.input_size = input_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 2クラスロジット"""
        # 分類器入力サイズにリサイズ
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        return self.classifier(x_norm)


# ========== Guided-Diffusion浄化クラス ==========
class GuidedDiffusionPurifier(nn.Module):
    """Guided-Diffusion (ImageNet Pretrained) による浄化
    
    入力: RGB画像 [0,1]
    出力: 浄化後のRGB画像 [0,1]
    """
    def __init__(self, diffusion_model, diffusion, device, start_t=80, T_purify=50, eta=0.0):
        super().__init__()
        self.model = diffusion_model
        self.diffusion = diffusion
        self.device = device
        self.start_t = start_t
        self.T_purify = T_purify
        self.eta = eta
    
    def pixel_to_diffusion(self, x_pixel):
        """[0,1] → [-1,1]"""
        return x_pixel * 2.0 - 1.0
    
    def diffusion_to_pixel(self, x_diff):
        """[-1,1] → [0,1]"""
        return torch.clamp((x_diff + 1.0) / 2.0, 0, 1)
    
    @torch.no_grad()
    def purify(self, x_pixel):
        """
        RGB画像 [0,1] を浄化
        
        注意: 入力は224x224だが、Guided-Diffusionは256x256を期待
        """
        b = x_pixel.size(0)
        original_size = x_pixel.shape[-2:]
        
        # 256x256にリサイズ（Guided-Diffusionの入力サイズ）
        if original_size != (256, 256):
            x_pixel = F.interpolate(x_pixel, size=(256, 256), mode='bilinear', align_corners=False)
        
        # [0,1] → [-1,1]
        x_diff = self.pixel_to_diffusion(x_pixel)
        
        # Forward diffusion to start_t
        t = torch.full((b,), self.start_t, device=self.device, dtype=torch.long)
        noise = torch.randn_like(x_diff)
        x_t = self.diffusion.q_sample(x_diff, t, noise=noise)
        
        # Reverse diffusion
        end_t = max(self.start_t - self.T_purify, 0)
        indices = list(range(self.start_t, end_t, -1))
        
        for i in indices:
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            # モデル予測
            out = self.diffusion.p_mean_variance(
                self.model, x_t, t,
                clip_denoised=True,
                denoised_fn=None,
                model_kwargs={}
            )
            
            # DDIM step
            if i > 0:
                nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
                x_t = out["mean"] + nonzero_mask * (self.eta * torch.sqrt(out["variance"])) * torch.randn_like(x_t)
            else:
                x_t = out["mean"]
        
        x_purified = torch.clamp(x_t, -1.0, 1.0)
        
        # [-1,1] → [0,1]
        x_purified = self.diffusion_to_pixel(x_purified)
        
        # 元のサイズに戻す
        if original_size != (256, 256):
            x_purified = F.interpolate(x_purified, size=original_size, mode='bilinear', align_corners=False)
        
        return x_purified
    
    def forward(self, x_pixel):
        return self.purify(x_pixel)


class GuidedDiffusionDefenseWrapper(nn.Module):
    """Guided-Diffusion浄化 + ViT分類器のラッパー"""
    def __init__(self, purifier, classifier, mean, std, input_size=224):
        super().__init__()
        self.purifier = purifier
        self.classifier = classifier
        self.input_size = input_size
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 浄化 → 2クラスロジット"""
        x_purified = self.purifier(x)
        
        # 分類器入力サイズにリサイズ
        if x_purified.shape[-1] != self.input_size or x_purified.shape[-2] != self.input_size:
            x_purified = F.interpolate(x_purified, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        mean = self.mean.to(x_purified.device)
        std = self.std.to(x_purified.device)
        x_norm = (x_purified - mean) / std
        return self.classifier(x_norm)


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み（ViT分類器で正しく分類されたサンプル）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['normal', 'tumor'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_models(args, device):
    """ViT分類器とGuided-Diffusionを読み込み"""
    # ViT分類器（2クラス: normal, tumor）
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 2)
    )
    
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    print(f"Loaded ViT classifier from {args.clf_ckpt}")
    
    # Guided-Diffusion (ImageNet 256x256 unconditional)
    print(f"\nLoading Guided-Diffusion from: {args.diffusion_ckpt}")
    
    model_config = model_and_diffusion_defaults()
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
        'use_fp16': False,
        'use_scale_shift_norm': True,
    })
    
    diffusion_model, diffusion = create_model_and_diffusion(**model_config)
    diffusion_model.load_state_dict(
        torch.load(args.diffusion_ckpt, map_location="cpu")
    )
    diffusion_model = diffusion_model.to(device).eval()
    print("Loaded Guided-Diffusion model")
    
    return classifier, diffusion_model, diffusion


# ========== 予測取得と精度計算（統合） ==========
def get_predictions_and_accuracy(model, x, y, bs=32, device=None, desc="Evaluation"):
    """モデルの予測を取得して精度も計算（重複計算を避けるため統合）
    
    Returns:
        predictions: numpy array of predictions
        accuracy: float accuracy value
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc=desc, total=n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            batch_preds = outputs.argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    
    predictions = torch.cat(preds).numpy()
    accuracy = correct / len(x)
    
    return predictions, accuracy


def get_predictions_and_accuracy_with_purification(model, purifier, x, y, bs=4, device=None, desc="Purification"):
    """浄化付きで予測と精度を取得（重複計算を避けるため統合）
    
    Returns:
        predictions: numpy array of predictions
        accuracy: float accuracy value
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    correct = 0
    
    for i in tqdm(range(n_batches), desc=desc, total=n_batches):
        start_idx = i * bs
        end_idx = min((i + 1) * bs, len(x))
        x_batch = x[start_idx:end_idx].to(device)
        y_batch = y[start_idx:end_idx].to(device)
        
        # 浄化
        x_purified = purifier.purify(x_batch)
        
        with torch.no_grad():
            outputs = model(x_purified)
            batch_preds = outputs.argmax(dim=1)
            preds.append(batch_preds.cpu())
            correct += (batch_preds == y_batch).sum().item()
    
    predictions = torch.cat(preds).numpy()
    accuracy = correct / len(x)
    
    return predictions, accuracy


# ========== 後方互換性のためのラッパー関数 ==========
def get_accuracy(model, x, y, bs=32, device=None, desc="Computing accuracy"):
    """モデルの精度を計算（後方互換性用）"""
    _, acc = get_predictions_and_accuracy(model, x, y, bs, device, desc)
    return acc


def get_accuracy_with_purification(model, purifier, x, y, bs=4, device=None, desc="Purification"):
    """浄化付きで精度を計算（後方互換性用）"""
    _, acc = get_predictions_and_accuracy_with_purification(model, purifier, x, y, bs, device, desc)
    return acc


def get_predictions(model, x, bs=32, device=None, desc="Getting predictions"):
    """モデルの予測を取得（後方互換性用）"""
    preds, _ = get_predictions_and_accuracy(model, x, torch.zeros(len(x)), bs, device, desc)
    return preds


def get_predictions_with_purification(model, purifier, x, bs=4, device=None, desc="Getting predictions with purification"):
    """浄化付きで予測を取得（後方互換性用）"""
    preds, _ = get_predictions_and_accuracy_with_purification(model, purifier, x, torch.zeros(len(x)), bs, device, desc)
    return preds


# ========== 混同行列出力 ==========
def print_confusion_matrix(y_true, y_pred, title, classes=None):
    """混同行列をテキスト出力"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        
        print(f"\n{title}:")
        if classes:
            print(f"  Classes: {classes}")
        print(f"  TN: {tn:4d}  FP: {fp:4d}")
        print(f"  FN: {fn:4d}  TP: {tp:4d}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
                'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    return {}


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified, y_true, preds_clean, preds_adv, preds_defended,
                       classes, save_dir, max_samples=10):
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        pred_clean = classes[preds_clean[i]] if classes else str(preds_clean[i])
        pred_adv = classes[preds_adv[i]] if classes else str(preds_adv[i])
        pred_def = classes[preds_defended[i]] if classes else str(preds_defended[i])
        
        quad = torch.cat([x_clean[i:i+1], x_adv[i:i+1], x_purified[i:i+1]], dim=0)
        grid = make_grid(quad, nrow=3, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}_clean{pred_clean}_adv{pred_adv}_def{pred_def}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # GPU設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"autoattack_eps{args.epsilon:.4f}_t{args.start_t}_T{args.T_purify}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier, diffusion_model, diffusion = load_models(args, device)
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # Guided-Diffusion浄化器
    purifier = GuidedDiffusionPurifier(
        diffusion_model, diffusion, device,
        start_t=args.start_t, T_purify=args.T_purify, eta=args.eta
    ).to(device)
    
    # 防御付き分類器
    defended_model = GuidedDiffusionDefenseWrapper(
        purifier, classifier, IMAGENET_MEAN, IMAGENET_STD
    ).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + Guided-Diffusion (ImageNet Pretrained) Defense Evaluation")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"Version: {args.version}")
    print(f"Diffusion: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    pred_clean, clean_acc = get_predictions_and_accuracy(
        classifier_model, x_test, y_test, bs=args.batch_size, device=device,
        desc="Evaluating clean images (classifier only)"
    )
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with Guided-Diffusion purification...")
    pred_clean_purified, clean_purified_acc = get_predictions_and_accuracy_with_purification(
        classifier_model, purifier, x_test, y_test, bs=4, device=device,
        desc="Evaluating clean images (with purification)"
    )
    print(f"Clean accuracy (with purification): {clean_purified_acc:.4f}")
    results['clean_acc_with_diffusion'] = clean_purified_acc
    
    # ========== 3. AutoAttack ==========
    print("\n[3/4] Running AutoAttack...")
    start_time = time.time()
    
    adversary = AutoAttack(classifier_model, norm=args.norm, eps=args.epsilon, version=args.version, device=device)
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
    
    attack_time = time.time() - start_time
    print(f"AutoAttack completed in {attack_time:.2f}s")
    
    pred_adv_no_def, adv_acc_no_defense = get_predictions_and_accuracy(
        classifier_model, x_adv, y_test, bs=args.batch_size, device=device,
        desc="Evaluating adversarial images (no defense)"
    )
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度 ==========
    print("\n[4/4] Evaluating adversarial images with Guided-Diffusion purification...")
    pred_adv_defended, adv_defended_acc = get_predictions_and_accuracy_with_purification(
        classifier_model, purifier, x_adv, y_test, bs=4, device=device,
        desc="Evaluating adversarial images (with purification)"
    )
    print(f"Adversarial accuracy (with purification): {adv_defended_acc:.4f}")
    results['adv_acc_with_diffusion'] = adv_defended_acc
    
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f}, Norm: {args.norm}")
    print(f"Defense: Guided-Diffusion (ImageNet 256x256 unconditional)")
    print(f"         start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  ViT classifier only:         {results['clean_acc_classifier']:.4f}")
    print(f"  With Guided-Diffusion:       {results['clean_acc_with_diffusion']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (AutoAttack):")
    print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    print(f"  With Guided-Diffusion:       {results['adv_acc_with_diffusion']:.4f}")
    print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # 注: 予測は既に上で計算済み（重複計算を避けるため）
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with Guided-Diffusion)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. AutoAttack Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. AutoAttack Images (with Guided-Diffusion)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== サンプル画像保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified = []
    for i in tqdm(range(n_samples), desc="Purifying samples for visualization"):
        x_pur = purifier.purify(x_adv[i:i+1].to(device))
        x_purified.append(x_pur.cpu())
    x_purified = torch.cat(x_purified, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(),
        x_adv[:n_samples].cpu(),
        x_purified,
        y_test[:n_samples].cpu().numpy(),
        pred_clean[:n_samples],
        pred_adv_no_def[:n_samples],
        pred_adv_defended[:n_samples],
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # ==================== 結果保存 ====================
    torch.save({
        'x_clean': x_test.cpu(),
        'x_adv': x_adv.cpu(),
        'y': y_test.cpu(),
        'epsilon': args.epsilon,
        'attack': 'autoattack',
        'version': args.version,
        'norm': args.norm,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    
    # サマリー保存
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PCam - AutoAttack + Guided-Diffusion Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: AutoAttack ({args.version})\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Norm: {args.norm}\n")
        f.write(f"Diffusion: start_t={args.start_t}, T_purify={args.T_purify}\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  ViT classifier only:         {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With Guided-Diffusion:       {results['clean_acc_with_diffusion']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (AutoAttack):\n")
        f.write(f"  Without defense:             {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With Guided-Diffusion:       {results['adv_acc_with_diffusion']:.4f}\n")
        f.write(f"  Defense improvement:         {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n")
    
    # JSON保存
    results_json = {
        'dataset': 'PCam',
        'classifier': 'ViT-B/16',
        'defense': 'Guided-Diffusion',
        'attack': 'autoattack',
        'version': args.version,
        'norm': args.norm,
        'epsilon': args.epsilon,
        'diffusion_start_t': args.start_t,
        'diffusion_T_purify': args.T_purify,
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'clean_acc_with_diffusion': results['clean_acc_with_diffusion'],
        'adv_acc_no_defense': results['adv_acc_no_defense'],
        'adv_acc_with_diffusion': results['adv_acc_with_diffusion'],
        'defense_improvement': results['defense_improvement'],
        'attack_time': results['attack_time'],
    }
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to: {log_dir}")
    
    return results


if __name__ == '__main__':
    main()
