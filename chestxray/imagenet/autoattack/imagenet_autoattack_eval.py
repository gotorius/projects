"""
ChestX-ray Dataset - AutoAttack + Guided-Diffusion (ImageNet Pretrained) Defense
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
1. クリーン画像の分類精度
2. クリーン画像を浄化した後の分類精度
3. AutoAttack敵対的画像の分類精度（防御なし）
4. AutoAttack敵対的画像を浄化した後の分類精度（防御あり）
"""

"""
# 基本実行（デフォルト設定）
python imagenet_autoattack_eval.py

# パラメータ指定
python imagenet_autoattack_eval.py \
    --epsilon 0.03137 \
    --start_t 80 \
    --T_purify 50 \
    --cached_samples /mnt/data1/gotou/projects/chestxray/correct_samples_500.pt \
    --gpu 0
"""

import os
import sys
import argparse
import random
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
from tqdm.auto import tqdm

# Guided-diffusionモジュールのインポート
sys.path.insert(0, '/mnt/data1/gotou/kaggle/guided-diffusion')
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

# AutoAttack
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not installed. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray AutoAttack + Guided-Diffusion Defense')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon (pixel scale 0-1)')
    parser.add_argument('--attack_version', type=str, default='standard',
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
    
    # キャッシュされたサンプル
    parser.add_argument('--cached_samples', type=str, 
                        default='/mnt/data1/gotou/projects/chestxray/correct_samples_500.pt',
                        help='Path to cached samples (.pt file)')
    
    # パス設定
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray',
                        help='Data directory')
    parser.add_argument('--diffusion_ckpt', type=str, 
                        default='/mnt/data1/gotou/kaggle/guided-diffusion/256x256_diffusion_uncond.pt',
                        help='Guided-Diffusion checkpoint path')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/chestxray/imagenet/autoattack/results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== 分類器ラッパー ==========
class ClassifierWrapper(nn.Module):
    """分類器のラッパー
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
        x_purified_pixel = self.diffusion_to_pixel(x_purified)
        
        # 元のサイズに戻す
        if original_size != (256, 256):
            x_purified_pixel = F.interpolate(x_purified_pixel, size=original_size, mode='bilinear', align_corners=False)
        
        return x_purified_pixel
    
    def forward(self, x_pixel):
        return self.purify(x_pixel)


class DiffusionDefenseWrapper(nn.Module):
    """Guided-Diffusion浄化 + 分類器のラッパー
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
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
def load_cached_samples(cached_path, device):
    """キャッシュされたサンプルを読み込み"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    print(f"Loaded {len(x_test)} samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    return x_test, y_test


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """分類器を読み込み"""
    classifier = models.resnet50(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    ckpt = torch.load(args.clf_ckpt, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
    else:
        classifier.load_state_dict(ckpt)
    classifier = classifier.to(device).eval()
    print(f"Loaded classifier from {args.clf_ckpt}")
    return classifier


def load_guided_diffusion(args, device):
    """Guided-Diffusionモデルを読み込み"""
    print("\nLoading Guided-Diffusion model (ImageNet pretrained)...")
    
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
    state_dict = torch.load(args.diffusion_ckpt, map_location=device)
    diffusion_model.load_state_dict(state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()
    
    print(f"Loaded Guided-Diffusion model from {args.diffusion_ckpt}")
    print(f"Model image size: 256x256")
    print(f"Diffusion steps: {diffusion.num_timesteps}")
    
    return diffusion_model, diffusion


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=8, device=None):
    """モデルの精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Evaluating", leave=False):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    
    return correct / len(x)


# ========== 予測取得 ==========
def get_predictions(model, x, bs=8, device=None):
    """モデルの予測を取得"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Predicting", leave=False):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds.append(outputs.argmax(dim=1).cpu())
    
    return torch.cat(preds).numpy()


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


# ========== AutoAttack実行 ==========
def run_autoattack(model, x_test, y_test, epsilon, version, device, batch_size=8):
    """AutoAttackを実行して敵対的サンプルを生成"""
    print(f"\nRunning AutoAttack with epsilon={epsilon:.4f}, version={version}...")
    
    # デバイスを明示的に指定してAutoAttackを初期化
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version=version, verbose=True, device=device)
    adversary.seed = 42
    
    x_adv = adversary.run_standard_evaluation(
        x_test.to(device), 
        y_test.to(device), 
        bs=batch_size
    )
    
    print(f"Generated {len(x_adv)} adversarial samples")
    
    return x_adv.cpu()


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_purified_clean, x_purified_adv, 
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # 4枚を並べて保存: Clean, Clean+Diffusion, Adv, Adv+Diffusion
        quad = torch.cat([
            x_clean[i:i+1],
            x_purified_clean[i:i+1],
            x_adv[i:i+1],
            x_purified_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Clean+Diffusion | Adversarial | Adv+Diffusion]")


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.output_dir, f"autoattack_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    diffusion_model, diffusion = load_guided_diffusion(args, device)
    
    # 浄化器
    purifier = GuidedDiffusionPurifier(
        diffusion_model, diffusion, device,
        start_t=args.start_t,
        T_purify=args.T_purify,
        eta=args.eta
    )
    
    # ラッパー作成
    classifier_model = ClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD, input_size=224).to(device).eval()
    defense_model = DiffusionDefenseWrapper(purifier, classifier, IMAGENET_MEAN, IMAGENET_STD, input_size=224).to(device).eval()
    
    # クラス名
    test_dir = os.path.join(args.data_dir, 'test')
    classes = sorted([d.name for d in Path(test_dir).iterdir() if d.is_dir()])
    print(f"Classes: {classes}")
    
    # データ読み込み
    x_test, y_test = load_cached_samples(args.cached_samples, device)
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("AutoAttack + Guided-Diffusion (ImageNet) Defense Evaluation")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Attack Version: {args.attack_version}")
    print(f"Diffusion: start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像を浄化した後の精度 ==========
    print("\n[2/4] Evaluating clean images with Guided-Diffusion purification...")
    clean_purified_acc = get_accuracy(defense_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with Diffusion): {clean_purified_acc:.4f}")
    results['clean_acc_with_diffusion'] = clean_purified_acc
    
    # ========== 3. AutoAttack & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running AutoAttack and evaluating adversarial images...")
    start_time = time.time()
    x_adv = run_autoattack(classifier_model, x_test, y_test, args.epsilon, args.attack_version, device, args.batch_size)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像を浄化した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with Guided-Diffusion purification...")
    adv_defended_acc = get_accuracy(defense_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with Diffusion): {adv_defended_acc:.4f}")
    results['adv_acc_with_diffusion'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Attack: AutoAttack ({args.attack_version}), Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Defense: Guided-Diffusion (ImageNet), start_t={args.start_t}, T_purify={args.T_purify}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  Classifier only:          {results['clean_acc_classifier']:.4f}")
    print(f"  With Diffusion:           {results['clean_acc_with_diffusion']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (AutoAttack):")
    print(f"  Without defense:          {results['adv_acc_no_defense']:.4f}")
    print(f"  With Diffusion:           {results['adv_acc_with_diffusion']:.4f}")
    print(f"  Defense improvement:      {results['defense_improvement']:+.4f}")
    print(f"-"*70)
    print(f"Attack time: {results['attack_time']:.2f}s")
    print(f"{'='*70}")
    
    # ==================== 混同行列 ====================
    print(f"\n{'='*70}")
    print("Confusion Matrices")
    print(f"{'='*70}")
    
    # 予測取得
    pred_clean = get_predictions(classifier_model, x_test, bs=args.batch_size, device=device)
    pred_clean_purified = get_predictions(defense_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions(defense_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (Classifier only)", classes)
    cm_clean_purified = print_confusion_matrix(y_true, pred_clean_purified, "2. Clean Images (with Diffusion)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with Diffusion)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_purified': cm_clean_purified,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== 浄化画像を生成して保存 ====================
    print("\nGenerating purified samples for visualization...")
    n_samples = min(10, len(x_test))
    x_purified_clean = []
    x_purified_adv = []
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Generating samples"):
            x_purified_clean.append(purifier(x_test[i:i+1].to(device)).cpu())
            x_purified_adv.append(purifier(x_adv[i:i+1].to(device)).cpu())
    
    x_purified_clean = torch.cat(x_purified_clean, dim=0)
    x_purified_adv = torch.cat(x_purified_adv, dim=0)
    
    save_sample_images(
        x_test[:n_samples].cpu(), 
        x_adv[:n_samples].cpu(),
        x_purified_clean,
        x_purified_adv,
        y_test[:n_samples].cpu().numpy(), 
        classes,
        os.path.join(log_dir, 'samples')
    )
    
    # ==================== 敵対的サンプル保存 ====================
    torch.save({
        'x_clean': x_test.cpu(),
        'x_adv': x_adv.cpu(),
        'y': y_test.cpu(),
        'epsilon': args.epsilon,
        'attack_version': args.attack_version,
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - AutoAttack + Guided-Diffusion (ImageNet) Defense\n")
        f.write("="*70 + "\n\n")
        f.write(f"Attack: AutoAttack ({args.attack_version})\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Defense: Guided-Diffusion (ImageNet), start_t={args.start_t}, T_purify={args.T_purify}\n")
        f.write(f"Diffusion Model: {args.diffusion_ckpt}\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  Classifier only:          {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With Diffusion:           {results['clean_acc_with_diffusion']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (AutoAttack):\n")
        f.write(f"  Without defense:          {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With Diffusion:           {results['adv_acc_with_diffusion']:.4f}\n")
        f.write(f"  Defense improvement:      {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (Classifier)", cm_clean), 
                         ("Clean (with Diffusion)", cm_clean_purified),
                         ("Adversarial (No Defense)", cm_adv_no_def),
                         ("Adversarial (with Diffusion)", cm_adv_defended)]:
            if cm:
                f.write(f"{name}:\n")
                f.write(f"  TN: {cm['tn']:4d}  FP: {cm['fp']:4d}\n")
                f.write(f"  FN: {cm['fn']:4d}  TP: {cm['tp']:4d}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.4f}\n")
                f.write(f"  Precision: {cm['precision']:.4f}, Recall: {cm['recall']:.4f}, F1: {cm['f1']:.4f}\n\n")
    
    # JSON形式でも保存
    results_json = {
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
    print(f"✅ Summary: {summary_path}")
    
    return results


if __name__ == '__main__':
    main()
