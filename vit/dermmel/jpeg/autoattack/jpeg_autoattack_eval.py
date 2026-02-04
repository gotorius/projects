"""
DermMel Dataset - AutoAttack + JPEG Compression Defense (ViT Classifier)
AutoAttackによる強力な敵対的攻撃に対するJPEG圧縮防御の検証

AutoAttack:
- APGD-CE: Auto-PGD with cross-entropy loss
- APGD-DLR: Auto-PGD with difference of logits ratio loss  
- FAB: Fast Adaptive Boundary attack
- Square: Square attack (query-based)

実行例:
python jpeg_autoattack_eval.py --epsilon 0.031 --quality 11 --gpu 0
"""

import os
import sys
import io
import argparse
import random
import time
import json

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
from datetime import datetime
from tqdm.auto import tqdm

# AutoAttackのインポート
try:
    from autoattack import AutoAttack
except ImportError:
    print("AutoAttack not found. Install with: pip install git+https://github.com/fra31/auto-attack")
    sys.exit(1)


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel AutoAttack + JPEG Defense (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack perturbation epsilon (pixel scale 0-1)')
    parser.add_argument('--norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Attack norm')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # JPEG圧縮設定
    parser.add_argument('--quality', type=int, default=11,
                        help='JPEG compression quality (1-100, lower = more compression)')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/dermmel/jpeg/autoattack/results',
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
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== JPEG圧縮防御クラス ==========
class JPEGDefense(nn.Module):
    """JPEG圧縮による防御処理"""
    def __init__(self, quality=11):
        super().__init__()
        self.quality = quality
    
    def compress_single(self, img_tensor):
        """単一画像のJPEG圧縮"""
        img = img_tensor.detach().clamp(0, 1).cpu()
        arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=self.quality, subsampling=0, optimize=True)
        buf.seek(0)
        pil_j = Image.open(buf).convert('RGB')
        arr_j = np.array(pil_j).astype(np.float32) / 255.0
        ten_j = torch.from_numpy(arr_j).permute(2, 0, 1)
        return ten_j
    
    def forward(self, x):
        """バッチ処理"""
        device = x.device
        x_list = []
        for i in range(x.size(0)):
            x_list.append(self.compress_single(x[i]))
        return torch.stack(x_list, dim=0).to(device)


class JPEGDefenseWrapper(nn.Module):
    """JPEG圧縮 + ViT分類器のラッパー（AutoAttack用）"""
    def __init__(self, jpeg_defense, classifier, mean, std):
        super().__init__()
        self.jpeg_defense = jpeg_defense
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_jpeg = self.jpeg_defense(x)
        x_norm = (x_jpeg - self.mean) / self.std
        return self.classifier(x_norm)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ViT分類器を読み込み"""
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 2)
    )
    
    checkpoint = torch.load(args.clf_ckpt, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f"Loaded ViT classifier from {args.clf_ckpt}")
    
    return classifier


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NotMelanoma', 'Melanoma'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  - Class distribution: {torch.bincount(y_test).tolist()}")
    print(f"  - Image shape: {x_test.shape}")
    print(f"  - Image range: [{x_test.min():.3f}, {x_test.max():.3f}]")
    return x_test, y_test, classes


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    all_preds = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
    
    return correct / len(x), np.array(all_preds)


# ========== メイン ==========
def main():
    args = parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("DermMel AutoAttack + JPEG Defense Evaluation (ViT)")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.1f}/255)")
    print(f"Norm: {args.norm}")
    print(f"Version: {args.version}")
    print(f"JPEG Quality: {args.quality}")
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    
    # ラッパー作成
    clf_wrapper = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
    clf_wrapper.eval()
    
    # JPEG防御
    jpeg_defense = JPEGDefense(quality=args.quality)
    jpeg_clf_wrapper = JPEGDefenseWrapper(jpeg_defense, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
    jpeg_clf_wrapper.eval()
    
    # ========== 1. クリーン画像の精度 ==========
    print(f"\n[1/4] Evaluating clean images...")
    clean_acc, clean_preds = get_accuracy(clf_wrapper, x_test, y_test, args.batch_size, device)
    print(f"  Clean accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    
    # ========== 2. クリーン画像 + JPEG圧縮の精度 ==========
    print(f"\n[2/4] Evaluating clean images + JPEG compression (quality={args.quality})...")
    clean_jpeg_acc, clean_jpeg_preds = get_accuracy(jpeg_clf_wrapper, x_test, y_test, args.batch_size, device)
    print(f"  Clean + JPEG accuracy: {clean_jpeg_acc:.4f} ({clean_jpeg_acc*100:.2f}%)")
    
    # ========== 3. AutoAttack敵対的画像の生成と評価 ==========
    print(f"\n[3/4] Running AutoAttack ({args.version}, eps={args.epsilon:.5f})...")
    print("  This may take a while...")
    
    # AutoAttack用のモデルを作成
    adversary = AutoAttack(clf_wrapper, norm=args.norm, eps=args.epsilon, 
                           version=args.version, verbose=True)
    
    start_time = time.time()
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), 
                                               bs=args.batch_size)
    attack_time = time.time() - start_time
    print(f"  AutoAttack completed in {attack_time:.1f}s")
    
    # 敵対的画像の精度（防御なし）
    adv_acc, adv_preds = get_accuracy(clf_wrapper, x_adv.cpu(), y_test, args.batch_size, device)
    print(f"  Adversarial accuracy (no defense): {adv_acc:.4f} ({adv_acc*100:.2f}%)")
    
    # ========== 4. AutoAttack敵対的画像 + JPEG圧縮の精度 ==========
    print(f"\n[4/4] Evaluating adversarial images + JPEG compression...")
    adv_jpeg_acc, adv_jpeg_preds = get_accuracy(jpeg_clf_wrapper, x_adv.cpu(), y_test, args.batch_size, device)
    print(f"  Adversarial + JPEG accuracy: {adv_jpeg_acc:.4f} ({adv_jpeg_acc*100:.2f}%)")
    
    # ========== 結果サマリー ==========
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: DermMel")
    print(f"Model: ViT-B/16")
    print(f"Attack: AutoAttack ({args.version}, {args.norm}, eps={args.epsilon:.5f})")
    print(f"Defense: JPEG Compression (quality={args.quality})")
    print(f"-" * 60)
    print(f"{'Condition':<35} {'Accuracy':>10}")
    print(f"-" * 60)
    print(f"{'Clean':<35} {clean_acc:>10.4f}")
    print(f"{'Clean + JPEG':<35} {clean_jpeg_acc:>10.4f}")
    print(f"{'AutoAttack (no defense)':<35} {adv_acc:>10.4f}")
    print(f"{'AutoAttack + JPEG':<35} {adv_jpeg_acc:>10.4f}")
    print(f"{'='*60}")
    print(f"Defense improvement: {adv_jpeg_acc - adv_acc:+.4f} ({(adv_jpeg_acc - adv_acc)*100:+.2f}%)")
    print(f"Clean accuracy drop: {clean_jpeg_acc - clean_acc:+.4f} ({(clean_jpeg_acc - clean_acc)*100:+.2f}%)")
    
    # 結果保存
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'dataset': 'dermmel',
        'model': 'ViT-B/16',
        'attack': 'AutoAttack',
        'attack_version': args.version,
        'norm': args.norm,
        'epsilon': args.epsilon,
        'jpeg_quality': args.quality,
        'clean_acc': clean_acc,
        'clean_jpeg_acc': clean_jpeg_acc,
        'adv_acc': adv_acc,
        'adv_jpeg_acc': adv_jpeg_acc,
        'defense_improvement': adv_jpeg_acc - adv_acc,
        'clean_acc_drop': clean_jpeg_acc - clean_acc,
        'attack_time_sec': attack_time,
        'n_samples': len(x_test),
        'timestamp': timestamp
    }
    
    result_path = os.path.join(args.output_dir, f'autoattack_jpeg_results_{timestamp}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {result_path}")
    
    # 可視化サンプル保存
    n_vis = min(8, len(x_test))
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # クリーン vs 敵対的 vs JPEG圧縮後の比較
    x_adv_cpu = x_adv[:n_vis].cpu()
    x_jpeg = jpeg_defense(x_adv_cpu.to(device)).cpu()
    
    comparison = torch.cat([
        x_test[:n_vis],      # Clean
        x_adv_cpu,           # Adversarial
        x_jpeg               # Adversarial + JPEG
    ], dim=0)
    
    grid = make_grid(comparison, nrow=n_vis, normalize=False, padding=2)
    save_path = os.path.join(vis_dir, f'comparison_autoattack_{timestamp}.png')
    save_image(grid, save_path)
    print(f"Visualization saved to: {save_path}")
    
    # 敵対的サンプルを保存（後で使用可能）
    adv_save_path = os.path.join(args.output_dir, f'autoattack_adv_samples_{timestamp}.pt')
    torch.save({
        'x_adv': x_adv.cpu(),
        'x_clean': x_test,
        'y_test': y_test,
        'epsilon': args.epsilon,
        'attack_version': args.version,
        'classes': classes
    }, adv_save_path)
    print(f"Adversarial samples saved to: {adv_save_path}")


if __name__ == '__main__':
    main()
