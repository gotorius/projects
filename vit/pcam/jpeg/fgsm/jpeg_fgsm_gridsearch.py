"""
PCam Dataset - FGSM Attack + JPEG Compression Defense Grid Search (ViT Classifier)
JPEG Quality パラメータのグリッドサーチスクリプト

評価内容:
- JPEG Quality を5間隔で変化させて防御効果を比較
- Quality: 5, 10, 15, 20, ..., 95, 100

実行例:
python jpeg_fgsm_gridsearch.py --gpu 0
python jpeg_fgsm_gridsearch.py --epsilon 0.03137 --gpu 0
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
import pandas as pd


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='PCam FGSM + JPEG Defense Grid Search (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon (pixel scale 0-1)')
    
    # グリッドサーチ設定
    parser.add_argument('--quality_start', type=int, default=5,
                        help='Starting JPEG quality')
    parser.add_argument('--quality_end', type=int, default=100,
                        help='Ending JPEG quality')
    parser.add_argument('--quality_step', type=int, default=5,
                        help='JPEG quality step size')
    
    # データ設定
    parser.add_argument('--n_samples_per_class', type=int, default=25,
                        help='Number of samples per class')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/pcam/jpeg/fgsm/results',
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
    """ViT分類器のラッパー"""
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        return self.classifier(x_norm)


# ========== JPEG圧縮防御クラス ==========
class JPEGDefense(nn.Module):
    """JPEG圧縮による防御処理"""
    def __init__(self, quality=11):
        super().__init__()
        self.quality = quality
    
    def compress_single(self, img_tensor):
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
        device = x.device
        x_list = []
        for i in range(x.size(0)):
            x_list.append(self.compress_single(x[i]))
        return torch.stack(x_list, dim=0).to(device)


class JPEGDefenseWrapper(nn.Module):
    """JPEG圧縮 + ViT分類器のラッパー"""
    def __init__(self, jpeg_defense, classifier, mean, std):
        super().__init__()
        self.jpeg_defense = jpeg_defense
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_compressed = self.jpeg_defense(x)
        mean = self.mean.to(x_compressed.device)
        std = self.std.to(x_compressed.device)
        x_norm = (x_compressed - mean) / std
        return self.classifier(x_norm)


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x.requires_grad = True
    
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    
    x_adv = x + epsilon * grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    return x_adv


# ========== データ読み込み ==========
def load_cached_samples(cached_path, n_per_class=25):
    """キャッシュされたサンプルを読み込み（各クラスから指定数をサンプリング）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_all = cached['x_test']
    y_all = cached['y_test']
    classes = cached.get('classes', ['normal', 'tumor'])
    
    # 各クラスからn_per_class枚ずつサンプリング
    idx_class0 = (y_all == 0).nonzero(as_tuple=True)[0]
    idx_class1 = (y_all == 1).nonzero(as_tuple=True)[0]
    
    # シャッフルして選択
    perm0 = torch.randperm(len(idx_class0))[:n_per_class]
    perm1 = torch.randperm(len(idx_class1))[:n_per_class]
    
    selected_idx = torch.cat([idx_class0[perm0], idx_class1[perm1]])
    
    x_test = x_all[selected_idx]
    y_test = y_all[selected_idx]
    
    print(f"Selected {len(x_test)} samples ({n_per_class} per class)")
    print(f"  Class 0 ({classes[0]}): {(y_test == 0).sum().item()}")
    print(f"  Class 1 ({classes[1]}): {(y_test == 1).sum().item()}")
    print(f"  x_test shape: {x_test.shape}")
    
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ViT分類器を読み込み"""
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
    
    return classifier


# ========== 精度計算 ==========
def get_accuracy(model, x, y, bs=32, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * bs
            end_idx = min((i + 1) * bs, len(x))
            x_batch = x[start_idx:end_idx].to(device)
            y_batch = y[start_idx:end_idx].to(device)
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    
    return correct / len(x)


# ========== FGSM攻撃実行 ==========
def run_fgsm_attack(model, x_test, y_test, epsilon, device, batch_size=32):
    n_batches = (len(x_test) + batch_size - 1) // batch_size
    x_adv_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        x_batch = x_test[start_idx:end_idx].to(device)
        y_batch = y_test[start_idx:end_idx].to(device)
        
        x_adv_batch = fgsm_attack(model, x_batch, y_batch, epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    
    return torch.cat(x_adv_list, dim=0)


# ========== 単一quality評価 ==========
def evaluate_quality(classifier, x_test, y_test, x_adv, quality, device, batch_size):
    """指定したJPEG qualityでの防御効果を評価"""
    jpeg_defense = JPEGDefense(quality=quality)
    defense_model = JPEGDefenseWrapper(jpeg_defense, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # クリーン画像 + JPEG
    clean_jpeg_acc = get_accuracy(defense_model, x_test, y_test, bs=batch_size, device=device)
    
    # 敵対的画像 + JPEG
    adv_jpeg_acc = get_accuracy(defense_model, x_adv, y_test, bs=batch_size, device=device)
    
    return clean_jpeg_acc, adv_jpeg_acc


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
    log_dir = os.path.join(args.output_dir, f"gridsearch_fgsm_eps{args.epsilon:.4f}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み（各クラス25枚 = 合計50枚）
    x_test, y_test, classes = load_cached_samples(args.cached_samples, n_per_class=args.n_samples_per_class)
    
    # Quality範囲
    qualities = list(range(args.quality_start, args.quality_end + 1, args.quality_step))
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("FGSM Attack + JPEG Defense Grid Search (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"Quality Range: {args.quality_start} to {args.quality_end} (step={args.quality_step})")
    print(f"Qualities to test: {qualities}")
    print(f"Samples: {len(x_test)} ({args.n_samples_per_class} per class)")
    print(f"{'='*70}")
    
    # ========== ベースライン評価 ==========
    print("\n[Step 1/3] Evaluating baseline (no defense)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (no defense): {clean_acc:.4f}")
    
    # ========== FGSM攻撃 ==========
    print("\n[Step 2/3] Running FGSM attack...")
    start_time = time.time()
    x_adv = run_fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device, args.batch_size)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    print(f"Attack time: {attack_time:.2f}s")
    
    # ========== グリッドサーチ ==========
    print(f"\n[Step 3/3] Grid search over JPEG quality...")
    results_list = []
    
    for quality in tqdm(qualities, desc="Quality Grid Search"):
        clean_jpeg_acc, adv_jpeg_acc = evaluate_quality(
            classifier, x_test, y_test, x_adv, quality, device, args.batch_size
        )
        
        defense_improvement = adv_jpeg_acc - adv_acc_no_defense
        clean_degradation = clean_acc - clean_jpeg_acc
        
        results_list.append({
            'quality': quality,
            'clean_acc_no_jpeg': clean_acc,
            'clean_acc_with_jpeg': clean_jpeg_acc,
            'clean_degradation': clean_degradation,
            'adv_acc_no_defense': adv_acc_no_defense,
            'adv_acc_with_jpeg': adv_jpeg_acc,
            'defense_improvement': defense_improvement,
        })
    
    # DataFrameに変換
    df = pd.DataFrame(results_list)
    
    # ==================== 結果表示 ====================
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")
    print(f"{'Quality':>8} | {'Clean':>8} | {'Clean+JPEG':>10} | {'Adv':>8} | {'Adv+JPEG':>10} | {'Defense↑':>10}")
    print("-" * 70)
    for _, row in df.iterrows():
        print(f"{row['quality']:>8} | {row['clean_acc_no_jpeg']:>8.4f} | {row['clean_acc_with_jpeg']:>10.4f} | "
              f"{row['adv_acc_no_defense']:>8.4f} | {row['adv_acc_with_jpeg']:>10.4f} | {row['defense_improvement']:>+10.4f}")
    print("-" * 70)
    
    # ベスト結果
    best_idx = df['defense_improvement'].idxmax()
    best_row = df.loc[best_idx]
    print(f"\n🏆 Best Quality: {int(best_row['quality'])}")
    print(f"   Defense Improvement: {best_row['defense_improvement']:+.4f}")
    print(f"   Adv Accuracy with JPEG: {best_row['adv_acc_with_jpeg']:.4f}")
    print(f"   Clean Degradation: {best_row['clean_degradation']:.4f}")
    
    # クリーン精度劣化が最小のquality
    min_degradation_idx = df['clean_degradation'].idxmin()
    min_degradation_row = df.loc[min_degradation_idx]
    print(f"\n📊 Minimum Clean Degradation: Quality={int(min_degradation_row['quality'])}")
    print(f"   Clean Degradation: {min_degradation_row['clean_degradation']:.4f}")
    print(f"   Defense Improvement: {min_degradation_row['defense_improvement']:+.4f}")
    
    # ==================== 結果保存 ====================
    # CSV保存
    csv_path = os.path.join(log_dir, 'gridsearch_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n📁 CSV saved to: {csv_path}")
    
    # JSON保存
    results_json = {
        'dataset': 'PCam',
        'classifier': 'ViT-B/16',
        'attack': 'FGSM',
        'epsilon': args.epsilon,
        'n_samples': len(x_test),
        'n_samples_per_class': args.n_samples_per_class,
        'quality_range': {
            'start': args.quality_start,
            'end': args.quality_end,
            'step': args.quality_step
        },
        'baseline': {
            'clean_acc': clean_acc,
            'adv_acc_no_defense': adv_acc_no_defense,
        },
        'best_quality': int(best_row['quality']),
        'best_defense_improvement': best_row['defense_improvement'],
        'attack_time': attack_time,
        'results': results_list
    }
    
    json_path = os.path.join(log_dir, 'gridsearch_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"📁 JSON saved to: {json_path}")
    
    # サマリー保存
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PCam - FGSM + JPEG Defense Grid Search (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: FGSM\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"Samples: {len(x_test)} ({args.n_samples_per_class} per class)\n")
        f.write(f"Quality Range: {args.quality_start} to {args.quality_end} (step={args.quality_step})\n\n")
        
        f.write("-"*70 + "\n")
        f.write("BASELINE (No Defense)\n")
        f.write("-"*70 + "\n")
        f.write(f"Clean accuracy: {clean_acc:.4f}\n")
        f.write(f"Adversarial accuracy: {adv_acc_no_defense:.4f}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("GRID SEARCH RESULTS\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Quality':>8} | {'Clean':>8} | {'Clean+JPEG':>10} | {'Adv':>8} | {'Adv+JPEG':>10} | {'Defense↑':>10}\n")
        f.write("-" * 70 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['quality']:>8} | {row['clean_acc_no_jpeg']:>8.4f} | {row['clean_acc_with_jpeg']:>10.4f} | "
                    f"{row['adv_acc_no_defense']:>8.4f} | {row['adv_acc_with_jpeg']:>10.4f} | {row['defense_improvement']:>+10.4f}\n")
        f.write("-" * 70 + "\n\n")
        
        f.write(f"🏆 Best Quality: {int(best_row['quality'])}\n")
        f.write(f"   Defense Improvement: {best_row['defense_improvement']:+.4f}\n")
        f.write(f"   Adv Accuracy with JPEG: {best_row['adv_acc_with_jpeg']:.4f}\n")
    
    print(f"📁 Summary saved to: {summary_path}")
    print(f"\n✅ Grid search completed!")
    
    return df


if __name__ == '__main__':
    main()
