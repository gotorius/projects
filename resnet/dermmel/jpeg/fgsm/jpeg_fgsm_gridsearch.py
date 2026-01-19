"""
DermMel - FGSM Attack + JPEG Compression Defense Grid Search

JPEG圧縮のquality値を2間隔でグリッドサーチして最適値を探索

実行例:
python jpeg_fgsm_gridsearch.py --epsilon 0.031 --quality_min 1 --quality_max 100 --quality_step 2
"""

import os
import sys
import io
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
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel JPEG Defense Grid Search - FGSM Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon')
    
    # JPEG圧縮グリッドサーチ設定
    parser.add_argument('--quality_min', type=int, default=1,
                        help='Minimum JPEG quality')
    parser.add_argument('--quality_max', type=int, default=100,
                        help='Maximum JPEG quality')
    parser.add_argument('--quality_step', type=int, default=2,
                        help='JPEG quality step size')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/jpeg/fgsm/gridsearch_results',
                        help='Output directory')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== JPEG圧縮防御 ==========
def jpeg_compress_batch(images, quality=75):
    """
    バッチ画像にJPEG圧縮を適用
    
    Args:
        images: (B, C, H, W) tensor [0, 1]
        quality: JPEG品質 (1-100)
    
    Returns:
        compressed images: (B, C, H, W) tensor [0, 1]
    """
    compressed = []
    for img in images:
        # [0,1] tensor to PIL Image
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # JPEG圧縮・展開
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # PIL Image to tensor
        compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
        compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
        compressed.append(compressed_tensor)
    
    return torch.stack(compressed)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """分類器を読み込み (DermMel用)"""
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
def evaluate(model, x_test, y_test, device, batch_size=32):
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


def evaluate_with_jpeg(classifier, x_test, y_test, device, quality=75, batch_size=32):
    """JPEG圧縮後の精度を計算（プログレスバーなし版）"""
    classifier.eval()
    
    correct = 0
    total = 0
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size].to(device)
            
            # JPEG圧縮
            x_compressed = jpeg_compress_batch(x_batch, quality=quality).to(device)
            
            # 分類
            x_norm = (x_compressed - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


# ========== ユーティリティ ==========
def compute_l2_norm(x1, x2):
    """L2ノルムを計算"""
    diff = (x1 - x2).view(x1.size(0), -1)
    return torch.norm(diff, p=2, dim=1).mean().item()


def plot_results(results_df, save_path, epsilon):
    """結果をプロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy vs Quality
    ax1 = axes[0, 0]
    ax1.plot(results_df['quality'], results_df['clean_acc_with_jpeg'], 
             'b-o', label='Clean + JPEG', markersize=4)
    ax1.plot(results_df['quality'], results_df['adv_acc_with_jpeg'], 
             'r-s', label='Adversarial + JPEG', markersize=4)
    ax1.axhline(y=results_df['clean_acc'].iloc[0], color='b', linestyle='--', 
                alpha=0.5, label=f'Clean (no JPEG): {results_df["clean_acc"].iloc[0]:.4f}')
    ax1.axhline(y=results_df['adv_acc_no_defense'].iloc[0], color='r', linestyle='--', 
                alpha=0.5, label=f'Adv (no defense): {results_df["adv_acc_no_defense"].iloc[0]:.4f}')
    ax1.set_xlabel('JPEG Quality')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Accuracy vs JPEG Quality (ε={epsilon:.4f})')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 100])
    ax1.set_ylim([0, 1])
    
    # 2. Defense Improvement vs Quality
    ax2 = axes[0, 1]
    ax2.plot(results_df['quality'], results_df['defense_improvement'], 
             'g-^', label='Defense Improvement', markersize=4)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('JPEG Quality')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Defense Improvement (Adv+JPEG - Adv)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 100])
    
    # Best quality annotation
    best_idx = results_df['defense_improvement'].idxmax()
    best_quality = results_df.loc[best_idx, 'quality']
    best_improvement = results_df.loc[best_idx, 'defense_improvement']
    ax2.annotate(f'Best: Q={best_quality}\n+{best_improvement:.4f}', 
                 xy=(best_quality, best_improvement),
                 xytext=(best_quality + 10, best_improvement + 0.02),
                 arrowprops=dict(arrowstyle='->', color='green'),
                 fontsize=9, color='green')
    
    # 3. Clean Accuracy Drop vs Quality
    ax3 = axes[1, 0]
    clean_drop = results_df['clean_acc'].iloc[0] - results_df['clean_acc_with_jpeg']
    ax3.plot(results_df['quality'], clean_drop, 'purple', marker='d', markersize=4)
    ax3.set_xlabel('JPEG Quality')
    ax3.set_ylabel('Accuracy Drop')
    ax3.set_title('Clean Accuracy Drop due to JPEG')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 100])
    
    # 4. Trade-off: Defense Improvement vs Clean Drop
    ax4 = axes[1, 1]
    scatter = ax4.scatter(clean_drop, results_df['defense_improvement'], 
                          c=results_df['quality'], cmap='viridis', s=30)
    plt.colorbar(scatter, ax=ax4, label='JPEG Quality')
    ax4.set_xlabel('Clean Accuracy Drop')
    ax4.set_ylabel('Defense Improvement')
    ax4.set_title('Trade-off: Defense vs Clean Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # Highlight best points
    # Best defense improvement
    ax4.scatter(clean_drop.iloc[best_idx], results_df.loc[best_idx, 'defense_improvement'],
                color='red', s=100, marker='*', label=f'Best Defense (Q={best_quality})')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")


def plot_detailed_accuracy(results_df, save_path, epsilon):
    """詳細な精度プロット"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    qualities = results_df['quality']
    
    ax.fill_between(qualities, results_df['adv_acc_no_defense'].iloc[0], 
                    results_df['adv_acc_with_jpeg'], alpha=0.3, color='green',
                    label='Defense Gain')
    ax.fill_between(qualities, results_df['clean_acc_with_jpeg'], 
                    results_df['clean_acc'].iloc[0], alpha=0.3, color='red',
                    label='Clean Loss')
    
    ax.plot(qualities, results_df['clean_acc_with_jpeg'], 
            'b-o', label='Clean + JPEG', markersize=3)
    ax.plot(qualities, results_df['adv_acc_with_jpeg'], 
            'r-s', label='Adversarial + JPEG', markersize=3)
    ax.axhline(y=results_df['clean_acc'].iloc[0], color='blue', linestyle='--', 
               alpha=0.7, linewidth=2, label=f'Clean baseline: {results_df["clean_acc"].iloc[0]:.4f}')
    ax.axhline(y=results_df['adv_acc_no_defense'].iloc[0], color='red', linestyle='--', 
               alpha=0.7, linewidth=2, label=f'Adversarial baseline: {results_df["adv_acc_no_defense"].iloc[0]:.4f}')
    
    ax.set_xlabel('JPEG Quality', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'JPEG Defense Performance vs Quality (FGSM ε={epsilon:.4f})', fontsize=14)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed plot to {save_path}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%m%d%H%M")
    eps_str = f"eps{args.epsilon:.4f}".replace(".", "p")
    log_dir = os.path.join(args.output_dir, f"{timestamp}_{eps_str}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # 結果ファイル
    results_file = open(os.path.join(log_dir, 'results.txt'), 'w')
    
    def write_and_print(text):
        print(text)
        results_file.write(text + '\n')
        results_file.flush()
    
    # モデルとデータの読み込み
    classifier = load_classifier(args, device)
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # Quality値のリスト作成
    qualities = list(range(args.quality_min, args.quality_max + 1, args.quality_step))
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FGSM Attack + JPEG Defense Grid Search (DermMel)")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"JPEG Quality Range: {args.quality_min} - {args.quality_max} (step={args.quality_step})")
    write_and_print(f"Total quality values to test: {len(qualities)}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}\n")
    
    # ベースライン評価
    write_and_print("[Step 1] Evaluating baselines...")
    clean_acc, _ = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"  Clean accuracy (no JPEG): {clean_acc:.4f}")
    
    # FGSM攻撃の実行
    write_and_print("\n[Step 2] Running FGSM attack...")
    start_time = time.time()
    x_adv_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM Attack"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    x_adv = torch.cat(x_adv_list, dim=0)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense, _ = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    write_and_print(f"  Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    write_and_print(f"  Attack time: {attack_time:.2f}s")
    
    # グリッドサーチ
    write_and_print(f"\n[Step 3] Grid Search over JPEG quality values...")
    write_and_print(f"{'Quality':>10} | {'Clean+JPEG':>12} | {'Adv+JPEG':>12} | {'Improvement':>12}")
    write_and_print("-" * 55)
    
    results_list = []
    
    for quality in tqdm(qualities, desc="Grid Search"):
        # Clean + JPEG
        clean_jpeg_acc = evaluate_with_jpeg(classifier, x_test, y_test, device, 
                                            quality, args.batch_size)
        
        # Adversarial + JPEG
        adv_jpeg_acc = evaluate_with_jpeg(classifier, x_adv, y_test, device, 
                                          quality, args.batch_size)
        
        defense_improvement = adv_jpeg_acc - adv_acc_no_defense
        
        results_list.append({
            'quality': quality,
            'clean_acc': clean_acc,
            'clean_acc_with_jpeg': clean_jpeg_acc,
            'adv_acc_no_defense': adv_acc_no_defense,
            'adv_acc_with_jpeg': adv_jpeg_acc,
            'defense_improvement': defense_improvement,
            'clean_drop': clean_acc - clean_jpeg_acc,
        })
        
        write_and_print(f"{quality:>10} | {clean_jpeg_acc:>12.4f} | {adv_jpeg_acc:>12.4f} | {defense_improvement:>+12.4f}")
    
    # DataFrameに変換
    results_df = pd.DataFrame(results_list)
    
    # 最適なquality値を見つける
    write_and_print(f"\n{'='*70}")
    write_and_print("GRID SEARCH RESULTS SUMMARY")
    write_and_print(f"{'='*70}")
    
    # 最大defense improvement
    best_defense_idx = results_df['defense_improvement'].idxmax()
    best_defense_quality = results_df.loc[best_defense_idx, 'quality']
    best_defense_improvement = results_df.loc[best_defense_idx, 'defense_improvement']
    best_adv_acc = results_df.loc[best_defense_idx, 'adv_acc_with_jpeg']
    best_clean_acc = results_df.loc[best_defense_idx, 'clean_acc_with_jpeg']
    
    write_and_print(f"\n[Best Defense Improvement]")
    write_and_print(f"  Quality: {best_defense_quality}")
    write_and_print(f"  Defense Improvement: {best_defense_improvement:+.4f}")
    write_and_print(f"  Adversarial Accuracy: {best_adv_acc:.4f}")
    write_and_print(f"  Clean Accuracy: {best_clean_acc:.4f}")
    
    # 最大adversarial accuracy
    best_adv_idx = results_df['adv_acc_with_jpeg'].idxmax()
    best_adv_quality = results_df.loc[best_adv_idx, 'quality']
    
    write_and_print(f"\n[Best Adversarial Accuracy with JPEG]")
    write_and_print(f"  Quality: {best_adv_quality}")
    write_and_print(f"  Adversarial Accuracy: {results_df.loc[best_adv_idx, 'adv_acc_with_jpeg']:.4f}")
    write_and_print(f"  Clean Accuracy: {results_df.loc[best_adv_idx, 'clean_acc_with_jpeg']:.4f}")
    
    # Clean精度低下が5%以内で最大defense improvement
    acceptable_drop = 0.05
    acceptable_df = results_df[results_df['clean_drop'] <= acceptable_drop]
    if len(acceptable_df) > 0:
        best_balanced_idx = acceptable_df['defense_improvement'].idxmax()
        best_balanced_quality = results_df.loc[best_balanced_idx, 'quality']
        write_and_print(f"\n[Best Quality with Clean Drop ≤ {acceptable_drop*100:.0f}%]")
        write_and_print(f"  Quality: {best_balanced_quality}")
        write_and_print(f"  Defense Improvement: {results_df.loc[best_balanced_idx, 'defense_improvement']:+.4f}")
        write_and_print(f"  Adversarial Accuracy: {results_df.loc[best_balanced_idx, 'adv_acc_with_jpeg']:.4f}")
        write_and_print(f"  Clean Accuracy: {results_df.loc[best_balanced_idx, 'clean_acc_with_jpeg']:.4f}")
        write_and_print(f"  Clean Drop: {results_df.loc[best_balanced_idx, 'clean_drop']:.4f}")
    
    # 統計情報
    write_and_print(f"\n[Statistics]")
    write_and_print(f"  Defense Improvement - Mean: {results_df['defense_improvement'].mean():.4f}, "
                   f"Std: {results_df['defense_improvement'].std():.4f}")
    write_and_print(f"  Adversarial Accuracy - Mean: {results_df['adv_acc_with_jpeg'].mean():.4f}, "
                   f"Std: {results_df['adv_acc_with_jpeg'].std():.4f}")
    
    write_and_print(f"\n{'='*70}")
    
    results_file.close()
    
    # CSVに保存
    csv_path = os.path.join(log_dir, 'gridsearch_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV results to {csv_path}")
    
    # JSONに保存
    json_results = {
        'config': vars(args),
        'baselines': {
            'clean_acc': float(clean_acc),
            'adv_acc_no_defense': float(adv_acc_no_defense),
        },
        'best_results': {
            'best_defense_improvement': {
                'quality': int(best_defense_quality),
                'defense_improvement': float(best_defense_improvement),
                'adv_acc_with_jpeg': float(best_adv_acc),
                'clean_acc_with_jpeg': float(best_clean_acc),
            },
            'best_adv_acc': {
                'quality': int(best_adv_quality),
                'adv_acc_with_jpeg': float(results_df.loc[best_adv_idx, 'adv_acc_with_jpeg']),
            }
        },
        'all_results': results_df.to_dict(orient='records')
    }
    
    json_path = os.path.join(log_dir, 'gridsearch_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved JSON results to {json_path}")
    
    # プロット作成
    plot_path = os.path.join(log_dir, 'gridsearch_plot.png')
    plot_results(results_df, plot_path, args.epsilon)
    
    detailed_plot_path = os.path.join(log_dir, 'accuracy_detail_plot.png')
    plot_detailed_accuracy(results_df, detailed_plot_path, args.epsilon)
    
    print(f"\nAll results saved to {log_dir}")
    print(f"\n=== BEST QUALITY FOR DEFENSE: {best_defense_quality} ===")


if __name__ == '__main__':
    main()
