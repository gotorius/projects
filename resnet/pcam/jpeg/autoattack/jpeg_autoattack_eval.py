"""
JPEG Compression-based Adversarial Defense Evaluation for PCam Dataset - AutoAttack

JPEG圧縮を用いた敵対的攻撃への防御評価

実行例:
python jpeg_autoattack_eval.py --epsilon 0.031 --version standard --jpeg_quality 11
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as transforms

from autoattack import AutoAttack


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='JPEG Defense Evaluation - AutoAttack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='AutoAttack epsilon')
    parser.add_argument('--version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    parser.add_argument('--n_examples', type=int, default=500,
                        help='Number of examples to attack')
    
    # JPEG圧縮設定
    parser.add_argument('--jpeg_quality', type=int, default=11,
                        help='JPEG compression quality (1-100, lower = more compression)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/pcam/jpeg/autoattack/results',
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


# ========== JPEG圧縮による防御 ==========
class JPEGDefense:
    """JPEG圧縮による防御"""
    def __init__(self, quality=11):
        self.quality = quality
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, x):
        """
        x: (B, 3, H, W), [0, 1] (unnormalized pixel values)
        return: JPEG compressed image (B, 3, H, W), [0, 1]
        """
        batch_size = x.size(0)
        device = x.device
        
        compressed_images = []
        for i in range(batch_size):
            # Tensor → PIL Image
            img = self.to_pil(x[i].cpu())
            
            # JPEG圧縮（メモリ上で実行）
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            
            # 圧縮された画像を読み込み
            compressed_img = Image.open(buffer)
            
            # PIL Image → Tensor
            compressed_tensor = self.to_tensor(compressed_img)
            compressed_images.append(compressed_tensor)
        
        return torch.stack(compressed_images).to(device)


# ========== 正規化付き分類器 (AutoAttack用) ==========
class NormalizedClassifier(nn.Module):
    """ImageNet正規化を含む分類器"""
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """分類器を読み込み"""
    data = torch.load(args.cached_samples, map_location='cpu')
    num_classes = len(data['classes'])
    
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


def evaluate_with_jpeg(jpeg_defense, classifier, x_test, y_test, device, batch_size=16, desc="JPEG compressing"):
    """JPEG圧縮後の精度を計算"""
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_compressed_all = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
        x_batch = x_test[i:i+batch_size].to(device)
        y_batch = y_test[i:i+batch_size].to(device)
        
        # JPEG圧縮
        x_compressed = jpeg_defense(x_batch)
        x_compressed_all.append(x_compressed.cpu())
        
        # 分類
        with torch.no_grad():
            x_norm = (x_compressed - mean) / std
            outputs = classifier(x_norm)
            _, predicted = outputs.max(1)
        
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
        predictions.extend(predicted.cpu().numpy())
    
    x_compressed_all = torch.cat(x_compressed_all, dim=0)
    return correct / total, np.array(predictions), x_compressed_all


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


def save_sample_images(x_clean, x_adv, x_compressed_clean, x_compressed_adv, labels, classes, save_dir):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    
    n = min(len(x_clean), 10)
    
    for i in range(n):
        label = classes[labels[i]]
        
        images = [x_clean[i], x_adv[i], x_compressed_clean[i], x_compressed_adv[i]]
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
    classifier = load_classifier(args, device)
    
    # JPEG防御
    jpeg_defense = JPEGDefense(quality=args.jpeg_quality)
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    # サンプル数制限
    if len(x_test) > args.n_examples:
        x_test = x_test[:args.n_examples]
        y_test = y_test[:args.n_examples]
    
    write_and_print(f"\n{'='*70}")
    write_and_print("AutoAttack + JPEG Defense Evaluation")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"JPEG Compression: Quality={args.jpeg_quality}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    # 1. クリーン画像の評価
    write_and_print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    # 2. クリーン画像 + JPEG圧縮
    write_and_print("\n[2/4] Evaluating clean images with JPEG compression...")
    clean_compressed_acc, pred_clean_compressed, x_compressed_clean = evaluate_with_jpeg(
        jpeg_defense, classifier, x_test, y_test, device, args.batch_size, "Compressing clean images"
    )
    l2_clean_compressed = compute_l2_norm(x_test, x_compressed_clean)
    write_and_print(f"Clean accuracy (with JPEG): {clean_compressed_acc:.4f}")
    write_and_print(f"L2 norm (clean vs compressed): {l2_clean_compressed:.4f}")
    results['clean_acc_with_jpeg'] = clean_compressed_acc
    results['l2_clean_vs_compressed'] = l2_clean_compressed
    
    # 3. AutoAttack
    write_and_print("\n[3/4] Running AutoAttack...")
    
    # 正規化を含むモデル
    model_normalized = NormalizedClassifier(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
    
    # AutoAttack
    autoattack_log = os.path.join(log_dir, 'autoattack.log')
    adversary = AutoAttack(model_normalized, norm='Linf', eps=args.epsilon, version=args.version,
                           device=device, log_path=autoattack_log)
    
    start_time = time.time()
    x_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
    attack_time = time.time() - start_time
    
    x_adv = x_adv.cpu()
    
    l2_clean_adv = compute_l2_norm(x_test, x_adv)
    adv_acc, pred_adv = evaluate(classifier, x_adv, y_test, device, args.batch_size)
    write_and_print(f"L2 norm (clean vs adversarial): {l2_clean_adv:.4f}")
    write_and_print(f"Adversarial accuracy (no defense): {adv_acc:.4f}")
    results['adv_acc_no_defense'] = adv_acc
    results['l2_clean_vs_adv'] = l2_clean_adv
    results['attack_time'] = attack_time
    
    # 4. 敵対的画像 + JPEG圧縮
    write_and_print("\n[4/4] Evaluating adversarial images with JPEG compression...")
    adv_compressed_acc, pred_adv_compressed, x_compressed_adv = evaluate_with_jpeg(
        jpeg_defense, classifier, x_adv, y_test, device, args.batch_size, "Compressing adversarial images"
    )
    l2_adv_compressed = compute_l2_norm(x_adv, x_compressed_adv)
    write_and_print(f"Adversarial accuracy (with JPEG): {adv_compressed_acc:.4f}")
    write_and_print(f"L2 norm (adversarial vs compressed): {l2_adv_compressed:.4f}")
    results['adv_acc_with_jpeg'] = adv_compressed_acc
    results['l2_adv_vs_compressed'] = l2_adv_compressed
    results['defense_improvement'] = adv_compressed_acc - adv_acc
    
    # 最終結果
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: AutoAttack ({args.version}), Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"JPEG Compression: Quality={args.jpeg_quality}")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With JPEG compression:       {results['clean_acc_with_jpeg']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (AutoAttack):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With JPEG compression:       {results['adv_acc_with_jpeg']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    write_and_print(f"  Clean vs Compressed:         {results['l2_clean_vs_compressed']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    write_and_print(f"  Adversarial vs Compressed:   {results['l2_adv_vs_compressed']:.4f}")
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
    cm_results['clean_compressed'] = print_confusion_matrix(y_true, pred_clean_compressed, "2. Clean Images (with JPEG)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_compressed'] = print_confusion_matrix(y_true, pred_adv_compressed, "4. Adversarial Images (with JPEG)", classes, results_file)
    
    # サンプル画像保存
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(x_test[:10], x_adv[:10], x_compressed_clean[:10], x_compressed_adv[:10],
                       y_test[:10], classes, samples_dir)
    
    results_file.close()
    
    # JSON形式でも保存
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
    print(f"Text results: {os.path.join(log_dir, 'results.txt')}")
    print(f"AutoAttack log: {autoattack_log}")


if __name__ == '__main__':
    main()
