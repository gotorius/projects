"""
DermMel - PGD Attack + JPEG Compression Defense Evaluation

JPEG圧縮を使用した敵対的防御評価 (PGD攻撃)

実行例:
python jpeg_pgd_eval.py --epsilon 0.031 --quality 11 --num_steps 20 --step_size 0.003
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


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='DermMel JPEG Defense Evaluation - PGD Attack')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='PGD perturbation epsilon')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Number of PGD steps')
    parser.add_argument('--step_size', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='Use random start for PGD')
    
    # JPEG圧縮設定
    parser.add_argument('--quality', type=int, default=11,
                        help='JPEG compression quality (1-100)')
    
    # パス設定
    parser.add_argument('--cached_samples', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt',
                        help='Path to cached correct samples')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
                        help='Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/dermmel/jpeg/pgd/results',
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


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, step_size, num_steps, random_start, device):
    """PGD攻撃"""
    x = x.clone().to(device)
    y = y.to(device)
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    # ランダム初期化
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
        
        # 摂動の制限
        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)
    
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


def evaluate_with_jpeg(classifier, x_test, y_test, device, quality=75, batch_size=32, desc="JPEG Defense"):
    """JPEG圧縮後の精度を計算"""
    classifier.eval()
    
    correct = 0
    total = 0
    predictions = []
    x_compressed_all = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc=desc):
            x_batch = x_test[i:i+batch_size]
            y_batch = y_test[i:i+batch_size].to(device)
            
            # JPEG圧縮
            x_compressed = jpeg_compress_batch(x_batch, quality=quality).to(device)
            x_compressed_all.append(x_compressed.cpu())
            
            # 分類
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
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    
    write_and_print(f"\n{'='*70}")
    write_and_print("PGD Attack + JPEG Compression Defense Evaluation (DermMel)")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"        Steps: {args.num_steps}, Step Size: {args.step_size:.4f}, Random Start: {args.random_start}")
    write_and_print(f"Defense: JPEG Compression, Quality: {args.quality}")
    write_and_print(f"Samples: {len(x_test)}")
    write_and_print(f"Classes: {classes}")
    write_and_print(f"{'='*70}")
    
    results = {}
    
    write_and_print("\n[1/4] Evaluating clean images (classifier only)...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    write_and_print(f"Clean accuracy: {clean_acc:.4f}")
    results['clean_acc'] = clean_acc
    
    write_and_print("\n[2/4] Evaluating clean images with JPEG compression...")
    clean_jpeg_acc, pred_clean_jpeg, x_compressed_clean = evaluate_with_jpeg(
        classifier, x_test, y_test, device, args.quality, args.batch_size, "Compressing clean images"
    )
    l2_clean_compressed = compute_l2_norm(x_test, x_compressed_clean)
    write_and_print(f"Clean accuracy (with JPEG): {clean_jpeg_acc:.4f}")
    write_and_print(f"L2 norm (clean vs compressed): {l2_clean_compressed:.4f}")
    results['clean_acc_with_jpeg'] = clean_jpeg_acc
    results['l2_clean_vs_compressed'] = l2_clean_compressed
    
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
    
    write_and_print("\n[4/4] Evaluating adversarial images with JPEG compression...")
    adv_jpeg_acc, pred_adv_jpeg, x_compressed_adv = evaluate_with_jpeg(
        classifier, x_adv, y_test, device, args.quality, args.batch_size, "Compressing adversarial images"
    )
    l2_adv_compressed = compute_l2_norm(x_adv, x_compressed_adv)
    write_and_print(f"Adversarial accuracy (with JPEG): {adv_jpeg_acc:.4f}")
    write_and_print(f"L2 norm (adversarial vs compressed): {l2_adv_compressed:.4f}")
    results['adv_acc_with_jpeg'] = adv_jpeg_acc
    results['l2_adv_vs_compressed'] = l2_adv_compressed
    results['defense_improvement'] = adv_jpeg_acc - adv_acc
    
    write_and_print(f"\n{'='*70}")
    write_and_print("FINAL RESULTS")
    write_and_print(f"{'='*70}")
    write_and_print(f"Attack: PGD, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    write_and_print(f"        Steps: {args.num_steps}, Step Size: {args.step_size:.4f}")
    write_and_print(f"Defense: JPEG Compression, Quality: {args.quality}")
    write_and_print(f"-"*70)
    write_and_print("Clean Accuracy:")
    write_and_print(f"  Classifier only:             {results['clean_acc']:.4f}")
    write_and_print(f"  With JPEG:                   {results['clean_acc_with_jpeg']:.4f}")
    write_and_print(f"-"*70)
    write_and_print("Adversarial Accuracy (PGD):")
    write_and_print(f"  Without defense:             {results['adv_acc_no_defense']:.4f}")
    write_and_print(f"  With JPEG:                   {results['adv_acc_with_jpeg']:.4f}")
    write_and_print(f"  Defense improvement:         {results['defense_improvement']:+.4f}")
    write_and_print(f"-"*70)
    write_and_print("L2 Norms:")
    write_and_print(f"  Clean vs Compressed:         {results['l2_clean_vs_compressed']:.4f}")
    write_and_print(f"  Clean vs Adversarial:        {results['l2_clean_vs_adv']:.4f}")
    write_and_print(f"  Adversarial vs Compressed:   {results['l2_adv_vs_compressed']:.4f}")
    write_and_print(f"-"*70)
    write_and_print(f"Attack time: {attack_time:.2f}s")
    write_and_print(f"{'='*70}")
    
    write_and_print(f"\n{'='*70}")
    write_and_print("Confusion Matrices")
    write_and_print(f"{'='*70}")
    
    y_true = y_test.numpy()
    cm_results = {}
    cm_results['clean'] = print_confusion_matrix(y_true, pred_clean, "1. Clean Images", classes, results_file)
    cm_results['clean_jpeg'] = print_confusion_matrix(y_true, pred_clean_jpeg, "2. Clean Images (with JPEG)", classes, results_file)
    cm_results['adv_no_defense'] = print_confusion_matrix(y_true, pred_adv, "3. Adversarial Images (No Defense)", classes, results_file)
    cm_results['adv_jpeg'] = print_confusion_matrix(y_true, pred_adv_jpeg, "4. Adversarial Images (with JPEG)", classes, results_file)
    
    write_and_print("\nSaving sample images...")
    samples_dir = os.path.join(log_dir, 'samples')
    save_sample_images(x_test[:10], x_adv[:10], x_compressed_clean[:10], x_compressed_adv[:10],
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
