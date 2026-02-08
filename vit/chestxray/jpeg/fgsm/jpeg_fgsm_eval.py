"""
ChestX-ray Dataset - FGSM Attack + JPEG Compression Defense (ViT Classifier)
DiffPureスタイルの敵対的防御検証スクリプト

評価内容:
1. クリーン画像の分類精度
2. クリーン画像をJPEG圧縮した後の分類精度
3. FGSM敵対的画像の分類精度（防御なし）
4. FGSM敵対的画像をJPEG圧縮した後の分類精度（防御あり）
"""

"""
# 基本実行（デフォルト設定: quality=11）
python jpeg_fgsm_eval.py

# パラメータ指定
python jpeg_fgsm_eval.py \
    --epsilon 0.03137 \
    --quality 11 \
    --gpu 0
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


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray FGSM Attack + JPEG Defense (ViT)')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='FGSM perturbation epsilon (pixel scale 0-1)')
    
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
                        default='/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt',
                        help='Path to cached samples (.pt file)')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/chestxray/jpeg/fgsm/results',
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
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 2クラスロジット"""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x_norm = (x - mean) / std
        return self.classifier(x_norm)


# ========== JPEG圧縮防御クラス ==========
class JPEGDefense(nn.Module):
    """JPEG圧縮による防御処理
    
    入力: RGB画像 [0,1]
    出力: JPEG圧縮後のRGB画像 [0,1]
    """
    def __init__(self, quality=11):
        super().__init__()
        self.quality = quality
    
    def compress_single(self, img_tensor):
        """単一画像のJPEG圧縮"""
        img = img_tensor.detach().clamp(0, 1).cpu()
        arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)  # HWC, uint8
        pil = Image.fromarray(arr)
        buf = io.BytesIO()
        # subsampling=0 で 4:4:4 を指定
        pil.save(buf, format='JPEG', quality=self.quality, subsampling=0, optimize=True)
        buf.seek(0)
        pil_j = Image.open(buf).convert('RGB')
        arr_j = np.array(pil_j).astype(np.float32) / 255.0
        ten_j = torch.from_numpy(arr_j).permute(2, 0, 1)  # CHW
        return ten_j
    
    def forward(self, x):
        """バッチ処理"""
        device = x.device
        x_list = []
        for i in range(x.size(0)):
            x_list.append(self.compress_single(x[i]))
        return torch.stack(x_list, dim=0).to(device)


class JPEGDefenseWrapper(nn.Module):
    """JPEG圧縮 + ViT分類器のラッパー
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, jpeg_defense, classifier, mean, std):
        super().__init__()
        self.jpeg_defense = jpeg_defense
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → JPEG圧縮 → 2クラスロジット"""
        x_compressed = self.jpeg_defense(x)
        mean = self.mean.to(x_compressed.device)
        std = self.std.to(x_compressed.device)
        x_norm = (x_compressed - mean) / std
        return self.classifier(x_norm)


# ========== FGSM攻撃 ==========
def fgsm_attack(model, x, y, epsilon, device):
    """
    FGSM攻撃
    
    Args:
        model: 分類器（入力は[0,1]のRGB画像）
        x: 入力画像 [B, 3, H, W] in [0, 1]
        y: ラベル [B]
        epsilon: 摂動の大きさ（ピクセルスケール 0-1）
        device: デバイス
    
    Returns:
        x_adv: 敵対的画像 [B, 3, H, W] in [0, 1]
    """
    x = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    x.requires_grad = True
    
    # Forward pass
    outputs = model(x)
    loss = F.cross_entropy(outputs, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    grad = x.grad.data
    
    # FGSM: x_adv = x + epsilon * sign(grad)
    x_adv = x + epsilon * grad.sign()
    
    # クリッピング [0, 1]
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    return x_adv


# ========== データ読み込み ==========
def load_cached_samples(cached_path):
    """キャッシュされたサンプルを読み込み（ViT分類器で正しく分類されたサンプル）"""
    print(f"\nLoading cached samples from: {cached_path}")
    cached = torch.load(cached_path, map_location='cpu')
    x_test = cached['x_test']
    y_test = cached['y_test']
    classes = cached.get('classes', ['NORMAL', 'PNEUMONIA'])
    print(f"Loaded {len(x_test)} correctly classified samples")
    print(f"  x_test shape: {x_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {classes}")
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_classifier(args, device):
    """ViT分類器を読み込み"""
    # ViT分類器（2クラス: NORMAL, PNEUMONIA）
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
    """モデルの精度を計算"""
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


# ========== 予測取得 ==========
def get_predictions(model, x, bs=32, device=None):
    """モデルの予測を取得"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    preds = []
    
    with torch.no_grad():
        for i in range(n_batches):
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


# ========== FGSM攻撃実行 ==========
def run_fgsm_attack(model, x_test, y_test, epsilon, device, batch_size=32):
    """FGSM攻撃を実行して敵対的サンプルを生成"""
    print(f"\nRunning FGSM attack with epsilon={epsilon:.4f}...")
    
    n_batches = (len(x_test) + batch_size - 1) // batch_size
    x_adv_list = []
    
    for i in tqdm(range(n_batches), desc="FGSM Attack"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x_test))
        x_batch = x_test[start_idx:end_idx].to(device)
        y_batch = y_test[start_idx:end_idx].to(device)
        
        x_adv_batch = fgsm_attack(model, x_batch, y_batch, epsilon, device)
        x_adv_list.append(x_adv_batch.cpu())
    
    x_adv = torch.cat(x_adv_list, dim=0)
    print(f"Generated {len(x_adv)} adversarial samples")
    
    return x_adv


# ========== サンプル画像保存 ==========
def save_sample_images(x_clean, x_adv, x_compressed_clean, x_compressed_adv, 
                       y_true, classes, save_dir, max_samples=10):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(x_clean), max_samples)
    
    for i in range(n):
        label = int(y_true[i])
        label_name = classes[label] if classes else str(label)
        
        # 4枚を並べて保存: Clean, Clean+JPEG, Adv, Adv+JPEG
        quad = torch.cat([
            x_clean[i:i+1],
            x_compressed_clean[i:i+1],
            x_adv[i:i+1],
            x_compressed_adv[i:i+1]
        ], dim=0)
        grid = make_grid(quad, nrow=4, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_{label_name}.png"))
    
    print(f"Saved {n} sample images to {save_dir}")
    print(f"  Format: [Clean | Clean+JPEG | Adversarial | Adv+JPEG]")


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
    log_dir = os.path.join(args.output_dir, f"fgsm_eps{args.epsilon:.4f}_q{args.quality}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output directory: {log_dir}")
    
    # モデル読み込み
    classifier = load_classifier(args, device)
    
    # JPEG防御
    jpeg_defense = JPEGDefense(quality=args.quality)
    
    # ラッパー作成
    classifier_model = ViTClassifierWrapper(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    defense_model = JPEGDefenseWrapper(jpeg_defense, classifier, IMAGENET_MEAN, IMAGENET_STD).to(device).eval()
    
    # データ読み込み
    x_test, y_test, classes = load_cached_samples(args.cached_samples)
    print(f"Classes: {classes}")
    
    # ==================== 評価開始 ====================
    print(f"\n{'='*70}")
    print("FGSM Attack + JPEG Compression Defense Evaluation (ViT Classifier)")
    print(f"{'='*70}")
    print(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"JPEG Quality: {args.quality}")
    print(f"Samples: {len(x_test)}")
    print(f"{'='*70}")
    
    results = {}
    
    # ========== 1. クリーン画像の精度 ==========
    print("\n[1/4] Evaluating clean images (ViT classifier only)...")
    clean_acc = get_accuracy(classifier_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (ViT classifier): {clean_acc:.4f}")
    results['clean_acc_classifier'] = clean_acc
    
    # ========== 2. クリーン画像をJPEG圧縮した後の精度 ==========
    print("\n[2/4] Evaluating clean images with JPEG compression...")
    clean_compressed_acc = get_accuracy(defense_model, x_test, y_test, bs=args.batch_size, device=device)
    print(f"Clean accuracy (with JPEG q={args.quality}): {clean_compressed_acc:.4f}")
    results['clean_acc_with_jpeg'] = clean_compressed_acc
    
    # ========== 3. FGSM攻撃 & 敵対的画像の精度（防御なし） ==========
    print("\n[3/4] Running FGSM attack and evaluating adversarial images...")
    start_time = time.time()
    x_adv = run_fgsm_attack(classifier_model, x_test, y_test, args.epsilon, device, args.batch_size)
    attack_time = time.time() - start_time
    
    adv_acc_no_defense = get_accuracy(classifier_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (no defense): {adv_acc_no_defense:.4f}")
    results['adv_acc_no_defense'] = adv_acc_no_defense
    results['attack_time'] = attack_time
    
    # ========== 4. 敵対的画像をJPEG圧縮した後の精度（防御あり） ==========
    print("\n[4/4] Evaluating adversarial images with JPEG compression...")
    adv_defended_acc = get_accuracy(defense_model, x_adv, y_test, bs=args.batch_size, device=device)
    print(f"Adversarial accuracy (with JPEG q={args.quality}): {adv_defended_acc:.4f}")
    results['adv_acc_with_jpeg'] = adv_defended_acc
    
    # 防御効果
    defense_improvement = adv_defended_acc - adv_acc_no_defense
    results['defense_improvement'] = defense_improvement
    
    # ==================== 最終結果 ====================
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Classifier: ViT-B/16")
    print(f"Attack: FGSM, Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)")
    print(f"JPEG Defense: quality={args.quality}")
    print(f"-"*70)
    print(f"Clean Accuracy:")
    print(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}")
    print(f"  With JPEG compression:    {results['clean_acc_with_jpeg']:.4f}")
    print(f"-"*70)
    print(f"Adversarial Accuracy (FGSM):")
    print(f"  Without defense:          {results['adv_acc_no_defense']:.4f}")
    print(f"  With JPEG compression:    {results['adv_acc_with_jpeg']:.4f}")
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
    pred_clean_compressed = get_predictions(defense_model, x_test, bs=args.batch_size, device=device)
    pred_adv_no_def = get_predictions(classifier_model, x_adv, bs=args.batch_size, device=device)
    pred_adv_defended = get_predictions(defense_model, x_adv, bs=args.batch_size, device=device)
    
    y_true = y_test.cpu().numpy()
    
    cm_clean = print_confusion_matrix(y_true, pred_clean, "1. Clean Images (ViT classifier only)", classes)
    cm_clean_compressed = print_confusion_matrix(y_true, pred_clean_compressed, "2. Clean Images (with JPEG)", classes)
    cm_adv_no_def = print_confusion_matrix(y_true, pred_adv_no_def, "3. Adversarial Images (No Defense)", classes)
    cm_adv_defended = print_confusion_matrix(y_true, pred_adv_defended, "4. Adversarial Images (with JPEG)", classes)
    
    results['confusion_matrices'] = {
        'clean': cm_clean,
        'clean_compressed': cm_clean_compressed,
        'adv_no_defense': cm_adv_no_def,
        'adv_defended': cm_adv_defended
    }
    
    # ==================== 圧縮画像を生成して保存 ====================
    print("\nGenerating compressed samples for visualization...")
    n_samples = min(10, len(x_test))
    x_compressed_clean = jpeg_defense(x_test[:n_samples])
    x_compressed_adv = jpeg_defense(x_adv[:n_samples])
    
    save_sample_images(
        x_test[:n_samples].cpu(), 
        x_adv[:n_samples].cpu(),
        x_compressed_clean,
        x_compressed_adv,
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
    }, os.path.join(log_dir, 'adversarial_samples.pt'))
    print(f"Saved adversarial samples to: {os.path.join(log_dir, 'adversarial_samples.pt')}")
    
    # ==================== サマリー保存 ====================
    summary_path = os.path.join(log_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ChestX-ray - FGSM Attack + JPEG Compression Defense (ViT Classifier)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Classifier: ViT-B/16\n")
        f.write(f"Attack: FGSM\n")
        f.write(f"Epsilon: {args.epsilon:.4f} ({args.epsilon*255:.1f}/255)\n")
        f.write(f"JPEG Defense: quality={args.quality}\n")
        f.write(f"Samples: {len(x_test)}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        f.write("Clean Accuracy:\n")
        f.write(f"  ViT classifier only:      {results['clean_acc_classifier']:.4f}\n")
        f.write(f"  With JPEG compression:    {results['clean_acc_with_jpeg']:.4f}\n\n")
        
        f.write("Adversarial Accuracy (FGSM):\n")
        f.write(f"  Without defense:          {results['adv_acc_no_defense']:.4f}\n")
        f.write(f"  With JPEG compression:    {results['adv_acc_with_jpeg']:.4f}\n")
        f.write(f"  Defense improvement:      {results['defense_improvement']:+.4f}\n\n")
        
        f.write(f"Attack time: {results['attack_time']:.2f}s\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRICES\n")
        f.write("-"*70 + "\n\n")
        
        for name, cm in [("Clean (ViT Classifier)", cm_clean), 
                         ("Clean (with JPEG)", cm_clean_compressed),
                         ("Adversarial (No Defense)", cm_adv_no_def),
                         ("Adversarial (with JPEG)", cm_adv_defended)]:
            if cm:
                f.write(f"{name}:\n")
                f.write(f"  TN: {cm['tn']:4d}  FP: {cm['fp']:4d}\n")
                f.write(f"  FN: {cm['fn']:4d}  TP: {cm['tp']:4d}\n")
                f.write(f"  Accuracy: {cm['accuracy']:.4f}\n")
                f.write(f"  Precision: {cm['precision']:.4f}, Recall: {cm['recall']:.4f}, F1: {cm['f1']:.4f}\n\n")
    
    # JSON形式でも保存
    results_json = {
        'classifier': 'ViT-B/16',
        'args': vars(args),
        'clean_acc_classifier': results['clean_acc_classifier'],
        'clean_acc_with_jpeg': results['clean_acc_with_jpeg'],
        'adv_acc_no_defense': results['adv_acc_no_defense'],
        'adv_acc_with_jpeg': results['adv_acc_with_jpeg'],
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
