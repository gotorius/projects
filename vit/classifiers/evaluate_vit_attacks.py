"""
ChestX-ray Dataset - ViT分類器の攻撃耐性評価
==================================================

評価内容:
1. クリーン画像でのテスト精度
2. FGSM敵対的画像での分類精度
3. PGD敵対的画像での分類精度
4. AutoAttack敵対的画像での分類精度

使用法:
    python evaluate_vit_attacks.py --gpu 0
    python evaluate_vit_attacks.py --batch_size 16 --gpu 1
    python evaluate_vit_attacks.py --epsilon 0.03137 --gpu 0
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
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms, datasets
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# AutoAttackのインポート
try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    print("Warning: AutoAttack not installed. AutoAttack evaluation will be skipped.")
    print("Install with: pip install git+https://github.com/fra31/auto-attack")
    AUTOATTACK_AVAILABLE = False


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ChestX-ray ViT Adversarial Attack Evaluation')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Perturbation epsilon (pixel scale 0-1)')
    parser.add_argument('--alpha', type=float, default=2/255,
                        help='PGD step size (pixel scale 0-1)')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD iterations')
    parser.add_argument('--random_start', action='store_true', default=True,
                        help='Use random start for PGD')
    
    # AutoAttack設定
    parser.add_argument('--aa_version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    parser.add_argument('--aa_norm', type=str, default='Linf',
                        choices=['Linf', 'L2'],
                        help='AutoAttack norm')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # パス設定
    parser.add_argument('--data_dir', type=str,
                        default='/mnt/data1/Public/MedImages/CellData/chest_xray/test',
                        help='Test data directory')
    parser.add_argument('--clf_ckpt', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
                        help='ViT Classifier checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/attack_eval_results',
                        help='Output directory')
    
    # GPU設定
    parser.add_argument('--gpu', type=int, default=1,
                        help='GPU ID to use')
    
    # スキップ設定
    parser.add_argument('--skip_clean', action='store_true',
                        help='Skip clean accuracy evaluation')
    parser.add_argument('--skip_fgsm', action='store_true',
                        help='Skip FGSM attack evaluation')
    parser.add_argument('--skip_pgd', action='store_true',
                        help='Skip PGD attack evaluation')
    parser.add_argument('--skip_autoattack', action='store_true',
                        help='Skip AutoAttack evaluation')
    
    return parser.parse_args()


# ========== 定数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']


# ========== ユーティリティ ==========
def set_seed(seed):
    """乱数シードの設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== データ変換 ==========
def get_test_transform(img_size=224):
    """テスト用変換（正規化なし、[0,1]範囲）"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0, 1]
    ])


# ========== ViT分類器ラッパー ==========
class ViTClassifierWrapper(nn.Module):
    """ViT分類器のラッパー
    入力: [0,1]のRGB画像
    出力: 2クラスロジット
    """
    def __init__(self, classifier, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        """x: [0,1]の画像 → 正規化 → 2クラスロジット"""
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== モデル読み込み ==========
def load_vit_classifier(ckpt_path, device):
    """ViT分類器を読み込み"""
    print(f"\nLoading ViT classifier from: {ckpt_path}")
    
    # ViT-B/16（2クラス: NORMAL, PNEUMONIA）
    classifier = models.vit_b_16(weights=None)
    in_features = classifier.heads.head.in_features
    classifier.heads.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(in_features, 2)
    )
    
    # チェックポイント読み込み
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        classifier.load_state_dict(ckpt['model_state_dict'])
        if 'best_val_acc' in ckpt:
            print(f"  Best validation accuracy: {ckpt['best_val_acc']:.4f}")
    else:
        classifier.load_state_dict(ckpt)
    
    classifier = classifier.to(device).eval()
    print("  Model loaded successfully.")
    
    return classifier


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


# ========== PGD攻撃 ==========
def pgd_attack(model, x, y, epsilon, alpha, steps, device, random_start=True):
    """
    PGD攻撃 (L_inf)
    
    Args:
        model: 分類器（入力は[0,1]のRGB画像）
        x: 入力画像 [B, 3, H, W] in [0, 1]
        y: ラベル [B]
        epsilon: 摂動の最大値（ピクセルスケール 0-1）
        alpha: ステップサイズ（ピクセルスケール 0-1）
        steps: 反復回数
        device: デバイス
        random_start: ランダム初期化
    
    Returns:
        x_adv: 敵対的画像 [B, 3, H, W] in [0, 1]
    """
    x_orig = x.clone().detach().to(device)
    y = y.clone().detach().to(device)
    
    # ランダム初期化
    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_orig.clone()
    
    # PGD反復
    for _ in range(steps):
        x_adv.requires_grad = True
        
        outputs = model(x_adv)
        loss = F.cross_entropy(outputs, y)
        
        model.zero_grad()
        loss.backward()
        grad = x_adv.grad.data
        
        # ステップ更新
        x_adv = x_adv + alpha * grad.sign()
        
        # L_inf ボールへの射影
        eta = torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x_orig + eta, 0.0, 1.0).detach()
    
    return x_adv


# ========== 精度計算 ==========
def evaluate_accuracy(model, dataloader, device, desc="Evaluating"):
    """クリーン画像での精度を評価"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_fgsm(model, dataloader, epsilon, device, desc="FGSM Attack"):
    """FGSM攻撃後の精度を評価"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)
        
        # FGSM攻撃
        adv_images = fgsm_attack(model, images, labels, epsilon, device)
        
        # 予測
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_pgd(model, dataloader, epsilon, alpha, steps, device, random_start=True, desc="PGD Attack"):
    """PGD攻撃後の精度を評価"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(dataloader, desc=desc):
        images = images.to(device)
        labels = labels.to(device)
        
        # PGD攻撃
        adv_images = pgd_attack(model, images, labels, epsilon, alpha, steps, device, random_start)
        
        # 予測
        with torch.no_grad():
            outputs = model(adv_images)
            preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_autoattack(model, dataloader, epsilon, version, norm, device, desc="AutoAttack"):
    """AutoAttack後の精度を評価"""
    if not AUTOATTACK_AVAILABLE:
        print("AutoAttack is not available. Skipping...")
        return None, None, None
    
    model.eval()
    
    # 全データを収集
    all_images = []
    all_labels = []
    
    print(f"\n{desc}: Collecting all test data...")
    for images, labels in tqdm(dataloader, desc="Loading data"):
        all_images.append(images)
        all_labels.append(labels)
    
    x_test = torch.cat(all_images, dim=0).to(device)
    y_test = torch.cat(all_labels, dim=0).to(device)
    
    print(f"  Total samples: {len(x_test)}")
    print(f"  Running AutoAttack (version={version}, norm={norm}, epsilon={epsilon:.5f})...")
    
    # AutoAttack実行
    adversary = AutoAttack(model, norm=norm, eps=epsilon, version=version, verbose=True)
    
    # バッチサイズを調整（メモリ節約）
    adversary.apgd.n_restarts = 1
    
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=32)
    
    # 精度計算
    with torch.no_grad():
        outputs = model(x_adv)
        preds = outputs.argmax(dim=1)
    
    correct = (preds == y_test).sum().item()
    accuracy = correct / len(y_test)
    
    return accuracy, preds.cpu().numpy(), y_test.cpu().numpy()


# ========== 結果出力 ==========
def print_confusion_matrix(y_true, y_pred, title, classes=CLASS_NAMES):
    """混同行列を表示"""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{title}:")
    print(f"{'':>10} {'Pred ' + classes[0]:>15} {'Pred ' + classes[1]:>15}")
    print(f"{'True ' + classes[0]:>10} {cm[0][0]:>15d} {cm[0][1]:>15d}")
    print(f"{'True ' + classes[1]:>10} {cm[1][0]:>15d} {cm[1][1]:>15d}")
    
    # メトリクス計算
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        
        print(f"\n  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        return {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1
        }
    
    return {}


def save_results(results, output_dir, filename='attack_eval_results.json'):
    """結果をJSONで保存"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filepath}")


# ========== メイン ==========
def main():
    args = parse_args()
    
    # 乱数シード設定
    set_seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # データセット読み込み
    print(f"\nLoading test dataset from: {args.data_dir}")
    test_transform = get_test_transform()
    test_dataset = datasets.ImageFolder(root=args.data_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Total test samples: {len(test_dataset)}")
    print(f"  Classes: {test_dataset.classes}")
    
    # クラス分布
    class_counts = {}
    for _, label in test_dataset.samples:
        class_name = test_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    print(f"  Class distribution: {class_counts}")
    
    # モデル読み込み
    classifier = load_vit_classifier(args.clf_ckpt, device)
    model = ViTClassifierWrapper(classifier).to(device)
    model.eval()
    
    # 結果格納
    results = {
        'timestamp': timestamp,
        'config': {
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'pgd_steps': args.pgd_steps,
            'random_start': args.random_start,
            'aa_version': args.aa_version,
            'aa_norm': args.aa_norm,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'data_dir': args.data_dir,
            'clf_ckpt': args.clf_ckpt,
        },
        'dataset': {
            'total_samples': len(test_dataset),
            'classes': test_dataset.classes,
            'class_distribution': class_counts,
        },
        'evaluations': {}
    }
    
    print("\n" + "="*70)
    print("Starting Adversarial Attack Evaluation")
    print("="*70)
    print(f"  Epsilon: {args.epsilon:.5f} ({args.epsilon*255:.2f}/255)")
    print(f"  PGD: alpha={args.alpha:.5f}, steps={args.pgd_steps}")
    print(f"  AutoAttack: version={args.aa_version}, norm={args.aa_norm}")
    print("="*70)
    
    # 1. クリーン画像での評価
    if not args.skip_clean:
        print("\n[1/4] Evaluating Clean Accuracy...")
        start_time = time.time()
        clean_acc, clean_preds, clean_labels = evaluate_accuracy(
            model, test_loader, device, desc="Clean Images"
        )
        clean_time = time.time() - start_time
        
        print(f"\nClean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
        clean_metrics = print_confusion_matrix(clean_labels, clean_preds, "Clean Confusion Matrix")
        
        results['evaluations']['clean'] = {
            'accuracy': clean_acc,
            'time_seconds': clean_time,
            **clean_metrics
        }
    
    # 2. FGSM攻撃での評価
    if not args.skip_fgsm:
        print("\n[2/4] Evaluating FGSM Attack...")
        start_time = time.time()
        fgsm_acc, fgsm_preds, fgsm_labels = evaluate_fgsm(
            model, test_loader, args.epsilon, device, desc="FGSM Attack"
        )
        fgsm_time = time.time() - start_time
        
        print(f"\nFGSM Accuracy: {fgsm_acc:.4f} ({fgsm_acc*100:.2f}%)")
        fgsm_metrics = print_confusion_matrix(fgsm_labels, fgsm_preds, "FGSM Confusion Matrix")
        
        results['evaluations']['fgsm'] = {
            'accuracy': fgsm_acc,
            'epsilon': args.epsilon,
            'time_seconds': fgsm_time,
            **fgsm_metrics
        }
    
    # 3. PGD攻撃での評価
    if not args.skip_pgd:
        print("\n[3/4] Evaluating PGD Attack...")
        start_time = time.time()
        pgd_acc, pgd_preds, pgd_labels = evaluate_pgd(
            model, test_loader, args.epsilon, args.alpha, args.pgd_steps,
            device, args.random_start, desc="PGD Attack"
        )
        pgd_time = time.time() - start_time
        
        print(f"\nPGD Accuracy: {pgd_acc:.4f} ({pgd_acc*100:.2f}%)")
        pgd_metrics = print_confusion_matrix(pgd_labels, pgd_preds, "PGD Confusion Matrix")
        
        results['evaluations']['pgd'] = {
            'accuracy': pgd_acc,
            'epsilon': args.epsilon,
            'alpha': args.alpha,
            'steps': args.pgd_steps,
            'random_start': args.random_start,
            'time_seconds': pgd_time,
            **pgd_metrics
        }
    
    # 4. AutoAttack での評価
    if not args.skip_autoattack and AUTOATTACK_AVAILABLE:
        print("\n[4/4] Evaluating AutoAttack...")
        start_time = time.time()
        aa_acc, aa_preds, aa_labels = evaluate_autoattack(
            model, test_loader, args.epsilon, args.aa_version, args.aa_norm,
            device, desc="AutoAttack"
        )
        aa_time = time.time() - start_time
        
        if aa_acc is not None:
            print(f"\nAutoAttack Accuracy: {aa_acc:.4f} ({aa_acc*100:.2f}%)")
            aa_metrics = print_confusion_matrix(aa_labels, aa_preds, "AutoAttack Confusion Matrix")
            
            results['evaluations']['autoattack'] = {
                'accuracy': aa_acc,
                'epsilon': args.epsilon,
                'version': args.aa_version,
                'norm': args.aa_norm,
                'time_seconds': aa_time,
                **aa_metrics
            }
    
    # サマリー表示
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Attack':<20} {'Accuracy':>15} {'Change':>15}")
    print("-"*50)
    
    baseline = results['evaluations'].get('clean', {}).get('accuracy', 0)
    for attack_name, eval_data in results['evaluations'].items():
        acc = eval_data['accuracy']
        change = acc - baseline
        change_str = f"{change:+.4f}" if attack_name != 'clean' else "-"
        print(f"{attack_name:<20} {acc:>14.4f} {change_str:>15}")
    
    print("="*70)
    
    # 結果保存
    save_results(results, output_dir)
    
    # 設定ファイルも保存
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
