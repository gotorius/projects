"""
ViT分類器の敵対的攻撃に対するロバスト性評価
- PCam, ChestX-ray, DermMelデータセット
- Clean, FGSM, PGD, AutoAttack の精度を計測

使用方法:
python vit_attack_evaluation.py --dataset pcam
python vit_attack_evaluation.py --dataset chestxray
python vit_attack_evaluation.py --dataset dermmel
python vit_attack_evaluation.py --dataset all  # 全データセット
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm.auto import tqdm

try:
    from autoattack import AutoAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    print("Warning: autoattack not installed. AutoAttack evaluation will be skipped.")


# ========== 設定 ==========
DATASET_CONFIG = {
    'pcam': {
        'cached_samples': '/mnt/data1/gotou/projects/vit/pcam/correct_samples_balanced_500_vit.pt',
        'clf_ckpt': '/mnt/data1/gotou/projects/vit/classifiers/checkpoints/pcam/20260117_210505/best_vit_pcam.pth',
        'num_classes': 2,
    },
    'chestxray': {
        'cached_samples': '/mnt/data1/gotou/projects/vit/chestxray/correct_samples_balanced_500_vit.pt',
        'clf_ckpt': '/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth',
        'num_classes': 2,
    },
    'dermmel': {
        'cached_samples': '/mnt/data1/gotou/projects/vit/dermmel/vit/correct_samples_balanced_500_vit.pt',
        'clf_ckpt': '/mnt/data1/gotou/projects/vit/classifiers/checkpoints/dermmel/20260118_175806/best_vit_dermmel.pth',
        'num_classes': 2,
    },
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='ViT Adversarial Robustness Evaluation')
    
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['pcam', 'chestxray', 'dermmel', 'all'],
                        help='Dataset to evaluate')
    
    # 攻撃設定
    parser.add_argument('--epsilon', type=float, default=8/255,
                        help='Perturbation epsilon (Linf)')
    parser.add_argument('--pgd_alpha', type=float, default=2/255,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=20,
                        help='PGD attack steps')
    parser.add_argument('--autoattack_version', type=str, default='standard',
                        choices=['standard', 'plus', 'rand'],
                        help='AutoAttack version')
    
    # 実行設定
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/attack_evaluation_results',
                        help='Output directory')
    parser.add_argument('--skip_autoattack', action='store_true',
                        help='Skip AutoAttack (takes long time)')
    
    return parser.parse_args()


# ========== ViTモデル構築 ==========
def get_vit_model(model_name='vit_b_16', num_classes=2, dropout=0.1):
    """Vision Transformer モデルの構築"""
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    
    return model


# ========== 正規化付き分類器 ==========
class NormalizedClassifier(nn.Module):
    """ImageNet正規化を含む分類器（AutoAttack用）"""
    def __init__(self, classifier, mean, std):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        x_norm = (x - self.mean) / self.std
        return self.classifier(x_norm)


# ========== データ読み込み ==========
def load_cached_samples(path):
    """キャッシュされたサンプルを読み込み"""
    data = torch.load(path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    classes = data['classes']
    return x_test, y_test, classes


# ========== モデル読み込み ==========
def load_classifier(ckpt_path, num_classes, device):
    """ViT分類器を読み込み"""
    classifier = get_vit_model(model_name='vit_b_16', num_classes=num_classes, dropout=0.1)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    return classifier


# ========== 攻撃関数 ==========
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


def pgd_attack(model, x, y, epsilon, alpha, steps, device):
    """PGD攻撃"""
    x = x.clone().to(device)
    x_adv = x.clone() + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x_adv, 0, 1)
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    for _ in range(steps):
        x_adv.requires_grad = True
        
        x_norm = (x_adv - mean) / std
        outputs = model(x_norm)
        loss = F.cross_entropy(outputs, y.to(device))
        
        loss.backward()
        
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
        
        x_adv = x_adv.detach()
    
    return x_adv


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


def evaluate_single_dataset(dataset_name, config, args, device, results_file):
    """単一データセットの評価"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {dataset_name.upper()}")
    print(f"{'='*70}")
    results_file.write(f"\n{'='*70}\n")
    results_file.write(f"Dataset: {dataset_name.upper()}\n")
    results_file.write(f"{'='*70}\n")
    
    # データ読み込み
    try:
        x_test, y_test, classes = load_cached_samples(config['cached_samples'])
        print(f"Loaded {len(x_test)} samples, Classes: {classes}")
        results_file.write(f"Samples: {len(x_test)}, Classes: {classes}\n")
    except Exception as e:
        print(f"Failed to load data: {e}")
        results_file.write(f"Failed to load data: {e}\n")
        return None
    
    # モデル読み込み
    try:
        classifier = load_classifier(config['clf_ckpt'], config['num_classes'], device)
        print(f"Loaded classifier from {config['clf_ckpt']}")
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        results_file.write(f"Failed to load classifier: {e}\n")
        return None
    
    results = {'dataset': dataset_name, 'classes': classes}
    
    # 1. クリーン画像の評価
    print("\n[1/4] Evaluating Clean Images...")
    clean_acc, pred_clean = evaluate(classifier, x_test, y_test, device, args.batch_size)
    print(f"Clean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)")
    results['clean_acc'] = clean_acc
    results_file.write(f"\nClean Accuracy: {clean_acc:.4f} ({clean_acc*100:.2f}%)\n")
    
    # 2. FGSM攻撃
    print("\n[2/4] Evaluating FGSM Attack...")
    x_adv_fgsm_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="FGSM"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = fgsm_attack(classifier, x_batch, y_batch, args.epsilon, device)
        x_adv_fgsm_list.append(x_adv_batch.cpu())
    x_adv_fgsm = torch.cat(x_adv_fgsm_list, dim=0)
    
    fgsm_acc, pred_fgsm = evaluate(classifier, x_adv_fgsm, y_test, device, args.batch_size)
    print(f"FGSM Accuracy: {fgsm_acc:.4f} ({fgsm_acc*100:.2f}%)")
    results['fgsm_acc'] = fgsm_acc
    results_file.write(f"FGSM Accuracy (eps={args.epsilon:.4f}): {fgsm_acc:.4f} ({fgsm_acc*100:.2f}%)\n")
    
    # 3. PGD攻撃
    print("\n[3/4] Evaluating PGD Attack...")
    x_adv_pgd_list = []
    for i in tqdm(range(0, len(x_test), args.batch_size), desc="PGD"):
        x_batch = x_test[i:i+args.batch_size]
        y_batch = y_test[i:i+args.batch_size]
        x_adv_batch = pgd_attack(classifier, x_batch, y_batch, args.epsilon, args.pgd_alpha, args.pgd_steps, device)
        x_adv_pgd_list.append(x_adv_batch.cpu())
    x_adv_pgd = torch.cat(x_adv_pgd_list, dim=0)
    
    pgd_acc, pred_pgd = evaluate(classifier, x_adv_pgd, y_test, device, args.batch_size)
    print(f"PGD Accuracy: {pgd_acc:.4f} ({pgd_acc*100:.2f}%)")
    results['pgd_acc'] = pgd_acc
    results_file.write(f"PGD Accuracy (eps={args.epsilon:.4f}, alpha={args.pgd_alpha:.4f}, steps={args.pgd_steps}): {pgd_acc:.4f} ({pgd_acc*100:.2f}%)\n")
    
    # 4. AutoAttack
    if not args.skip_autoattack and AUTOATTACK_AVAILABLE:
        print("\n[4/4] Evaluating AutoAttack...")
        model_normalized = NormalizedClassifier(classifier, IMAGENET_MEAN, IMAGENET_STD).to(device)
        
        adversary = AutoAttack(model_normalized, norm='Linf', eps=args.epsilon, 
                               version=args.autoattack_version, device=device, verbose=False)
        
        try:
            x_adv_auto = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), bs=args.batch_size)
            x_adv_auto = x_adv_auto.cpu()
            
            auto_acc, pred_auto = evaluate(classifier, x_adv_auto, y_test, device, args.batch_size)
            print(f"AutoAttack Accuracy: {auto_acc:.4f} ({auto_acc*100:.2f}%)")
            results['autoattack_acc'] = auto_acc
            results_file.write(f"AutoAttack Accuracy ({args.autoattack_version}): {auto_acc:.4f} ({auto_acc*100:.2f}%)\n")
        except Exception as e:
            print(f"AutoAttack failed: {e}")
            results['autoattack_acc'] = None
            results_file.write(f"AutoAttack failed: {e}\n")
    else:
        print("\n[4/4] Skipping AutoAttack...")
        results['autoattack_acc'] = None
        results_file.write("AutoAttack: Skipped\n")
    
    # サマリー
    print(f"\n--- {dataset_name.upper()} Summary ---")
    print(f"Clean:      {results['clean_acc']:.4f} ({results['clean_acc']*100:.2f}%)")
    print(f"FGSM:       {results['fgsm_acc']:.4f} ({results['fgsm_acc']*100:.2f}%)")
    print(f"PGD:        {results['pgd_acc']:.4f} ({results['pgd_acc']*100:.2f}%)")
    if results['autoattack_acc'] is not None:
        print(f"AutoAttack: {results['autoattack_acc']:.4f} ({results['autoattack_acc']*100:.2f}%)")
    
    return results


def main():
    args = parse_args()
    
    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 結果ファイル
    results_txt_path = os.path.join(args.output_dir, f'vit_attack_results_{timestamp}.txt')
    results_json_path = os.path.join(args.output_dir, f'vit_attack_results_{timestamp}.json')
    
    # 評価対象データセット
    if args.dataset == 'all':
        datasets_to_eval = ['pcam', 'chestxray', 'dermmel']
    else:
        datasets_to_eval = [args.dataset]
    
    all_results = {}
    
    with open(results_txt_path, 'w') as f:
        f.write(f"ViT Adversarial Robustness Evaluation\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Epsilon: {args.epsilon:.4f}\n")
        f.write(f"PGD Alpha: {args.pgd_alpha:.4f}, Steps: {args.pgd_steps}\n")
        f.write(f"AutoAttack Version: {args.autoattack_version}\n")
        
        for dataset_name in datasets_to_eval:
            config = DATASET_CONFIG[dataset_name]
            result = evaluate_single_dataset(dataset_name, config, args, device, f)
            if result:
                all_results[dataset_name] = result
        
        # 全体サマリー
        f.write(f"\n\n{'='*70}\n")
        f.write("OVERALL SUMMARY\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Dataset':<15} {'Clean':>10} {'FGSM':>10} {'PGD':>10} {'AutoAttack':>12}\n")
        f.write("-" * 60 + "\n")
        
        for ds, res in all_results.items():
            auto_str = f"{res['autoattack_acc']:.4f}" if res['autoattack_acc'] else "N/A"
            f.write(f"{ds:<15} {res['clean_acc']:>10.4f} {res['fgsm_acc']:>10.4f} {res['pgd_acc']:>10.4f} {auto_str:>12}\n")
    
    # JSON保存
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # コンソールにも全体サマリー表示
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Clean':>10} {'FGSM':>10} {'PGD':>10} {'AutoAttack':>12}")
    print("-" * 60)
    
    for ds, res in all_results.items():
        auto_str = f"{res['autoattack_acc']:.4f}" if res['autoattack_acc'] else "N/A"
        print(f"{ds:<15} {res['clean_acc']:>10.4f} {res['fgsm_acc']:>10.4f} {res['pgd_acc']:>10.4f} {auto_str:>12}")
    
    print(f"\nResults saved to:")
    print(f"  - {results_txt_path}")
    print(f"  - {results_json_path}")


if __name__ == '__main__':
    main()
