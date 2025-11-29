"""
PCam Dataset - AutoAttack + JPEG Compression Defense
JPEG圧縮によるAutoAttack敵対的攻撃からの防御検証スクリプト

DiffPureスタイルの実装：
- 4種類の攻撃に対応: apgd-ce, apgd-t, fab-t, square
- attack_version: 'standard', 'rand', 'custom' をサポート
"""

import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import io
import time
import random
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import seaborn as sns

from autoattack import AutoAttack


# ========== 引数パーサー ==========
def parse_args():
    parser = argparse.ArgumentParser(description='PCam AutoAttack + JPEG Defense')
    # 攻撃設定
    parser.add_argument('--attack_version', type=str, default='standard',
                        choices=['standard', 'rand', 'custom'],
                        help='Attack version: standard, rand, or custom')
    parser.add_argument('--attack_type', type=str, default='apgd-ce',
                        help='Attack type for custom version (comma-separated, e.g., apgd-ce,square)')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'],
                        help='Lp norm for attack')
    parser.add_argument('--adv_eps', type=float, default=8/255,
                        help='Adversarial perturbation epsilon')
    parser.add_argument('--eot_iter', type=int, default=20,
                        help='EOT iterations for rand version')
    
    # JPEG防御設定
    parser.add_argument('--jpeg_quality', type=int, nargs='+', default=[11],
                        help='JPEG quality levels to test')
    
    # 実行設定
    parser.add_argument('--adv_batch_size', type=int, default=64,
                        help='Batch size for adversarial attack')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--data_seed', type=int, default=0,
                        help='Data random seed')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to evaluate (0 for all)')
    
    # GPU設定
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                        help='GPU IDs to use (comma-separated)')
    
    return parser.parse_args()


# ========== 設定 ==========
DATA_DIR = '/mnt/data1/gotou/projects/data'
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train')
LABELS_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
CLF_CKPT = os.path.join(DATA_DIR, 'best_model_weights.pth')

# ImageNet正規化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ========== データセット定義 ==========
class PCamDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels = labels_df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_id = self.labels.iloc[idx, 0]
        label = self.labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# ========== AutoAttack用のモデルラッパー ==========
class NormalizedModel(nn.Module):
    """分類器単体のラッパー（正規化を内部で行う）"""
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        # x は [0,1] の範囲の画像
        x_norm = (x - self.mean) / self.std
        logits = self.model(x_norm)
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        # 2クラス分類のためにロジットを2次元に変換
        return torch.stack([-logits, logits], dim=1)


class JPEGDefenseModel(nn.Module):
    """JPEG圧縮防御を組み込んだモデル"""
    def __init__(self, model, mean, std, jpeg_quality=75):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.jpeg_quality = jpeg_quality
    
    def jpeg_compress_batch(self, images_pixel):
        """バッチ画像にJPEG圧縮を適用（ピクセル空間[0,1]）"""
        batch_size = images_pixel.size(0)
        compressed_images = []
        
        for i in range(batch_size):
            img_np = images_pixel[i].detach().cpu().numpy().transpose(1, 2, 0)
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=self.jpeg_quality)
            buffer.seek(0)
            compressed_pil = Image.open(buffer).convert('RGB')
            
            compressed_np = np.array(compressed_pil).astype(np.float32) / 255.0
            compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
            compressed_images.append(compressed_tensor)
        
        return torch.stack(compressed_images).to(images_pixel.device)
    
    def forward(self, x):
        # x は [0,1] の範囲の画像
        # JPEG圧縮を適用
        x_compressed = self.jpeg_compress_batch(x)
        # 正規化して分類
        x_norm = (x_compressed - self.mean) / self.std
        logits = self.model(x_norm)
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return torch.stack([-logits, logits], dim=1)


# ========== データ読み込み関数 ==========
def load_data(args, batch_size):
    """検証データをロード"""
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 正規化は行わない（[0,1]のまま）
    ])
    
    labels_df = pd.read_csv(LABELS_CSV)
    _, val_df = train_test_split(labels_df, test_size=0.1, 
                                  random_state=42, stratify=labels_df['label'])
    
    val_dataset = PCamDataset(TRAIN_IMG_DIR, val_df, val_transform)
    
    # サンプル数制限
    if args.num_samples > 0 and args.num_samples < len(val_dataset):
        # ランダムサンプリング
        np.random.seed(args.data_seed)
        indices = np.random.choice(len(val_dataset), args.num_samples, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # 全データをテンソルに変換
    x_list, y_list = [], []
    for images, labels in tqdm(val_loader, desc="Loading data"):
        x_list.append(images)
        y_list.append(labels)
    
    x_val = torch.cat(x_list, dim=0)
    y_val = torch.cat(y_list, dim=0)
    
    print(f"Loaded {len(x_val)} samples")
    return x_val, y_val


# ========== 分類器読み込み関数 ==========
def get_classifier(device):
    """ResNet50分類器を読み込み"""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    state_dict = torch.load(CLF_CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


# ========== 精度計算関数 ==========
def get_accuracy(model, x, y, bs=100, device=None):
    """モデルの精度を計算"""
    if device is None:
        device = next(model.parameters()).device
    
    n_batches = (len(x) + bs - 1) // bs
    correct = 0
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * bs
            end = min(start + bs, len(x))
            x_batch = x[start:end].to(device)
            y_batch = y[start:end].to(device)
            
            out = model(x_batch)
            pred = out.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
    
    return correct / len(x)


# ========== AutoAttack評価関数（DiffPureスタイル） ==========
def eval_autoattack(args, config, classifier_model, jpeg_model, x_val, y_val, 
                    adv_batch_size, log_dir, device):
    """
    DiffPureスタイルのAutoAttack評価
    """
    attack_version = args.attack_version
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')
    print(f'{args.lp_norm}, epsilon: {args.adv_eps}')
    
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    
    # ---------------- 分類器単体への攻撃 ----------------
    print(f'\n{"="*70}')
    print(f'Applying AutoAttack to classifier [{args.lp_norm}]...')
    print(f'{"="*70}')
    
    # 初期精度
    start_time = time.time()
    init_acc = get_accuracy(classifier_model, x_val, y_val, bs=adv_batch_size, device=device)
    print(f'Initial accuracy: {init_acc:.2%}, time: {time.time() - start_time:.2f}s')
    
    # AutoAttackの仕様: attacks_to_runはversion='custom'の場合のみ使用可能
    if attack_version == 'custom':
        adversary_classifier = AutoAttack(
            classifier_model, 
            norm=args.lp_norm, 
            eps=args.adv_eps,
            version='custom', 
            attacks_to_run=attack_list,
            log_path=f'{log_dir}/log_classifier.txt', 
            device=device
        )
        adversary_classifier.apgd.n_restarts = 1
        adversary_classifier.fab.n_restarts = 1
        adversary_classifier.apgd_targeted.n_restarts = 1
        adversary_classifier.fab.n_target_classes = 1
        adversary_classifier.apgd_targeted.n_target_classes = 1
        adversary_classifier.square.n_queries = 5000
    else:
        adversary_classifier = AutoAttack(
            classifier_model, 
            norm=args.lp_norm, 
            eps=args.adv_eps,
            version=attack_version,
            log_path=f'{log_dir}/log_classifier.txt', 
            device=device
        )
    
    if attack_version == 'rand':
        adversary_classifier.apgd.eot_iter = args.eot_iter
        print(f'[classifier] rand version with eot_iter: {adversary_classifier.apgd.eot_iter}')
    
    start_time = time.time()
    x_adv_classifier = adversary_classifier.run_standard_evaluation(
        x_val, y_val, bs=adv_batch_size
    )
    classifier_time = time.time() - start_time
    
    # 敵対的精度
    robust_acc_classifier = get_accuracy(classifier_model, x_adv_classifier, y_val, 
                                         bs=adv_batch_size, device=device)
    print(f'Robust accuracy (classifier): {robust_acc_classifier:.2%}, time: {classifier_time:.2f}s')
    
    # 保存
    torch.save([x_adv_classifier.cpu(), y_val.cpu()], 
               f'{log_dir}/x_adv_classifier_sd{args.seed}.pt')
    
    # ---------------- JPEG防御モデルへの攻撃 ----------------
    print(f'\n{"="*70}')
    print(f'Applying AutoAttack to JPEG defense model [{args.lp_norm}]...')
    print(f'{"="*70}')
    
    # 初期精度
    start_time = time.time()
    init_acc_jpeg = get_accuracy(jpeg_model, x_val, y_val, bs=adv_batch_size, device=device)
    print(f'Initial accuracy (JPEG): {init_acc_jpeg:.2%}, time: {time.time() - start_time:.2f}s')
    
    # AutoAttackの仕様: attacks_to_runはversion='custom'の場合のみ使用可能
    if attack_version == 'custom':
        adversary_jpeg = AutoAttack(
            jpeg_model, 
            norm=args.lp_norm, 
            eps=args.adv_eps,
            version='custom', 
            attacks_to_run=attack_list,
            log_path=f'{log_dir}/log_jpeg_defense.txt', 
            device=device
        )
        adversary_jpeg.apgd.n_restarts = 1
        adversary_jpeg.fab.n_restarts = 1
        adversary_jpeg.apgd_targeted.n_restarts = 1
        adversary_jpeg.fab.n_target_classes = 1
        adversary_jpeg.apgd_targeted.n_target_classes = 1
        adversary_jpeg.square.n_queries = 5000
    else:
        adversary_jpeg = AutoAttack(
            jpeg_model, 
            norm=args.lp_norm, 
            eps=args.adv_eps,
            version=attack_version,
            log_path=f'{log_dir}/log_jpeg_defense.txt', 
            device=device
        )
    
    if attack_version == 'rand':
        adversary_jpeg.apgd.eot_iter = args.eot_iter
        print(f'[jpeg_defense] rand version with eot_iter: {adversary_jpeg.apgd.eot_iter}')
    
    start_time = time.time()
    x_adv_jpeg = adversary_jpeg.run_standard_evaluation(
        x_val, y_val, bs=adv_batch_size
    )
    jpeg_time = time.time() - start_time
    
    # 敵対的精度
    robust_acc_jpeg = get_accuracy(jpeg_model, x_adv_jpeg, y_val, 
                                   bs=adv_batch_size, device=device)
    print(f'Robust accuracy (JPEG defense): {robust_acc_jpeg:.2%}, time: {jpeg_time:.2f}s')
    
    # 保存
    torch.save([x_adv_jpeg.cpu(), y_val.cpu()], 
               f'{log_dir}/x_adv_jpeg_sd{args.seed}.pt')
    
    # ---------------- 分類器への攻撃サンプルにJPEG防御を適用 ----------------
    print(f'\n{"="*70}')
    print(f'Applying JPEG defense to classifier adversarial samples...')
    print(f'{"="*70}')
    
    defended_acc = get_accuracy(jpeg_model, x_adv_classifier, y_val, 
                                bs=adv_batch_size, device=device)
    print(f'Defended accuracy (JPEG on classifier adv): {defended_acc:.2%}')
    
    return {
        'init_acc_classifier': init_acc,
        'robust_acc_classifier': robust_acc_classifier,
        'classifier_time': classifier_time,
        'init_acc_jpeg': init_acc_jpeg,
        'robust_acc_jpeg': robust_acc_jpeg,
        'jpeg_time': jpeg_time,
        'defended_acc': defended_acc,
        'defense_improvement': defended_acc - robust_acc_classifier,
        'x_adv_classifier': x_adv_classifier,
        'x_adv_jpeg': x_adv_jpeg,
    }


# ========== 混同行列プロット関数 ==========
def plot_confusion_matrix(y_true, y_pred, title, filename):
    """混同行列をプロットして保存"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall)>0 else 0.0
        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                'precision': precision, 'recall': recall, 'f1': f1}
    return {}


# ========== サンプル画像保存関数 ==========
def save_sample_images(x_clean, x_adv, y_true, save_dir, max_samples=20):
    """サンプル画像を保存"""
    os.makedirs(save_dir, exist_ok=True)
    n_samples = min(len(x_clean), max_samples)
    
    for i in range(n_samples):
        # Clean
        save_image(x_clean[i], os.path.join(save_dir, f"{i:04d}_label{y_true[i]}_clean.png"))
        # Adversarial
        save_image(x_adv[i], os.path.join(save_dir, f"{i:04d}_label{y_true[i]}_adv.png"))
        # Triplet
        triplet = torch.stack([x_clean[i], x_adv[i]], dim=0)
        grid = make_grid(triplet, nrow=2, padding=5, pad_value=1.0)
        save_image(grid, os.path.join(save_dir, f"{i:04d}_label{y_true[i]}_pair.png"))


# ========== メイン評価関数 ==========
def robustness_eval(args):
    """DiffPureスタイルのロバスト性評価"""
    
    # GPU設定
    gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"GPU IDs: {gpu_ids}")
    
    # ログディレクトリ設定
    middle_name = '_'.join([args.attack_version]) if args.attack_version in ['standard', 'rand'] \
        else '_'.join([args.attack_version, args.attack_type])
    
    all_results = {}
    
    for jpeg_quality in args.jpeg_quality:
        print(f"\n{'='*70}")
        print(f"Testing JPEG Quality: {jpeg_quality}")
        print(f"{'='*70}")
        
        log_dir = os.path.join(
            '/mnt/data1/gotou/projects/pcam/jpeg/autoattack/defense_results',
            f'quality_{jpeg_quality}', middle_name,
            f'seed{args.seed}', f'data{args.data_seed}'
        )
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイル
        log_file = open(os.path.join(log_dir, 'log.txt'), 'w')
        def log_print(msg):
            print(msg)
            log_file.write(msg + '\n')
            log_file.flush()
        
        log_print(f"="*70)
        log_print(f"PCam - AutoAttack + JPEG Compression Defense")
        log_print(f"="*70)
        log_print(f"Attack version: {args.attack_version}")
        log_print(f"Lp norm: {args.lp_norm}")
        log_print(f"Epsilon: {args.adv_eps}")
        log_print(f"JPEG quality: {jpeg_quality}")
        log_print(f"="*70)
        
        # バッチサイズ設定
        ngpus = len(gpu_ids) if torch.cuda.is_available() else 1
        adv_batch_size = args.adv_batch_size * ngpus
        log_print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')
        
        # 分類器読み込み
        log_print('\nLoading classifier...')
        classifier = get_classifier(device)
        
        # モデル作成
        classifier_model = NormalizedModel(classifier, IMAGENET_MEAN, IMAGENET_STD)
        jpeg_model = JPEGDefenseModel(classifier, IMAGENET_MEAN, IMAGENET_STD, jpeg_quality)
        
        # 複数GPU対応
        if ngpus > 1:
            classifier_model = nn.DataParallel(classifier_model, device_ids=gpu_ids)
            jpeg_model = nn.DataParallel(jpeg_model, device_ids=gpu_ids)
        
        classifier_model = classifier_model.eval().to(device)
        jpeg_model = jpeg_model.eval().to(device)
        
        # データ読み込み
        log_print('\nLoading data...')
        x_val, y_val = load_data(args, adv_batch_size)
        
        # AutoAttack評価
        results = eval_autoattack(
            args, None, classifier_model, jpeg_model,
            x_val, y_val, adv_batch_size, log_dir, device
        )
        
        # 結果出力
        log_print(f"\n{'='*70}")
        log_print("==== Summary ====")
        log_print(f"{'='*70}")
        log_print(f"Attack: {args.attack_version}, Norm: {args.lp_norm}, Eps: {args.adv_eps}")
        log_print(f"-"*70)
        log_print(f"Classifier (no defense):")
        log_print(f"  Initial accuracy:  {results['init_acc_classifier']:.4f}")
        log_print(f"  Robust accuracy:   {results['robust_acc_classifier']:.4f}")
        log_print(f"  Attack time:       {results['classifier_time']:.2f}s")
        log_print(f"-"*70)
        log_print(f"JPEG Defense (Q={jpeg_quality}):")
        log_print(f"  Initial accuracy:  {results['init_acc_jpeg']:.4f}")
        log_print(f"  Robust accuracy:   {results['robust_acc_jpeg']:.4f}")
        log_print(f"  Attack time:       {results['jpeg_time']:.2f}s")
        log_print(f"-"*70)
        log_print(f"Defense effectiveness:")
        log_print(f"  Defended accuracy: {results['defended_acc']:.4f}")
        log_print(f"  Improvement:       {results['defense_improvement']:+.4f}")
        log_print(f"{'='*70}")
        
        # サンプル画像保存
        log_print("\nSaving sample images...")
        save_sample_images(
            x_val[:20].cpu(),
            results['x_adv_classifier'][:20].cpu(),
            y_val[:20].cpu().numpy(),
            os.path.join(log_dir, 'samples')
        )
        
        # 混同行列
        log_print("Generating confusion matrices...")
        y_true = y_val.cpu().numpy()
        
        with torch.no_grad():
            # Clean
            clean_pred = classifier_model(x_val.to(device)).argmax(dim=1).cpu().numpy()
            # Adversarial (classifier)
            adv_pred = classifier_model(results['x_adv_classifier'].to(device)).argmax(dim=1).cpu().numpy()
            # Defended
            defended_pred = jpeg_model(results['x_adv_classifier'].to(device)).argmax(dim=1).cpu().numpy()
            # JPEG attacked
            jpeg_adv_pred = jpeg_model(results['x_adv_jpeg'].to(device)).argmax(dim=1).cpu().numpy()
        
        cm_clean = plot_confusion_matrix(y_true, clean_pred, "Clean Images",
                                         os.path.join(log_dir, "cm_clean.png"))
        cm_adv = plot_confusion_matrix(y_true, adv_pred, f"Adversarial ({args.attack_version})",
                                       os.path.join(log_dir, "cm_adversarial.png"))
        cm_defended = plot_confusion_matrix(y_true, defended_pred, f"JPEG Defended (Q={jpeg_quality})",
                                            os.path.join(log_dir, "cm_defended.png"))
        cm_jpeg_adv = plot_confusion_matrix(y_true, jpeg_adv_pred, f"JPEG Attacked ({args.attack_version})",
                                            os.path.join(log_dir, "cm_jpeg_attacked.png"))
        
        # 詳細結果CSV
        stats_df = pd.DataFrame({
            'true_label': y_true,
            'pred_clean': clean_pred,
            'pred_adv': adv_pred,
            'pred_defended': defended_pred,
            'pred_jpeg_attacked': jpeg_adv_pred,
        })
        stats_df.to_csv(os.path.join(log_dir, 'predictions.csv'), index=False)
        
        # サマリーテキスト
        summary_path = os.path.join(log_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"="*70 + "\n")
            f.write(f"PCam - AutoAttack + JPEG Compression Defense\n")
            f.write(f"="*70 + "\n\n")
            f.write(f"Attack: {args.attack_version}\n")
            f.write(f"Attacks: {['apgd-ce', 'apgd-t', 'fab-t', 'square'] if args.attack_version == 'standard' else args.attack_type}\n")
            f.write(f"Norm: {args.lp_norm}, Epsilon: {args.adv_eps}\n")
            f.write(f"JPEG Quality: {jpeg_quality}\n\n")
            f.write(f"Classifier:\n")
            f.write(f"  Clean acc: {results['init_acc_classifier']:.4f}\n")
            f.write(f"  Robust acc: {results['robust_acc_classifier']:.4f}\n")
            f.write(f"\nJPEG Defense:\n")
            f.write(f"  Clean acc: {results['init_acc_jpeg']:.4f}\n")
            f.write(f"  Robust acc: {results['robust_acc_jpeg']:.4f}\n")
            f.write(f"  Defended acc: {results['defended_acc']:.4f}\n")
            f.write(f"  Improvement: {results['defense_improvement']:+.4f}\n")
        
        log_print(f"\n✅ Results saved to: {log_dir}")
        log_file.close()
        
        # 結果保存
        all_results[jpeg_quality] = {
            'init_acc_classifier': results['init_acc_classifier'],
            'robust_acc_classifier': results['robust_acc_classifier'],
            'init_acc_jpeg': results['init_acc_jpeg'],
            'robust_acc_jpeg': results['robust_acc_jpeg'],
            'defended_acc': results['defended_acc'],
            'defense_improvement': results['defense_improvement'],
        }
        
        # GPUメモリ解放
        del results['x_adv_classifier'], results['x_adv_jpeg']
        torch.cuda.empty_cache()
    
    # 全体サマリー
    print("\n" + "="*70)
    print("==== Overall Summary ====")
    print("="*70)
    
    summary_df = pd.DataFrame([
        {
            'JPEG_Quality': q,
            'Clean_Acc': r['init_acc_classifier'],
            'Robust_Acc': r['robust_acc_classifier'],
            'JPEG_Clean_Acc': r['init_acc_jpeg'],
            'JPEG_Robust_Acc': r['robust_acc_jpeg'],
            'Defended_Acc': r['defended_acc'],
            'Improvement': r['defense_improvement'],
        }
        for q, r in all_results.items()
    ])
    print(summary_df.to_string(index=False))
    
    # 全体サマリー保存
    base_dir = '/mnt/data1/gotou/projects/pcam/jpeg/autoattack/defense_results'
    summary_csv = os.path.join(base_dir, f'overall_summary_{middle_name}_sd{args.seed}.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Overall summary saved to: {summary_csv}")
    
    print("\n" + "="*70)
    print("Evaluation completed successfully!")
    print("="*70)
    
    return all_results


# ========== メイン ==========
if __name__ == '__main__':
    args = parse_args()
    
    # 乱数シード設定
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.benchmark = True
    
    robustness_eval(args)
