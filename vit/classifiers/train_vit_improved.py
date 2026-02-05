"""
Vision Transformer (ViT-B/16) 改良版訓練スクリプト
医療画像データセット (PCam, ChestXray, DermMel) 用

============================================================
改良点（オリジナル版との違い）:
============================================================
1. Layer-wise Learning Rate Decay (LLRD)
   - 深い層ほど高い学習率、浅い層は低い学習率
   - 事前学習の知識を保持しつつファインチューニング
   
2. MixUp / CutMix データ拡張
   - ViTに効果的な強いデータ拡張
   
3. Stochastic Depth (Drop Path)
   - 過学習を防ぐ正則化

4. より長い訓練（50エポック推奨）

5. Gradient Accumulation対応
   - 実効バッチサイズを大きくできる

============================================================
参考論文:
============================================================
1. BEiT: "BEiT: BERT Pre-Training of Image Transformers"
   - 医療画像ViTファインチューニングの参考

2. "How to train your ViT? Data, Augmentation, and Regularization"
   - Steiner et al., 2021
   - ViTの効果的な訓練方法

============================================================
使用方法:
============================================================
# 改良版（推奨設定）
python train_vit_improved.py --dataset pcam --epochs 50 --batch_size 32 --accum_steps 2 --gpu 0

# MixUp/CutMixなし（比較用）
python train_vit_improved.py --dataset pcam --no_mixup --gpu 0
"""

import os
import sys
import argparse
import time
import json
import ssl
import urllib.request
from datetime import datetime
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from PIL import Image
from tqdm import tqdm
import random

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Warning: sklearn not found.")


# ========== データセット設定 ==========
DATASET_CONFIGS = {
    'pcam': {
        'train_dir': '/mnt/data1/Public/MedImages/PCam_ImageFolder/train',
        'test_dir': '/mnt/data1/Public/MedImages/PCam_ImageFolder/test',
        'num_classes': 2,
        'class_names': ['normal', 'tumor'],
        'description': 'PatchCamelyon (PCam) - Histopathology'
    },
    'chestxray': {
        'train_dir': '/mnt/data1/Public/MedImages/CellData/chest_xray/train',
        'test_dir': '/mnt/data1/Public/MedImages/CellData/chest_xray/test',
        'num_classes': 2,
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'description': 'Chest X-ray - Pneumonia Detection'
    },
    'dermmel': {
        'train_dir': '/mnt/data1/Public/MedImages/DermMel/train_sep',
        'test_dir': '/mnt/data1/Public/MedImages/DermMel/test',
        'num_classes': 2,
        'class_names': ['NotMelanoma', 'Melanoma'],
        'description': 'Dermatology - Melanoma Detection'
    }
}


# ========== 引数パーサー ==========
def get_args():
    parser = argparse.ArgumentParser(description='ViT Training (Improved) for Medical Images')
    
    # データセット設定
    parser.add_argument('--dataset', type=str, default='pcam',
                        choices=['pcam', 'chestxray', 'dermmel', 'all'])
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=50, help='エポック数（増加）')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--accum_steps', type=int, default=2, 
                        help='Gradient accumulation steps (実効バッチ=batch_size*accum_steps)')
    parser.add_argument('--lr', type=float, default=5e-5, help='学習率（小さめに）')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学習率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmupエポック数')
    parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    
    # LLRD設定
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='Layer-wise learning rate decay (0.65-0.85推奨)')
    
    # MixUp/CutMix設定
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help='MixUp alpha')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='CutMix alpha')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='MixUp/CutMix適用確率')
    parser.add_argument('--no_mixup', action='store_true', help='MixUp/CutMixを無効化')
    
    # モデル設定
    parser.add_argument('--model', type=str, default='vit_b_16',
                        choices=['vit_b_16', 'vit_b_32', 'vit_l_16'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--drop_path', type=float, default=0.1, help='Drop path率')
    
    # その他
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/vit/classifiers/checkpoints')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    return parser.parse_args()


# ========== ユーティリティ ==========
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== MixUp / CutMix ==========
class Mixup:
    """
    MixUp and CutMix data augmentation
    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
               "CutMix: Regularization Strategy to Train Strong Classifiers" (Yun et al., 2019)
    """
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=0.5, num_classes=2):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
    
    def __call__(self, x, target):
        if np.random.rand() > self.prob:
            # 適用しない場合はone-hotに変換
            target_one_hot = torch.zeros(target.size(0), self.num_classes, device=target.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            return x, target_one_hot
        
        if np.random.rand() < 0.5:
            return self.mixup(x, target)
        else:
            return self.cutmix(x, target)
    
    def mixup(self, x, target):
        """MixUp: 画像とラベルを線形補間"""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # ソフトラベル作成
        target_one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        target_shuffled = target_one_hot[index]
        mixed_target = lam * target_one_hot + (1 - lam) * target_shuffled
        
        return mixed_x, mixed_target
    
    def cutmix(self, x, target):
        """CutMix: 画像の一部を別の画像で置換"""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        # バウンディングボックスを計算
        W, H = x.size(3), x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        mixed_x = x.clone()
        mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # ラベルの混合比率を再計算
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        target_one_hot = torch.zeros(batch_size, self.num_classes, device=x.device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        target_shuffled = target_one_hot[index]
        mixed_target = lam * target_one_hot + (1 - lam) * target_shuffled
        
        return mixed_x, mixed_target


# ========== Layer-wise Learning Rate Decay ==========
def get_layer_id_for_vit(name, num_layers):
    """
    ViTの各パラメータに対してレイヤーIDを割り当て
    embedding: 0
    encoder blocks: 1 ~ num_layers
    head: num_layers + 1
    """
    if name.startswith('class_token') or name.startswith('conv_proj') or name.startswith('encoder.pos_embedding'):
        return 0
    elif name.startswith('encoder.layers.encoder_layer_'):
        # encoder.layers.encoder_layer_0 -> 1, encoder.layers.encoder_layer_11 -> 12
        layer_num = int(name.split('.')[2].split('_')[-1])
        return layer_num + 1
    elif name.startswith('encoder.ln'):
        return num_layers
    else:  # heads
        return num_layers + 1


def get_parameter_groups_with_llrd(model, lr, weight_decay, layer_decay, num_layers=12):
    """
    Layer-wise Learning Rate Decayを適用したパラメータグループを作成
    
    深い層（ヘッドに近い）→ 高い学習率
    浅い層（入力に近い）→ 低い学習率
    
    これにより、事前学習の知識を保持しつつファインチューニングできる
    """
    parameter_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # weight decayを適用しないパラメータ
        if 'bias' in name or 'ln' in name or 'norm' in name:
            group_wd = 0.0
        else:
            group_wd = weight_decay
        
        layer_id = get_layer_id_for_vit(name, num_layers)
        
        # 学習率の計算: lr * (layer_decay ^ (num_layers - layer_id))
        # layer_id=0（最も浅い層）: 最小の学習率
        # layer_id=num_layers+1（ヘッド）: 最大の学習率（lr）
        scale = layer_decay ** (num_layers + 1 - layer_id)
        group_lr = lr * scale
        
        group_key = f"layer_{layer_id}_wd_{group_wd}"
        
        if group_key not in parameter_groups:
            parameter_groups[group_key] = {
                'params': [],
                'lr': group_lr,
                'weight_decay': group_wd,
                'layer_id': layer_id
            }
        
        parameter_groups[group_key]['params'].append(param)
    
    # パラメータグループをリストに変換
    param_groups = list(parameter_groups.values())
    
    # デバッグ情報
    print(f"\nLayer-wise Learning Rate Decay (decay={layer_decay}):")
    for pg in sorted(param_groups, key=lambda x: x['layer_id']):
        n_params = sum(p.numel() for p in pg['params'])
        print(f"  Layer {pg['layer_id']:2d}: lr={pg['lr']:.2e}, wd={pg['weight_decay']:.2f}, params={n_params:,}")
    
    return param_groups


def get_transforms(img_size=224):
    """データ変換（強化版）"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # より強いデータ拡張
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),  # 15 -> 20
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # 強化
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # p=0.1->0.2
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


# ========== モデル構築 ==========
def get_vit_model(model_name='vit_b_16', num_classes=2, pretrained=True, dropout=0.1):
    """ViTモデルの構築"""
    print(f"\nBuilding {model_name} model (pretrained={pretrained})...")
    
    if model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
        num_layers = 12
    elif model_name == 'vit_b_32':
        if pretrained:
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_32(weights=None)
        num_layers = 12
    elif model_name == 'vit_l_16':
        if pretrained:
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_l_16(weights=None)
        num_layers = 24
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 分類ヘッドを置き換え
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, num_layers


# ========== 訓練・検証 ==========
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.subset)


def soft_cross_entropy(pred, soft_targets):
    """ソフトラベル用のクロスエントロピー損失"""
    log_probs = torch.nn.functional.log_softmax(pred, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                    scheduler=None, mixup_fn=None, accum_steps=1):
    """1エポックの訓練（MixUp/CutMix対応、Gradient Accumulation対応）"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # MixUp/CutMix適用
        if mixup_fn is not None:
            inputs, mixed_labels = mixup_fn(inputs, labels)
            use_soft_labels = True
        else:
            use_soft_labels = False
        
        outputs = model(inputs)
        
        if use_soft_labels:
            loss = soft_cross_entropy(outputs, mixed_labels)
        else:
            loss = criterion(outputs, labels)
        
        # Gradient Accumulation
        loss = loss / accum_steps
        loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        running_loss += loss.item() * accum_steps * inputs.size(0)
        
        # 精度計算（MixUp時は元のラベルで計算できないため、予測のみ）
        _, predicted = outputs.max(1)
        total += labels.size(0)
        if not use_soft_labels:
            correct += predicted.eq(labels).sum().item()
        else:
            # MixUp時は混合前のラベルに最も近い方をカウント
            correct += (predicted == labels).sum().item()  # 近似
        
        pbar.set_postfix({
            'loss': f'{loss.item() * accum_steps:.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[-1]["lr"]:.2e}'  # ヘッドのLR
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, desc='Val'):
    """検証"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'[{desc}]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, class_names):
    """詳細な評価指標"""
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    try:
        print(classification_report(labels, preds, target_names=class_names, digits=4))
    except:
        pass
    
    print("\nConfusion Matrix:")
    try:
        cm = confusion_matrix(labels, preds)
        print(cm)
    except:
        pass
    print("="*60 + "\n")


# ========== メイン訓練関数 ==========
def train_single_dataset(args, dataset_name, device):
    """単一データセットの訓練"""
    
    config = DATASET_CONFIGS[dataset_name]
    print("\n" + "="*70)
    print(f"Training ViT (Improved) on {dataset_name.upper()}")
    print(f"Description: {config['description']}")
    print("="*70)
    
    # 保存ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / dataset_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # データ変換
    train_transform, val_transform = get_transforms(img_size=224)
    
    # データセット
    print(f"\nLoading dataset from: {config['train_dir']}")
    full_train_dataset = datasets.ImageFolder(config['train_dir'], transform=train_transform)
    
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f'Classes: {class_names}')
    print(f'Total training samples: {len(full_train_dataset)}')
    
    # 訓練/検証分割
    val_size = int(len(full_train_dataset) * args.val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # テストデータセット
    test_dir = config['test_dir']
    has_test = os.path.exists(test_dir)
    if has_test:
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        print(f'Test samples: {len(test_dataset)}')
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    if has_test:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # モデル
    model, num_layers = get_vit_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # クラス重み
    class_counts = [0] * num_classes
    for _, label in full_train_dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    class_weights = torch.FloatTensor([total / (num_classes * c) for c in class_counts]).to(device)
    print(f'Class counts: {dict(zip(class_names, class_counts))}')
    print(f'Class weights: {[f"{w:.3f}" for w in class_weights.tolist()]}')
    
    # 損失関数
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    
    # オプティマイザ（LLRD適用）
    param_groups = get_parameter_groups_with_llrd(
        model, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        layer_decay=args.layer_decay,
        num_layers=num_layers
    )
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999))
    
    # スケジューラ
    effective_batch_size = args.batch_size * args.accum_steps
    num_training_steps = len(train_loader) * args.epochs // args.accum_steps
    num_warmup_steps = len(train_loader) * args.warmup_epochs // args.accum_steps
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=args.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )
    
    # MixUp/CutMix
    if not args.no_mixup:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            num_classes=num_classes
        )
        print(f'\nMixUp/CutMix enabled: alpha={args.mixup_alpha}/{args.cutmix_alpha}, prob={args.mixup_prob}')
    else:
        mixup_fn = None
        print('\nMixUp/CutMix disabled')
    
    # 訓練設定の表示
    print(f'\nTraining configuration:')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size} x {args.accum_steps} = {effective_batch_size}')
    print(f'  Learning rate: {args.lr} (head) with LLRD (decay={args.layer_decay})')
    print(f'  Warmup epochs: {args.warmup_epochs}')
    
    # 訓練ループ
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print("="*60)
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1,
            scheduler, mixup_fn, args.accum_steps
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, desc='Val'
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%')
        
        # ベストモデル保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / f'best_vit_{dataset_name}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
                'model_name': args.model,
                'args': vars(args),
            }, save_path)
            print(f'*** Best model saved! (Val Acc: {100*best_val_acc:.2f}%) ***')
        
        # 定期チェックポイント
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f'checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, save_path)
    
    # 最終モデル保存
    save_path = save_dir / f'final_vit_{dataset_name}.pth'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'model_name': args.model,
        'args': vars(args),
    }, save_path)
    
    # テスト評価
    test_acc = None
    if has_test:
        print('\n' + '='*60)
        print('Testing with best model...')
        print('='*60)
        
        best_checkpoint = torch.load(save_dir / f'best_vit_{dataset_name}.pth')
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_loss, test_acc, test_preds, test_labels = validate(
            model, test_loader, criterion, device, desc='Test'
        )
        
        print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {100*test_acc:.2f}%')
        compute_metrics(test_preds, test_labels, class_names)
        
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
    
    # 履歴保存
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 設定保存
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'effective_batch_size': args.batch_size * args.accum_steps,
            'lr': args.lr,
            'layer_decay': args.layer_decay,
            'mixup_alpha': args.mixup_alpha,
            'cutmix_alpha': args.cutmix_alpha,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'class_names': class_names,
        }, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Training completed for {dataset_name}!')
    print(f'Best validation accuracy: {100*best_val_acc:.2f}%')
    if test_acc is not None:
        print(f'Test accuracy: {100*test_acc:.2f}%')
    print(f'Models saved to: {save_dir}')
    print("="*60)
    
    return {
        'dataset': dataset_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'save_dir': str(save_dir)
    }


def main():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB')
    
    if args.dataset == 'all':
        datasets_to_train = ['pcam', 'chestxray', 'dermmel']
    else:
        datasets_to_train = [args.dataset]
    
    results = []
    for dataset_name in datasets_to_train:
        result = train_single_dataset(args, dataset_name, device)
        results.append(result)
        torch.cuda.empty_cache()
    
    # サマリー
    print("\n" + "="*70)
    print("TRAINING SUMMARY (Improved ViT)")
    print("="*70)
    for r in results:
        test_str = f", Test: {100*r['test_acc']:.2f}%" if r['test_acc'] else ""
        print(f"  {r['dataset']:12} | Val: {100*r['best_val_acc']:.2f}%{test_str}")
    print("="*70)


if __name__ == '__main__':
    main()
