"""
Vision Transformer (ViT-B/16) 訓練スクリプト
医療画像データセット (PCam, ChestXray, DermMel) 用

============================================================
参考論文:
============================================================
1. ViT: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
   - Dosovitskiy et al., ICLR 2021
   - https://arxiv.org/abs/2010.11929

2. DeiT: "Training data-efficient image transformers & distillation through attention"
   - Touvron et al., ICML 2021
   - https://arxiv.org/abs/2012.12877

============================================================
使用方法:
============================================================
# PCam データセット
python train_vit.py --dataset pcam --epochs 30 --batch_size 32 --gpu 0

# ChestXray データセット
python train_vit.py --dataset chestxray --epochs 30 --batch_size 32 --gpu 0

# DermMel データセット
python train_vit.py --dataset dermmel --epochs 30 --batch_size 32 --gpu 0

# すべてのデータセットを順次訓練
python train_vit.py --dataset all --epochs 30 --batch_size 32 --gpu 0

============================================================
メモリ目安 (ViT-B/16, 224x224):
============================================================
- batch_size=16: ~6GB VRAM
- batch_size=32: ~10GB VRAM
- batch_size=64: ~18GB VRAM (RTX 2080 Ti では厳しい)
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
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("Warning: sklearn not found. Some metrics may not be available.")


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
    parser = argparse.ArgumentParser(description='ViT Training for Medical Images')
    
    # データセット設定
    parser.add_argument('--dataset', type=str, default='pcam',
                        choices=['pcam', 'chestxray', 'dermmel', 'all'],
                        help='訓練するデータセット')
    
    # 訓練設定
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmupエポック数')
    parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    
    # モデル設定
    parser.add_argument('--model', type=str, default='vit_b_16',
                        choices=['vit_b_16', 'vit_b_32', 'vit_l_16'],
                        help='ViTモデルの種類')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='ImageNet事前学習済みモデルを使用')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='バックボーンを凍結してヘッドのみ訓練')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # その他
    parser.add_argument('--save_dir', type=str,
                        default='/mnt/data1/gotou/projects/classifiers/vit/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoaderのworker数')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--gpu', type=int, default=0, help='使用するGPU ID')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='ラベルスムージング')
    
    return parser.parse_args()


# ========== ユーティリティ ==========
def set_seed(seed):
    """再現性のための乱数シード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms(img_size=224):
    """
    データ変換の定義
    ViT用に最適化された拡張
    """
    # ImageNetの正規化パラメータ
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # 少し大きくリサイズ
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # 医療画像では有効
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.1),  # Cutout的な拡張
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


# ========== モデル構築 ==========
def get_vit_model(model_name='vit_b_16', num_classes=2, pretrained=True, dropout=0.1):
    """
    Vision Transformer モデルの構築
    
    Args:
        model_name: 'vit_b_16', 'vit_b_32', 'vit_l_16'
        num_classes: 出力クラス数
        pretrained: ImageNet事前学習済みを使用するか
        dropout: Dropout率
    """
    print(f"\nBuilding {model_name} model (pretrained={pretrained})...")
    
    # モデルの取得
    if model_name == 'vit_b_16':
        if pretrained:
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        if pretrained:
            model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        if pretrained:
            model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        else:
            model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 分類ヘッドを置き換え
    # ViTのヘッドは model.heads.head
    in_features = model.heads.head.in_features
    
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    
    # パラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def freeze_backbone(model):
    """バックボーンを凍結"""
    for name, param in model.named_parameters():
        if 'heads' not in name:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after freezing: {trainable_params:,}")


# ========== 訓練・検証 ==========
class TransformDataset(Dataset):
    """Subset用のTransformラッパー"""
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


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scheduler=None):
    """1エポックの訓練"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping (Transformerでは重要)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # バッチ内スケジューラ更新 (warmup用)
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
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
    """詳細な評価指標の計算"""
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    try:
        print(classification_report(labels, preds, target_names=class_names, digits=4))
    except:
        # sklearn がない場合
        for i, name in enumerate(class_names):
            mask = labels == i
            acc = (preds[mask] == labels[mask]).mean() if mask.sum() > 0 else 0
            print(f"  {name}: {acc:.4f}")
    
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
    print(f"Training ViT on {dataset_name.upper()}")
    print(f"Description: {config['description']}")
    print("="*70)
    
    # 保存ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / dataset_name / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Save directory: {save_dir}")
    
    # データ変換
    train_transform, val_transform = get_transforms(img_size=224)
    
    # データセットの読み込み
    print(f"\nLoading dataset from: {config['train_dir']}")
    
    full_train_dataset = datasets.ImageFolder(config['train_dir'], transform=train_transform)
    
    # クラス情報
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f'Classes: {class_names}')
    print(f'Total training samples: {len(full_train_dataset)}')
    
    # 訓練/検証の分割
    val_size = int(len(full_train_dataset) * args.val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 検証データには別の変換を適用
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # テストデータセット
    test_dir = config['test_dir']
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
        print(f'Test samples: {len(test_dataset)}')
        has_test = True
    else:
        print(f'Warning: Test directory not found: {test_dir}')
        has_test = False
    
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
    
    # モデル構築
    model = get_vit_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    if args.freeze_backbone:
        freeze_backbone(model)
    
    model = model.to(device)
    
    # クラス重みの計算（不均衡データ対策）
    class_counts = [0] * num_classes
    for _, label in full_train_dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    class_weights = torch.FloatTensor([total / (num_classes * c) for c in class_counts]).to(device)
    print(f'Class counts: {dict(zip(class_names, class_counts))}')
    print(f'Class weights: {[f"{w:.3f}" for w in class_weights.tolist()]}')
    
    # 損失関数（ラベルスムージング付き）
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    
    # オプティマイザ（AdamW with weight decay）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学習率スケジューラ（Warmup + Cosine Annealing）
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = len(train_loader) * args.warmup_epochs
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=num_warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_steps]
    )
    
    # 訓練ループ
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Initial LR: {args.lr}')
    print(f'Warmup epochs: {args.warmup_epochs}')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_epoch = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs}')
        print("="*60)
        
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1, scheduler
        )
        
        # 検証
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, desc='Val'
        )
        
        # 履歴の保存
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}%')
        
        # ベストモデルの保存
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
        
        # 定期的にチェックポイント保存
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f'checkpoint_epoch{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
                'model_name': args.model,
            }, save_path)
    
    # 最終モデルの保存
    save_path = save_dir / f'final_vit_{dataset_name}.pth'
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'model_name': args.model,
        'args': vars(args),
    }, save_path)
    
    # ベストモデルでテスト
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
        
        # テスト結果を保存
        history['test_loss'] = test_loss
        history['test_acc'] = test_acc
    
    # 訓練履歴の保存
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # 設定の保存
    config_path = save_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc if has_test else None,
            'class_names': class_names,
        }, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Training completed for {dataset_name}!')
    print(f'Best validation accuracy: {100*best_val_acc:.2f}%')
    if has_test:
        print(f'Test accuracy: {100*test_acc:.2f}%')
    print(f'Models saved to: {save_dir}')
    print("="*60)
    
    return {
        'dataset': dataset_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc if has_test else None,
        'save_dir': str(save_dir)
    }


def main():
    args = get_args()
    set_seed(args.seed)
    
    # デバイス設定
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(args.gpu)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB')
    
    # 訓練対象のデータセット
    if args.dataset == 'all':
        datasets_to_train = ['pcam', 'chestxray', 'dermmel']
    else:
        datasets_to_train = [args.dataset]
    
    # 各データセットで訓練
    results = []
    for dataset_name in datasets_to_train:
        result = train_single_dataset(args, dataset_name, device)
        results.append(result)
        
        # GPU メモリ解放
        torch.cuda.empty_cache()
    
    # 全体のサマリー
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for r in results:
        test_str = f", Test: {100*r['test_acc']:.2f}%" if r['test_acc'] else ""
        print(f"  {r['dataset']:12} | Val: {100*r['best_val_acc']:.2f}%{test_str}")
    print("="*70)


if __name__ == '__main__':
    main()
