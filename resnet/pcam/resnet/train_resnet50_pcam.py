"""
PCam (PatchCamelyon) データセットの分類モデルをResNet50で訓練するコード
データセット: /mnt/data1/Public/MedImages/PCam_ImageFolder
クラス: normal, tumor (2クラス分類)
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='PCam ResNet50 Training')
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/data1/Public/MedImages/PCam_ImageFolder',
                        help='データセットのパス')
    parser.add_argument('--save_dir', type=str, 
                        default='/mnt/data1/gotou/projects/pcam/resnet/checkpoints',
                        help='モデル保存先')
    parser.add_argument('--batch_size', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=30, help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoaderのworker数')
    parser.add_argument('--val_split', type=float, default=0.1, help='検証データの割合')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='ImageNet事前学習済みモデルを使用')
    parser.add_argument('--resume', type=str, default=None, help='再開するチェックポイント')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    return parser.parse_args()


def set_seed(seed):
    """再現性のための乱数シード設定"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    """データ変換の定義"""
    # PCamの画像サイズは96x96、ResNet50の入力は224x224
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_model(num_classes=2, pretrained=True):
    """ResNet50モデルの構築"""
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet50(weights=None)
    
    # 最終層を2クラス分類用に変更
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """1エポックの訓練"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """検証"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
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
    
    # 各クラスの精度
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def compute_metrics(preds, labels, class_names):
    """詳細な評価指標の計算"""
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n" + "="*50)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)
    print("="*50 + "\n")


def main():
    args = get_args()
    set_seed(args.seed)
    
    # 保存ディレクトリの作成
    os.makedirs(args.save_dir, exist_ok=True)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # データ変換
    train_transform, val_transform = get_transforms()
    
    # データセットの読み込み
    print(f'\nLoading dataset from: {args.data_dir}')
    
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'test')
    
    # 訓練データを訓練/検証に分割
    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    
    # クラス情報
    class_names = full_train_dataset.classes
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
    
    # 検証データには別の変換を適用するためのラッパー
    class TransformDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, idx):
            img, label = self.subset.dataset.samples[self.subset.indices[idx]]
            from PIL import Image
            img = Image.open(img).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        
        def __len__(self):
            return len(self.subset)
    
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # テストデータセット
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    print(f'Test samples: {len(test_dataset)}')
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # モデル
    print(f'\nBuilding ResNet50 model (pretrained={args.pretrained})')
    model = get_model(num_classes=len(class_names), pretrained=args.pretrained)
    model = model.to(device)
    
    # クラス重みの計算（不均衡データ対策）
    class_counts = [0, 0]
    for _, label in full_train_dataset.samples:
        class_counts[label] += 1
    
    total = sum(class_counts)
    class_weights = torch.FloatTensor([total / (2 * c) for c in class_counts]).to(device)
    print(f'Class counts: {dict(zip(class_names, class_counts))}')
    print(f'Class weights: {class_weights.tolist()}')
    
    # 損失関数、オプティマイザ、スケジューラ
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # チェックポイントから再開
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # 訓練ループ
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Batch size: {args.batch_size}, Learning rate: {args.lr}')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.2e}')
        print("="*60)
        
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # 検証
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch+1
        )
        
        scheduler.step()
        
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
            save_path = os.path.join(args.save_dir, 'best_resnet50_pcam.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
            }, save_path)
            print(f'*** Best model saved! (Val Acc: {100*best_val_acc:.2f}%) ***')
        
        # 定期的にチェックポイント保存
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
            }, save_path)
    
    # 最終モデルの保存
    save_path = os.path.join(args.save_dir, 'final_resnet50_pcam.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'class_names': class_names,
    }, save_path)
    
    # ベストモデルでテスト
    print('\n' + '='*60)
    print('Testing with best model...')
    print('='*60)
    
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'best_resnet50_pcam.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, 'Test'
    )
    
    print(f'\nTest Loss: {test_loss:.4f} | Test Acc: {100*test_acc:.2f}%')
    compute_metrics(test_preds, test_labels, class_names)
    
    # 訓練履歴の保存
    import json
    history_path = os.path.join(args.save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {100*best_val_acc:.2f}%')
    print(f'Test accuracy: {100*test_acc:.2f}%')
    print(f'Models saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
