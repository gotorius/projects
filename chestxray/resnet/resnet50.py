"""
ChestXray (肺炎分類) - ResNet50訓練スクリプト
二値分類: NORMAL vs PNEUMONIA
最良モデルを保存して敵対的防御実験に使用
"""

import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
from tqdm.auto import tqdm

# === 設定 ===
DATA_DIR = '/mnt/data1/Public/MedImages/CellData/chest_xray'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUT_DIR = os.path.join('/mnt/data1/gotou/kaggle/chestxray', 'resnet50_models')
os.makedirs(OUT_DIR, exist_ok=True)

# 訓練データから検証データを分割する割合
VALIDATION_SPLIT = 0.15  # 15%を検証に使用

# ハイパーパラメータ
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# クラス名
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  # データセットのフォルダ名と一致
NUM_CLASSES = len(CLASS_NAMES)

print(f"Device: {DEVICE}")
print(f"Training directory: {TRAIN_DIR}")
print(f"Test directory: {TEST_DIR}")
print(f"Output directory: {OUT_DIR}")
print(f"Validation split: {VALIDATION_SPLIT:.1%}")

# === データ拡張・前処理 ===
# 訓練データ: データ拡張あり（ChestXray用: 左右反転は除外）
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.CenterCrop(IMAGE_SIZE),
    # X線画像では左右の位置情報が重要なため、HorizontalFlipは除外
    transforms.RandomRotation(10),  # 軽微な回転のみ
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 軽微な明度調整
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet統計
])

# 検証・テストデータ: データ拡張なし
valid_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === データセット読み込み ===
# 訓練データを読み込み（後で訓練/検証に分割）
full_train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=valid_transform)

# 訓練データを訓練/検証に分割
val_size = int(len(full_train_dataset) * VALIDATION_SPLIT)
train_size = len(full_train_dataset) - val_size

# シード固定して再現性を確保
torch.manual_seed(42)
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, val_size])

# 検証データセットには拡張なしのtransformを適用するため、一時的にデータセットを作成
# （random_splitは元のtransformを保持するため、検証データでも拡張が適用される）
# より良い方法: 検証用に別途読み込み
valid_dataset_proper = datasets.ImageFolder(TRAIN_DIR, transform=valid_transform)
# 同じインデックスを使用して分割
_, valid_dataset_indices = random_split(
    range(len(full_train_dataset)), 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
# Subsetを使用して検証データを作成
from torch.utils.data import Subset
valid_dataset = Subset(valid_dataset_proper, list(valid_dataset_indices))

print(f"\nDataset Statistics:")
print(f"  Total training samples: {len(full_train_dataset)}")
print(f"  Training samples (after split): {len(train_dataset)}")
print(f"  Validation samples: {len(valid_dataset)}")
print(f"  Test samples: {len(test_dataset)}")
print(f"  Classes: {full_train_dataset.classes}")
print(f"  Class to index: {full_train_dataset.class_to_idx}")

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

# === モデル定義 ===
def create_resnet50(num_classes=2, pretrained=True):
    """
    ResNet50モデルを作成
    最終層を2クラス分類用に変更
    """
    model = models.resnet50(pretrained=pretrained)
    
    # 最終全結合層を置き換え
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

model = create_resnet50(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)

print(f"\nModel: ResNet50")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# === 損失関数・最適化 ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 学習率スケジューラ: validation lossが改善しない場合に学習率を下げる
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

# === 訓練・検証関数 ===
def train_epoch(model, dataloader, criterion, optimizer, device):
    """1エポックの訓練"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = inputs.size(0)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 統計
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples += batch_size
        
        # 進捗表示
        pbar.set_postfix({
            'loss': running_loss / total_samples,
            'acc': running_corrects.double().item() / total_samples
        })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion, device):
    """1エポックの検証"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 統計
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            total_samples += batch_size
            
            # 進捗表示
            pbar.set_postfix({
                'loss': running_loss / total_samples,
                'acc': running_corrects.double().item() / total_samples
            })
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

# === 訓練ループ ===
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir):
    """
    モデルを訓練し、最良のモデルを保存
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'valid_loss': [],
        'valid_acc': []
    }
    
    print("\n" + "="*60)
    print("Training Start")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 60)
        
        start_time = time.time()
        
        # 訓練
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 検証
        valid_loss, valid_acc = validate_epoch(model, valid_loader, criterion, device)
        
        # 学習率スケジューラ更新
        scheduler.step(valid_loss)
        
        # 履歴保存
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_acc)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}  Valid Acc: {valid_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s  LR: {current_lr:.2e}")
        
        # 最良モデルの保存
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # 最良モデルを保存
            best_model_path = os.path.join(save_dir, 'resnet50_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                'class_names': CLASS_NAMES,
                'class_to_idx': full_train_dataset.class_to_idx,
            }, best_model_path)
            print(f"  *** New best model saved! (Acc: {valid_acc:.4f}) ***")
        
        # 定期的にチェックポイント保存
        if epoch % 10 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(save_dir, f'resnet50_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'valid_acc': valid_acc,
                'valid_loss': valid_loss,
                'class_names': CLASS_NAMES,
                'class_to_idx': full_train_dataset.class_to_idx,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    
    # 最良モデルの重みをロード
    model.load_state_dict(best_model_wts)
    
    return model, history

# === 推論用モデル保存関数 ===
def save_inference_model(model, save_path, class_names, class_to_idx):
    """
    推論専用のモデルを保存（軽量版）
    敵対的攻撃実験で使用
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
    }, save_path)
    print(f"\nInference model saved: {save_path}")

# === メイン実行 ===
if __name__ == "__main__":
    # 訓練実行
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCHS,
        device=DEVICE,
        save_dir=OUT_DIR
    )
    
    # 推論用モデルを保存（敵対的攻撃実験用）
    inference_model_path = os.path.join(OUT_DIR, 'resnet50_inference.pth')
    save_inference_model(
        trained_model, 
        inference_model_path, 
        CLASS_NAMES, 
        full_train_dataset.class_to_idx
    )
    
    # テストデータで最終評価
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)
    test_loss, test_acc = validate_epoch(trained_model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 訓練履歴を保存
    history_path = os.path.join(OUT_DIR, 'training_history.npz')
    np.savez(history_path, **history)
    print(f"Training history saved: {history_path}")
    
    print("\n" + "="*60)
    print("All tasks completed successfully!")
    print("="*60)
    print(f"\nSaved files:")
    print(f"  Best model: {os.path.join(OUT_DIR, 'resnet50_best.pth')}")
    print(f"  Inference model: {inference_model_path}")
    print(f"  Training history: {history_path}")
    print(f"\nTo use the model for adversarial attacks:")
    print(f"  model = models.resnet50()")
    print(f"  model.fc = nn.Linear(2048, 2)")
    print(f"  checkpoint = torch.load('{inference_model_path}')")
    print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
