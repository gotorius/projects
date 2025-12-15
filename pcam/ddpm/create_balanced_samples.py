"""
バランスの取れたサンプルセットを作成
normal: 250枚、tumor: 250枚の合計500枚
分類器で正しく分類された画像のみを使用
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import h5py
import numpy as np

# 定数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_classifier(ckpt_path, num_classes=2, device='cuda'):
    """分類器を読み込み"""
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    return classifier

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # パス設定
    test_x_path = '/mnt/data1/gotou/kaggle/pcamdata/camelyonpatch_level_2_split_test_x.h5'
    test_y_path = '/mnt/data1/gotou/kaggle/pcamdata/camelyonpatch_level_2_split_test_y.h5'
    clf_ckpt = '/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth'
    output_path = '/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt'
    
    # データ読み込み
    print("Loading test data from H5 files...")
    with h5py.File(test_x_path, 'r') as f:
        x_data = f['x'][:]  # (N, 96, 96, 3)
    with h5py.File(test_y_path, 'r') as f:
        y_data = f['y'][:, 0, 0, 0]  # (N,)
    
    print(f"Loaded {len(x_data)} samples")
    print(f"Label distribution: {np.bincount(y_data)}")
    
    # データ前処理: (N, 96, 96, 3) -> (N, 3, 224, 224), [0, 1]
    print("Preprocessing images...")
    x_tensor_list = []
    batch_size_preprocess = 100
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    
    for i in tqdm(range(0, len(x_data), batch_size_preprocess), desc="Resizing"):
        batch = x_data[i:i+batch_size_preprocess]
        # (B, 96, 96, 3) -> (B, 3, 96, 96), uint8 -> float32 [0, 1]
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0
        # Resize to 224x224
        batch_resized = transform(batch_tensor)
        x_tensor_list.append(batch_resized)
    
    x_all = torch.cat(x_tensor_list, dim=0)
    y_all = torch.from_numpy(y_data).long()
    
    print(f"Preprocessed shape: {x_all.shape}")
    
    # データローダー
    dataset = TensorDataset(x_all, y_all)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    classes = ['normal', 'tumor']
    
    # 分類器読み込み
    classifier = load_classifier(clf_ckpt, num_classes=2, device=device)
    print(f"Loaded classifier from {clf_ckpt}")
    
    # 正しく分類された画像を収集
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    correct_samples = {i: {'images': [], 'labels': []} for i in range(2)}
    target_per_class = 250
    
    print("\nCollecting correctly classified samples...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 正規化して予測
            images_norm = (images - mean) / std
            outputs = classifier(images_norm)
            _, predicted = outputs.max(1)
            
            # 正しく分類された画像を収集
            correct_mask = (predicted == labels)
            
            for i in range(len(images)):
                if correct_mask[i]:
                    label = labels[i].item()
                    if len(correct_samples[label]['images']) < target_per_class:
                        correct_samples[label]['images'].append(images[i].cpu())
                        correct_samples[label]['labels'].append(label)
            
            # 両クラスとも目標数に達したら終了
            if all(len(correct_samples[i]['images']) >= target_per_class 
                   for i in range(2)):
                break
    
    # 結果を確認
    print("\nCollected samples:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}: {len(correct_samples[i]['images'])} samples")
    
    # 各クラスから250枚ずつ取得
    x_test = []
    y_test = []
    
    for i in range(2):
        x_test.extend(correct_samples[i]['images'][:target_per_class])
        y_test.extend(correct_samples[i]['labels'][:target_per_class])
    
    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test)
    
    # 保存
    data = {
        'x_test': x_test,
        'y_test': y_test,
        'classes': classes
    }
    
    torch.save(data, output_path)
    print(f"\nSaved balanced samples to {output_path}")
    print(f"Total samples: {len(x_test)}")
    print(f"Label distribution: {torch.bincount(y_test)}")
    print(f"Classes: {classes}")

if __name__ == '__main__':
    main()
