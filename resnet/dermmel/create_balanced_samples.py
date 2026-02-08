"""
DermMel バランスの取れたサンプルセットを作成
Melanoma: 250枚、NotMelanoma: 250枚の合計500枚
分類器で正しく分類された画像のみを使用

データセット: /mnt/data1/Public/MedImages/DermMel/test
分類器: /mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 定数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_classifier(ckpt_path, num_classes=2, device='cuda:1'):
    """分類器を読み込み (DermMel用 - Dropoutなし)"""
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, num_classes)  # Dropoutなし
    
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
    test_data_dir = '/mnt/data1/Public/MedImages/DermMel/test'
    clf_ckpt = '/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth'
    output_path = '/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt'
    
    # データ変換（ImageNetの正規化なし、[0,1]のピクセル値で保存）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # [0, 1]
    ])
    
    # ImageFolderでデータ読み込み
    print(f"Loading test data from {test_data_dir}...")
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # クラス名を取得
    classes = test_dataset.classes
    print(f"Classes: {classes}")
    print(f"Total samples: {len(test_dataset)}")
    
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
            
            # 両クラスで目標数に達したか確認
            all_done = all(
                len(correct_samples[c]['images']) >= target_per_class
                for c in range(2)
            )
            if all_done:
                break
    
    # 結果の確認
    print("\nCollection results:")
    for c in range(2):
        print(f"  {classes[c]}: {len(correct_samples[c]['images'])} samples")
    
    # テンソルに変換
    all_images = []
    all_labels = []
    for c in range(2):
        all_images.extend(correct_samples[c]['images'])
        all_labels.extend(correct_samples[c]['labels'])
    
    x_test = torch.stack(all_images)
    y_test = torch.tensor(all_labels)
    
    print(f"\nTotal samples: {len(x_test)}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Label distribution: {torch.bincount(y_test).tolist()}")
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'x_test': x_test,
        'y_test': y_test,
        'classes': classes,
    }, output_path)
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
