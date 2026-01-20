"""
ChestXray ViT用のバランスの取れたサンプルセットを作成
NORMAL: 250枚、PNEUMONIA: 250枚の合計500枚
ViT分類器で正しく分類された画像のみを使用

データセット: /mnt/data1/Public/MedImages/CellData/chest_xray/test
分類器: /mnt/data1/gotou/projects/classifiers/vit/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth
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


def get_vit_model(model_name='vit_b_16', num_classes=2, dropout=0.1):
    """
    Vision Transformer モデルの構築（推論用）
    """
    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights=None)
    elif model_name == 'vit_b_32':
        model = models.vit_b_32(weights=None)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 分類ヘッドを置き換え（訓練時と同じ構造）
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    
    return model


def load_vit_classifier(ckpt_path, model_name='vit_b_16', num_classes=2, device='cuda'):
    """ViT分類器を読み込み"""
    model = get_vit_model(model_name=model_name, num_classes=num_classes)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # パス設定
    test_data_dir = '/mnt/data1/Public/MedImages/CellData/chest_xray/test'
    clf_ckpt = '/mnt/data1/gotou/projects/vit/classifiers/checkpoints/chestxray/20260117_190122/best_vit_chestxray.pth'
    output_dir = '/mnt/data1/gotou/projects/chestxray/vit'
    output_path = os.path.join(output_dir, 'correct_samples_balanced_500_vit.pt')
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # ViT分類器読み込み
    classifier = load_vit_classifier(clf_ckpt, model_name='vit_b_16', num_classes=2, device=device)
    print(f"Loaded ViT classifier from {clf_ckpt}")
    
    # 正しく分類された画像を収集
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    correct_samples = {i: {'images': [], 'labels': []} for i in range(2)}
    total_target = 500  # 合計目標数
    
    print("\nCollecting correctly classified samples...")
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 正規化して予測
            images_norm = (images - mean) / std
            outputs = classifier(images_norm)
            _, predicted = outputs.max(1)
            
            # 正しく分類された画像を全て収集
            correct_mask = (predicted == labels)
            
            for i in range(len(images)):
                if correct_mask[i]:
                    label = labels[i].item()
                    correct_samples[label]['images'].append(images[i].cpu())
                    correct_samples[label]['labels'].append(label)
    
    # 結果を確認
    print("\nCorrectly classified samples per class:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}: {len(correct_samples[i]['images'])} samples")
    
    # 動的に枚数を決定（ポジティブクラスを優先、不足分をネガティブから補充）
    pneumonia_count = len(correct_samples[1]['images'])  # pneumoniaはクラス1
    normal_count = total_target - pneumonia_count  # normal（クラス0）は残りで補充
    
    print(f"\nTarget distribution for 500 total samples:")
    print(f"  PNEUMONIA (positive): {pneumonia_count} samples")
    print(f"  NORMAL (negative): {normal_count} samples")
    
    # 十分なサンプルがあるか確認
    if len(correct_samples[0]['images']) < normal_count:
        normal_count = len(correct_samples[0]['images'])
        actual_total = pneumonia_count + normal_count
        print(f"\nWarning: Not enough NORMAL samples. Using {normal_count} NORMAL samples")
        print(f"Actual total: {actual_total} samples")
    
    # 各クラスから必要な枚数を取得
    x_test = []
    y_test = []
    
    # ポジティブクラス（PNEUMONIA）を全て追加
    x_test.extend(correct_samples[1]['images'][:pneumonia_count])
    y_test.extend(correct_samples[1]['labels'][:pneumonia_count])
    
    # ネガティブクラス（NORMAL）から必要な分を追加
    x_test.extend(correct_samples[0]['images'][:normal_count])
    y_test.extend(correct_samples[0]['labels'][:normal_count])
    
    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test)
    
    # 保存
    data = {
        'x_test': x_test,
        'y_test': y_test,
        'classes': classes,
        'classifier': 'vit_b_16',
        'checkpoint': clf_ckpt
    }
    
    torch.save(data, output_path)
    print(f"\nSaved balanced samples to {output_path}")
    print(f"Total samples: {len(x_test)}")
    print(f"Label distribution: {torch.bincount(y_test).tolist()}")
    print(f"Classes: {classes}")
    
    # 検証: 保存したデータで正解率を確認
    print("\n=== Verification ===")
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        x_norm = (x_test - mean) / std
        outputs = classifier(x_norm)
        _, predicted = outputs.max(1)
        accuracy = (predicted == y_test).float().mean().item()
    
    print(f"Verification accuracy on saved samples: {accuracy*100:.2f}%")


if __name__ == '__main__':
    main()
