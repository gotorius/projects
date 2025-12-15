"""
PCamデータセット用: 分類器で正しく分類された先頭500枚を抽出してキャッシュ
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm

# 設定
DATA_DIR = '/mnt/data1/Public/MedImages/PCam_ImageFolder/test'
CLASSIFIER_PATH = '/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth'
OUTPUT_PATH = '/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_500.pt'
IMAGE_SIZE = 224
NUM_SAMPLES = 500
GPU_ID = 0

# ImageNet正規化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def main():
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # データセット
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f'Total test samples: {len(dataset)}')
    print(f'Classes: {dataset.classes}')
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 分類器
    classifier = models.resnet50(weights=None)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, len(dataset.classes))
    )
    
    checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device).eval()
    print(f'Loaded classifier from {CLASSIFIER_PATH}')
    
    # 正解サンプルを収集
    correct_images = []
    correct_labels = []
    correct_indices = []
    
    sample_idx = 0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Finding correct samples'):
            if len(correct_images) >= NUM_SAMPLES:
                break
            
            images = images.to(device)
            labels = labels.to(device)
            
            # 正規化して予測
            images_norm = (images - mean) / std
            outputs = classifier(images_norm)
            _, predicted = outputs.max(1)
            
            # 正解したサンプルを保存
            correct_mask = (predicted == labels)
            for i in range(len(images)):
                if correct_mask[i] and len(correct_images) < NUM_SAMPLES:
                    correct_images.append(images[i].cpu())
                    correct_labels.append(labels[i].cpu())
                    correct_indices.append(sample_idx + i)
            
            sample_idx += len(images)
    
    # テンソルに変換
    x_test = torch.stack(correct_images)
    y_test = torch.stack(correct_labels)
    indices = torch.tensor(correct_indices)
    
    print(f'\nCollected {len(x_test)} correctly classified samples')
    print(f'Label distribution: {torch.bincount(y_test)}')
    
    # 保存
    torch.save({
        'x_test': x_test,
        'y_test': y_test,
        'indices': indices,
        'classes': dataset.classes,
        'data_dir': DATA_DIR,
        'classifier_path': CLASSIFIER_PATH,
        'num_samples': len(x_test)
    }, OUTPUT_PATH)
    
    print(f'\nSaved to {OUTPUT_PATH}')
    print(f'Shape: x_test={x_test.shape}, y_test={y_test.shape}')

if __name__ == '__main__':
    main()
