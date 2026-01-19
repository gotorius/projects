"""
訓練済みResNet50分類器のテストデータセット精度評価スクリプト

対象データセット:
- PCam (PatchCamelyon): 病理画像の腫瘍分類
- ChestXray: 肺炎分類
- DermMel: メラノーマ分類

各データセットに対して:
1. モデルを読み込み
2. ImageFolder形式のテストデータを読み込み
3. 精度を計算
4. クラスごとの詳細を表示
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# === 設定 ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# ImageNet正規化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ========== モデルパス ==========
MODEL_PATHS = {
    'PCam': '/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
    'ChestXray': '/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
    'DermMel': '/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
}

# ========== テストデータパス (ImageFolder形式) ==========
DATA_PATHS = {
    'PCam': '/mnt/data1/Public/MedImages/PCam_ImageFolder/test',
    'ChestXray': '/mnt/data1/Public/MedImages/CellData/chest_xray/test',
    'DermMel': '/mnt/data1/Public/MedImages/DermMel/test',
}


def get_test_transform():
    """テストデータ用の変換"""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_model_pcam(checkpoint_path, device):
    """PCam用のResNet50モデルを読み込み
    
    PCamのモデルは fc = Sequential(Dropout, Linear) の構造
    """
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 2)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model


def load_model_standard(checkpoint_path, device):
    """ChestXray/DermMel用のResNet50モデルを読み込み
    
    標準的な fc = Linear の構造
    """
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    return model


def load_test_dataset(data_path):
    """ImageFolder形式のテストデータを読み込み"""
    transform = get_test_transform()
    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return dataset, loader


def evaluate(model, data_loader, device):
    """モデルの精度を評価
    
    Args:
        model: 評価するモデル
        data_loader: テストデータローダー
        device: 使用デバイス
    
    Returns:
        accuracy: 全体精度
        all_labels: 真のラベル配列
        all_predictions: 予測ラベル配列
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_labels), np.array(all_predictions)


def print_evaluation_results(dataset_name, accuracy, y_true, y_pred, classes):
    """評価結果を表示"""
    print(f"\n{'='*60}")
    print(f" {dataset_name} - ResNet50 Evaluation Results")
    print(f"{'='*60}")
    
    print(f"\n全体精度 (Accuracy): {accuracy*100:.2f}%")
    print(f"サンプル数: {len(y_true)}")
    
    # クラスごとのサンプル数
    print(f"\nクラス分布:")
    for i, cls in enumerate(classes):
        count = (y_true == i).sum()
        print(f"  {cls}: {count}")
    
    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混同行列:")
    print(f"              予測")
    header = "".join([f"{cls:>12}" for cls in classes])
    print(f"             {header}")
    for i, cls in enumerate(classes):
        row = "".join([f"{cm[i,j]:>12}" for j in range(len(classes))])
        print(f"真値 {cls:>8} {row}")
    
    # 分類レポート
    print(f"\n詳細レポート:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))


def main():
    print(f"使用デバイス: {DEVICE}")
    print(f"\n{'#'*60}")
    print(f" ResNet50 分類器 - テストデータセット精度評価")
    print(f"{'#'*60}")
    
    results = {}
    
    # 各データセットを評価
    for dataset_name in ['PCam', 'ChestXray', 'DermMel']:
        print(f"\n\n{'*'*60}")
        print(f" Processing: {dataset_name}")
        print(f"{'*'*60}")
        
        model_path = MODEL_PATHS[dataset_name]
        data_path = DATA_PATHS[dataset_name]
        
        # パスの存在確認
        if not os.path.exists(model_path):
            print(f"Error: モデルが見つかりません: {model_path}")
            continue
        if not os.path.exists(data_path):
            print(f"Error: テストデータが見つかりません: {data_path}")
            continue
        
        print(f"モデルパス: {model_path}")
        print(f"データパス: {data_path}")
        
        # モデル読み込み
        if dataset_name == 'PCam':
            model = load_model_pcam(model_path, DEVICE)
        else:
            model = load_model_standard(model_path, DEVICE)
        print(f"モデル読み込み完了")
        
        # データ読み込み
        dataset, data_loader = load_test_dataset(data_path)
        classes = dataset.classes
        print(f"テストデータ読み込み完了: {len(dataset)} samples")
        print(f"クラス: {classes}")
        print(f"クラス→インデックス: {dataset.class_to_idx}")
        
        # 評価
        accuracy, y_true, predictions = evaluate(model, data_loader, DEVICE)
        
        # 結果表示
        print_evaluation_results(dataset_name, accuracy, y_true, predictions, classes)
        
        # 結果を保存
        results[dataset_name] = {
            'accuracy': accuracy,
            'num_samples': len(dataset),
            'classes': classes
        }
        
        # メモリ解放
        del model
        torch.cuda.empty_cache()
    
    # 全体のサマリー
    print(f"\n\n{'#'*60}")
    print(f" Summary - All Datasets")
    print(f"{'#'*60}")
    print(f"\n{'Dataset':<15} {'Accuracy':>12} {'Samples':>10}")
    print(f"{'-'*40}")
    for name, res in results.items():
        print(f"{name:<15} {res['accuracy']*100:>11.2f}% {res['num_samples']:>10}")
    print(f"{'-'*40}")


if __name__ == '__main__':
    main()
