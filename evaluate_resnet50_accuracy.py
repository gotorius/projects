"""
訓練済みResNet50分類器のテストデータセット精度評価スクリプト

対象データセット:
- PCam (PatchCamelyon): 病理画像の腫瘍分類
- ChestXray: 肺炎分類
- DermMel: メラノーマ分類

各データセットに対して:
1. モデルを読み込み
2. テストデータを読み込み
3. 精度を計算
4. クラスごとの詳細を表示
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

# === 設定 ===
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ImageNet正規化パラメータ
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ========== モデルパス ==========
MODEL_PATHS = {
    'PCam': '/mnt/data1/gotou/projects/pcam/resnet/checkpoints/best_resnet50_pcam.pth',
    'ChestXray': '/mnt/data1/gotou/projects/chestxray/resnet/resnet50_best.pth',
    'DermMel': '/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth',
}

# ========== テストデータパス ==========
DATA_PATHS = {
    'PCam': '/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_balanced_500.pt',
    'ChestXray': '/mnt/data1/gotou/projects/chestxray/correct_samples_500.pt',
    'DermMel': '/mnt/data1/gotou/projects/dermmel/ddpm/correct_samples_balanced_500.pt',
}

# ========== クラス名 (データファイルにクラス情報がない場合のフォールバック) ==========
CLASS_NAMES = {
    'PCam': ['normal', 'tumor'],
    'ChestXray': ['NORMAL', 'PNEUMONIA'],
    'DermMel': ['Melanoma', 'NotMelanoma'],
}


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


def load_test_data(data_path, dataset_name):
    """テストデータを読み込み"""
    data = torch.load(data_path, map_location='cpu')
    x_test = data['x_test']
    y_test = data['y_test']
    # classesキーがあれば使用、なければ事前定義を使用
    classes = data.get('classes', CLASS_NAMES[dataset_name])
    return x_test, y_test, classes


def evaluate(model, x_test, y_test, device, batch_size=32):
    """モデルの精度を評価
    
    Args:
        model: 評価するモデル
        x_test: テスト画像 (N, C, H, W) [0, 1]の範囲
        y_test: テストラベル (N,)
        device: 使用デバイス
        batch_size: バッチサイズ
    
    Returns:
        accuracy: 全体精度
        predictions: 予測ラベル配列
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(x_test), batch_size), desc='Evaluating'):
            x_batch = x_test[i:i+batch_size].to(device)
            y_batch = y_test[i:i+batch_size].to(device)
            
            # ImageNet正規化
            x_norm = (x_batch - mean) / std
            
            outputs = model(x_norm)
            _, predicted = outputs.max(1)
            
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(predictions)


def print_evaluation_results(dataset_name, accuracy, y_true, y_pred, classes):
    """評価結果を表示"""
    print(f"\n{'='*60}")
    print(f" {dataset_name} - ResNet50 Evaluation Results")
    print(f"{'='*60}")
    
    print(f"\n全体精度 (Accuracy): {accuracy*100:.2f}%")
    print(f"サンプル数: {len(y_true)}")
    
    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混同行列:")
    print(f"              予測")
    print(f"             {classes[0]:>12} {classes[1]:>12}")
    print(f"真値 {classes[0]:>8}  {cm[0,0]:>12} {cm[0,1]:>12}")
    print(f"     {classes[1]:>8}  {cm[1,0]:>12} {cm[1,1]:>12}")
    
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
        x_test, y_test, classes = load_test_data(data_path, dataset_name)
        print(f"テストデータ読み込み完了: {len(x_test)} samples")
        print(f"クラス: {classes}")
        
        # 評価
        accuracy, predictions = evaluate(model, x_test, y_test, DEVICE)
        
        # 結果表示
        y_true = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test
        print_evaluation_results(dataset_name, accuracy, y_true, predictions, classes)
        
        # 結果を保存
        results[dataset_name] = {
            'accuracy': accuracy,
            'num_samples': len(x_test),
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
