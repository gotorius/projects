"""
訓練済みResNet50モデルを読み込むためのユーティリティ
敵対的攻撃実験で使用
"""

import torch
import torch.nn as nn
from torchvision import models

def load_resnet50_for_inference(model_path, device='cuda'):
    """
    推論用ResNet50モデルを読み込む
    
    Args:
        model_path: モデルファイルのパス
        device: 'cuda' or 'cpu'
    
    Returns:
        model: 読み込まれたモデル (eval mode)
        class_names: クラス名のリスト
        class_to_idx: クラス名→インデックスの辞書
    """
    # デバイス設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # モデル構造を作成
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)  # 2クラス分類用
    
    # 重みを読み込み
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 評価モードに設定
    model = model.to(device)
    model.eval()
    
    # クラス情報を取得
    class_names = checkpoint.get('class_names', ['Melanoma', 'NotMelanoma'])
    class_to_idx = checkpoint.get('class_to_idx', {'Melanoma': 0, 'NotMelanoma': 1})
    
    print(f"Model loaded from: {model_path}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Class to index: {class_to_idx}")
    
    return model, class_names, class_to_idx

def load_resnet50_for_training(model_path, device='cuda'):
    """
    訓練再開用にResNet50モデルを読み込む（optimizer, scheduler含む）
    
    Args:
        model_path: モデルファイルのパス
        device: 'cuda' or 'cpu'
    
    Returns:
        model: 読み込まれたモデル
        checkpoint: チェックポイント全体（epoch, optimizer, schedulerなど）
    """
    # デバイス設定
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # モデル構造を作成
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    
    # チェックポイントを読み込み
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    print(f"Model loaded from: {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Validation Accuracy: {checkpoint.get('valid_acc', 'N/A'):.4f}")
    
    return model, checkpoint


# === 使用例 ===
if __name__ == "__main__":
    import os
    
    # 推論用モデルの読み込み例
    model_path = '/mnt/data1/gotou/kaggle/dermmel/resnet50_models/resnet50_inference.pth'
    
    if os.path.exists(model_path):
        print("="*60)
        print("Loading inference model...")
        print("="*60)
        model, class_names, class_to_idx = load_resnet50_for_inference(model_path)
        
        # モデルのサマリー
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model mode: {'eval' if not model.training else 'train'}")
        
        # テスト入力
        dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy_input)
            pred_class = torch.argmax(output, dim=1).item()
            print(f"\nTest inference:")
            print(f"  Input shape: {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Predicted class: {class_names[pred_class]}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first by running: python resnet50.py")
