# ChestXray ResNet50 分類モデル訓練

ChestXrayデータセット（NORMAL vs PNEUMONIA）でResNet50を訓練し、敵対的攻撃実験用のモデルを保存します。

## ディレクトリ構成

```
/mnt/data1/Public/MedImages/CellData/chest_xray/
├── train/
│   ├── NORMAL/        (1,341枚)
│   └── PNEUMONIA/     (3,891枚)
└── test/
    ├── NORMAL/        (234枚)
    └── PNEUMONIA/     (390枚)
```

**注意**: 訓練データのみ提供されているため、訓練データの15%を検証データとして分割します。

## データ統計

- **訓練データ合計**: 5,232枚
  - 訓練用: 4,448枚 (85%)
  - 検証用: 784枚 (15%)
- **テストデータ**: 624枚
- **クラス**: NORMAL (0), PNEUMONIA (1)
- **クラス不均衡**: PNEUMONIA が約75%を占める

## ファイル説明

- **resnet50.py**: ResNet50訓練スクリプト（メイン）
- **load_model.py**: 訓練済みモデルを読み込むユーティリティ
- **README_resnet50.md**: このファイル

## 使用方法

### 1. モデルの訓練

```bash
cd /mnt/data1/gotou/kaggle/chestxray
python resnet50.py
```

訓練が完了すると、以下のファイルが `resnet50_models/` に保存されます：

- `resnet50_best.pth`: 最良の検証精度を達成したモデル（完全版）
- `resnet50_inference.pth`: 推論専用モデル（敵対的攻撃実験用）
- `resnet50_epoch{N}.pth`: 定期チェックポイント（10エポックごと）
- `training_history.npz`: 訓練履歴（loss, accuracy）

### 2. モデルの読み込み（敵対的攻撃実験用）

```python
from load_model import load_resnet50_for_inference

# モデルを読み込み
model_path = 'resnet50_models/resnet50_inference.pth'
model, class_names, class_to_idx = load_resnet50_for_inference(model_path)

# 推論
import torch
from torchvision import transforms
from PIL import Image

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 画像を読み込んで予測
image = Image.open('chest_xray.jpeg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)
input_tensor = input_tensor.to(next(model.parameters()).device)

with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    print(f"Predicted: {class_names[pred_class]}")
```

### 3. 訓練の再開

```python
from load_model import load_resnet50_for_training

model, checkpoint = load_resnet50_for_training('resnet50_models/resnet50_best.pth')

# optimizerとschedulerを復元
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

start_epoch = checkpoint['epoch'] + 1
# 訓練を続ける...
```

## ハイパーパラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| IMAGE_SIZE | 224 | 入力画像サイズ |
| BATCH_SIZE | 32 | バッチサイズ |
| EPOCHS | 50 | 訓練エポック数 |
| LEARNING_RATE | 1e-4 | 初期学習率 |
| VALIDATION_SPLIT | 0.15 | 検証データ分割率 |
| Optimizer | Adam | 最適化アルゴリズム |
| Scheduler | ReduceLROnPlateau | 学習率スケジューラ |

## データ拡張

### 訓練データ
- Resize → CenterCrop
- RandomRotation (±10度) ← 軽微な回転のみ
- ColorJitter (brightness, contrast) ← 軽微な調整
- ImageNet正規化
- **注意**: X線画像では左右の位置情報が医学的に重要なため、HorizontalFlip/VerticalFlipは除外

### 検証・テストデータ
- Resize (224×224)
- ImageNet正規化のみ

## モデル構造

- **ベースモデル**: ResNet50 (ImageNet事前訓練済み)
- **最終層**: Linear(2048 → 2) ← 2クラス分類用に変更
- **総パラメータ数**: 約23.5M

## クラス不均衡への対応

ChestXrayデータセットはPNEUMONIAクラスが約75%を占めるため、以下の対策を検討できます：

### オプション1: クラスウェイト
```python
# resnet50.py の criterion を変更
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# クラスウェイトを計算
labels = [label for _, label in full_train_dataset]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# 損失関数にウェイトを適用
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### オプション2: リサンプリング
```python
from torch.utils.data import WeightedRandomSampler

# サンプルウェイトを計算
targets = [label for _, label in train_dataset]
class_counts = np.bincount(targets)
sample_weights = [1.0 / class_counts[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoaderでsamplerを使用
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, ...)
```

現在のコードはクラスウェイトなしですが、必要に応じて上記を追加できます。

## 保存されるモデル情報

チェックポイントには以下の情報が含まれます：

```python
{
    'epoch': int,                    # エポック番号
    'model_state_dict': OrderedDict, # モデルの重み
    'optimizer_state_dict': dict,    # optimizer状態
    'scheduler_state_dict': dict,    # scheduler状態
    'valid_acc': float,              # 検証精度
    'valid_loss': float,             # 検証損失
    'class_names': list,             # ['NORMAL', 'PNEUMONIA']
    'class_to_idx': dict,            # {'NORMAL': 0, 'PNEUMONIA': 1}
}
```

## 敵対的攻撃実験での使用

訓練済みモデルは以下の敵対的攻撃手法の評価に使用できます：

- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- C&W (Carlini & Wagner)
- DeepFool
- など

### 攻撃コード例（FGSM）

```python
import torch
import torch.nn.functional as F
from load_model import load_resnet50_for_inference

# モデル読み込み
model, _, _ = load_resnet50_for_inference('resnet50_models/resnet50_inference.pth')
model.eval()

# FGSM攻撃
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

# 入力画像と正解ラベル
input_tensor = ...  # shape: (1, 3, 224, 224)
target = ...        # shape: (1,)

# 勾配を有効化
input_tensor.requires_grad = True

# Forward
output = model(input_tensor)
loss = F.cross_entropy(output, target)

# Backward
model.zero_grad()
loss.backward()

# 攻撃画像を生成
perturbed_image = fgsm_attack(input_tensor, epsilon=0.03, data_grad=input_tensor.grad)

# 攻撃後の予測
with torch.no_grad():
    adv_output = model(perturbed_image)
    adv_pred = torch.argmax(adv_output, dim=1)
```

## 訓練時のTips

1. **GPUメモリ不足の場合**: `BATCH_SIZE`を16や8に減らす
2. **クラス不均衡が問題の場合**: クラスウェイトまたはリサンプリングを追加
3. **収束が遅い場合**: 学習率を調整（1e-3など）
4. **過学習の場合**: データ拡張を強化、またはDropoutを追加

## 評価指標

訓練中に以下の指標が記録されます：

- Training Loss
- Training Accuracy
- Validation Loss
- Validation Accuracy
- Test Accuracy (訓練後)

最良の検証精度を達成したモデルが自動的に保存されます。

## トラブルシューティング

### モデルが読み込めない
```python
# デバイスを明示的に指定
model, _, _ = load_resnet50_for_inference(model_path, device='cpu')
```

### CUDA out of memory
```python
# resnet50.py の BATCH_SIZE を減らす
BATCH_SIZE = 16  # または 8
```

### 訓練が進まない
- 学習率が小さすぎる可能性 → LEARNING_RATE = 1e-3 を試す
- データが正しく読み込まれているか確認 → print(full_train_dataset.class_to_idx)

## 医学画像の注意点

1. **左右反転の除外**: X線画像では心臓の位置や肺の形状が重要なため、HorizontalFlipは使用しません
2. **軽微なデータ拡張**: 過度な変換は医学的特徴を損なう可能性があるため、控えめに設定
3. **クラス不均衡**: 実際の医療現場でも疾患の有病率は偏るため、実用性を考慮した評価が重要

## ライセンス・引用

このコードを研究に使用する場合は、ChestXrayデータセットの出典を明記してください。

**データセット出典**:
- Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, V2
