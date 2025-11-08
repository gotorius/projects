# DermMel - FGSM攻撃 + DDPM防御検証

DermMelデータセット（Melanoma分類）に対するFGSM攻撃と拡散モデル（DDPM）による防御の有効性を検証します。

## 📋 概要

このスクリプトは以下の処理を行います：

1. **クリーン画像の分類**: 訓練済みResNet50で検証データを分類
2. **FGSM攻撃**: 敵対的摂動を加えた画像を生成
3. **DDPM浄化**: 拡散モデルで敵対的画像を浄化
4. **防御性能評価**: 攻撃前後と浄化後の精度を比較
5. **詳細分析**: 混同行列、ノルム分布、統計情報を生成

## 🔧 必要な事前準備

### 1. 分類器の訓練

```bash
cd /mnt/data1/gotou/kaggle/dermmel
python resnet50.py
```

訓練後、以下のファイルが生成されます：
- `resnet50_models/resnet50_inference.pth`

### 2. DDPMの訓練

```bash
cd /mnt/data1/gotou/kaggle/dermmel
python ddpm_train.py
```

訓練後、以下のファイルが生成されます：
- `ddpm_out/ddpm_epoch100.pth`

## 🚀 実行方法

```bash
cd /mnt/data1/gotou/kaggle/dermmel/fgsm
python fgsm.py
```

## ⚙️ 実験パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `epsilon_pixel` | 8/255 ≈ 0.031 | FGSM摂動の大きさ |
| `start_t` | 100 | DDPM拡散開始時刻 |
| `T_purify` | 50 | 逆拡散ステップ数 |
| `batch_size` | 32 | バッチサイズ |
| `MAX_IMAGES_TO_SAVE` | 20 | 保存する例示画像数 |

### パラメータの調整

コード内の以下の部分を変更できます：

```python
# 攻撃の強さを変更
epsilon_pixel = 8/255.0  # 例: 16/255, 32/255

# 浄化の強さを変更
start_t = 100      # より大きい値 = より強力な浄化
T_purify = 50      # より多いステップ = より高品質
```

## 📊 出力ファイル

実行後、`purify_examples/` ディレクトリに以下が生成されます：

### 画像ファイル
```
purify_examples/
├── triplets/                    # クリーン、攻撃、浄化の3枚並べた画像
│   ├── triplet_0000.png
│   ├── triplet_0001.png
│   └── ...
├── clean/                       # クリーン画像
│   ├── clean_0000.png
│   └── ...
├── adversarial/                 # 攻撃画像
│   ├── adv_0000.png
│   └── ...
└── purified/                    # 浄化画像
    ├── purified_0000.png
    └── ...
```

### 統計ファイル
```
purify_examples/
├── cm_clean.png                 # クリーン画像の混同行列
├── cm_adversarial.png           # 攻撃画像の混同行列
├── cm_purified.png              # 浄化画像の混同行列
├── detailed_results.csv         # 全サンプルの詳細データ
└── summary_statistics.txt       # サマリー統計
```

## 📈 評価指標

### 精度指標
- **Clean Accuracy**: クリーン画像に対する精度
- **Adversarial Accuracy**: FGSM攻撃後の精度
- **Purified Accuracy**: DDPM浄化後の精度
- **Defense Improvement**: 浄化による精度向上

### ノルム指標
- **L2 Norm**: ユークリッド距離（画像全体の変化量）
- **L∞ Norm**: 最大ピクセル差（最も変化したピクセル）

## 📝 結果の解釈

### 成功例
```
Clean accuracy:     1.0000 (3562/3562)
Adv (FGSM) accuracy:0.1234 (440/3562)
Purified accuracy:  0.8567 (3052/3562)
Defense improvement: +0.7333
```

**解釈**:
- FGSM攻撃により精度が87.66%低下
- DDPM浄化により73.33%回復
- 防御成功率: 約85.7%

### 混同行列の見方

```
              Predicted
           Melanoma | NotMelanoma
True  
Melanoma      TP   |     FN
NotMelanoma   FP   |     TN
```

- **TP (True Positive)**: Melanomaを正しく検出
- **FN (False Negative)**: Melanomaの見逃し（重要！）
- **FP (False Positive)**: 誤検出
- **TN (True Negative)**: NotMelanomaの正しい判定

## 🔍 トラブルシューティング

### モデルファイルが見つからない

```python
FileNotFoundError: [Errno 2] No such file or directory: 
  '/mnt/data1/gotou/kaggle/dermmel/resnet50_models/resnet50_inference.pth'
```

**解決策**: 先に分類器を訓練してください
```bash
cd /mnt/data1/gotou/kaggle/dermmel
python resnet50.py
```

### DDPMファイルが見つからない

```python
FileNotFoundError: '/mnt/data1/gotou/kaggle/dermmel/ddpm_out/ddpm_epoch100.pth'
```

**解決策**: 
1. DDPMを訓練: `python ddpm_train.py`
2. または、別のエポックを指定:
```python
ddpm_ckpt = "/mnt/data1/gotou/kaggle/dermmel/ddpm_out/ddpm_epoch50.pth"
```

### CUDA out of memory

**解決策**: バッチサイズを減らす
```python
val_loader = DataLoader(val_dataset, batch_size=16, ...)  # 32 → 16
```

### 浄化効果が低い

**解決策**: パラメータを調整
```python
start_t = 200      # 100 → 200 (より強力な浄化)
T_purify = 100     # 50 → 100 (より多いステップ)
```

## 🧪 実験のバリエーション

### 1. より強い攻撃

```python
# より大きなεで攻撃
epsilon_pixel = 16/255.0  # 8/255 → 16/255
```

### 2. より弱い浄化

```python
# 軽微な浄化（処理時間短縮）
start_t = 50
T_purify = 25
```

### 3. 全検証データで評価

デフォルトで全検証データ（3,562枚）を使用しています。
サブセットで高速テストする場合：

```python
# 最初の100サンプルのみ
from torch.utils.data import Subset
val_dataset_subset = Subset(val_dataset, range(100))
val_loader = DataLoader(val_dataset_subset, batch_size=32, ...)
```

## 📚 参考

### FGSM攻撃
- Goodfellow et al. (2015), "Explaining and Harnessing Adversarial Examples"
- 勾配の符号を使った高速な攻撃手法
- ε = 8/255 が標準的な設定

### DDPM防御
- Ho et al. (2020), "Denoising Diffusion Probabilistic Models"
- Nie et al. (2022), "Diffusion Models for Adversarial Purification"
- 逐次的なデノイジングで摂動を除去

## 🎯 期待される結果

| 指標 | 典型的な値 |
|-----|-----------|
| Clean Accuracy | 92-95% |
| Adversarial Accuracy (ε=8/255) | 10-20% |
| Purified Accuracy | 75-85% |
| Defense Improvement | +60-70% |
| L2 Norm (Adversarial) | 0.03-0.05 |
| L∞ Norm (Adversarial) | 0.031 (=ε) |

**注**: 実際の値はモデルの訓練状況により異なります。

## 💡 次のステップ

1. **PGD攻撃**: より強力な反復型攻撃
2. **AutoAttack**: 最強の攻撃手法
3. **他の防御手法**: JPEG圧縮、GAN、VAEと比較
4. **パラメータ最適化**: start_tとT_purifyの最適値探索

---

**作成日**: 2025年11月8日
