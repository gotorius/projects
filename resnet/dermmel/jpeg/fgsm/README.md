# DermMel - JPEG圧縮による敵対的防御検証

## 概要

このスクリプトは、DermMel（皮膚悪性黒色腫）データセットに対して、FGSM攻撃とJPEG圧縮による防御性能を評価します。

## 実験設定

### データセット
- **名前**: DermMel (Melanoma分類)
- **クラス**: Melanoma / NotMelanoma (2クラス)
- **データパス**: `/mnt/data1/Public/MedImages/DermMel`
- **検証データ**: `valid/` ディレクトリ

### 攻撃手法
- **手法**: FGSM (Fast Gradient Sign Method)
- **摂動サイズ**: ε = 8/255 (ピクセルスケール)

### 防御手法
- **手法**: JPEG圧縮
- **品質レベル**: 50, 75, 90, 95
  - 50: 低品質（高圧縮率）
  - 75: 標準品質
  - 90: 高品質
  - 95: 最高品質（低圧縮率）

### 分類器
- **モデル**: ResNet50
- **重みパス**: `/mnt/data1/gotou/kaggle/dermmel/resnet50_models/resnet50_best.pth`

## 使用方法

### 基本実行

```bash
cd /mnt/data1/gotou/projects/dermmel/jpeg/fgsm
python fgsm.py
```

### 必要なライブラリ

```bash
pip install torch torchvision pillow pandas numpy matplotlib seaborn scikit-learn tqdm
```

## 出力結果

### ディレクトリ構造

```
dermmel/jpeg/fgsm/defense_results/
├── overall_summary.csv              # 全品質レベルの比較サマリー
├── quality_comparison.png           # 品質vs精度のグラフ
├── quality_50/                      # JPEG品質50の結果
│   ├── summary_statistics.txt
│   ├── detailed_results.csv
│   ├── cm_clean.png                 # 混同行列（クリーン画像）
│   ├── cm_adversarial.png          # 混同行列（敵対的画像）
│   ├── cm_compressed.png           # 混同行列（圧縮画像）
│   ├── triplets/                   # 3枚組画像（クリーン/敵対的/圧縮）
│   ├── clean/                      # クリーン画像
│   ├── adversarial/                # 敵対的画像
│   └── compressed/                 # JPEG圧縮画像
├── quality_75/                      # JPEG品質75の結果
├── quality_90/                      # JPEG品質90の結果
└── quality_95/                      # JPEG品質95の結果
```

### 評価指標

各品質レベルで以下の指標を計算：

1. **精度 (Accuracy)**
   - Clean Accuracy: クリーン画像での精度
   - Adversarial Accuracy: 敵対的画像での精度
   - Compressed Accuracy: JPEG圧縮後の精度
   - Defense Improvement: 防御による精度向上

2. **摂動ノルム**
   - L2ノルム（平均・標準偏差）
   - L∞ノルム（平均・標準偏差）
   - 敵対的画像 vs クリーン画像
   - 圧縮画像 vs クリーン画像

3. **混同行列メトリクス**
   - Precision（適合率）
   - Recall（再現率）
   - F1スコア

4. **防御成功率**
   - Attack Success Rate: 攻撃成功率
   - Defense Success Rate: 防御成功率（攻撃成功例のうち回復した割合）

## スクリプトの主要機能

### 1. FGSM攻撃

```python
adv_images_norm, adv_preds = fgsm_attack_improved(
    model=model,
    images=images_norm_correct,
    labels=labels_correct,
    epsilon_pixel=8/255.0,
    device=device,
    mean_tensor=imagenet_mean,
    std_tensor=imagenet_std,
    return_preds=True
)
```

### 2. JPEG圧縮防御

```python
compressed_images_norm = jpeg_compress_defense(
    adv_images_norm, 
    quality=75,  # JPEG品質
    mean_tensor=imagenet_mean,
    std_tensor=imagenet_std
)
```

### 3. 評価と可視化

- 各品質レベルで混同行列を生成
- トリプレット画像（クリーン/敵対的/圧縮）を保存
- 品質vs精度のグラフを作成

## カスタマイズ

### JPEG品質レベルの変更

スクリプト内の以下の行を編集：

```python
jpeg_qualities = [50, 75, 90, 95]  # 任意の品質レベルを追加
```

### 攻撃強度の変更

```python
epsilon_pixel = 8/255.0  # 摂動サイズを変更（例: 4/255, 16/255）
```

### 保存画像数の変更

```python
MAX_IMAGES_TO_SAVE = 20  # 保存する例示画像の数
```

## 注意事項

1. **データパスの確認**
   - DermMelデータセットが `/mnt/data1/Public/MedImages/DermMel` に配置されていることを確認
   - 分類器の重みが `/mnt/data1/gotou/kaggle/dermmel/resnet50_models/resnet50_best.pth` にあることを確認

2. **メモリ使用量**
   - バッチサイズは32に設定されています
   - GPUメモリが不足する場合は、スクリプト内の `batch_size` を減らしてください

3. **実行時間**
   - 複数の品質レベルをテストするため、実行には時間がかかります
   - 品質レベルを減らすことで実行時間を短縮できます

## 期待される結果

一般的に、以下の傾向が観察されます：

1. **品質が高いほど**（95 > 90 > 75 > 50）
   - 圧縮による画質劣化が少ない
   - クリーン画像の精度が高い
   - 敵対的摂動の除去効果が弱い

2. **品質が低いほど**（50 < 75 < 90 < 95）
   - 圧縮による画質劣化が大きい
   - 敵対的摂動の除去効果が強い
   - クリーン画像の精度も低下する可能性

3. **最適な品質レベル**
   - 防御効果とクリーン精度のトレードオフを考慮
   - 通常、75〜90が良いバランスを示す

## トラブルシューティング

### CUDA out of memory エラー

バッチサイズを減らす：
```python
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
```

### ファイルが見つからないエラー

データパスと分類器のパスを確認：
```python
DATA_DIR = '/mnt/data1/Public/MedImages/DermMel'
clf_ckpt = "/mnt/data1/gotou/kaggle/dermmel/resnet50_models/resnet50_best.pth"
```

## 参考文献

- FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- JPEG圧縮防御: Dziugaite et al., "A Study of the Effect of JPG Compression on Adversarial Images", arXiv 2016

## ライセンス

このスクリプトは研究・教育目的で使用できます。
