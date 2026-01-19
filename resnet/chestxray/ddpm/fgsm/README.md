# ChestXray FGSM攻撃 + DDPM防御 検証

このディレクトリには、ChestXray（胸部X線画像による肺炎分類）データセットに対するFGSM攻撃とDDPM拡散モデルによる防御の検証コードが含まれています。

## 📋 概要

**目的**: 訓練済みResNet50分類器に対してFGSM攻撃を行い、DDPM拡散モデルによる画像浄化で防御性能を評価

**データセット**: ChestXray (NORMAL vs PNEUMONIA)
- テストセット: 624画像
- 2クラス分類 (0=NORMAL, 1=PNEUMONIA)

**攻撃手法**: FGSM (Fast Gradient Sign Method)
- ε = 8/255 (標準設定)
- 正規化空間でのチャネルごとの適切な摂動計算

**防御手法**: DDPM (Denoising Diffusion Probabilistic Models)
- 部分的forward diffusion (t=0 → t=100)
- 逆拡散による画像浄化 (50ステップ)
- [-1,1]正規化での拡散処理

## 🔧 必要なモデルファイル

実行前に以下のファイルが必要です:

```bash
# ResNet50分類器 (最高精度モデル)
/mnt/data1/gotou/kaggle/chestxray/resnet50_models/resnet50_best.pth

# DDPM拡散モデル (エポック100)
/mnt/data1/gotou/kaggle/chestxray/ddpm_out/ddpm_epoch100.pth
```

## 🚀 実行方法

### 基本的な実行

```bash
cd /mnt/data1/gotou/kaggle/chestxray/fgsm
python fgsm.py
```

### 期待される実行時間

- テストデータ: 624画像
- GPU使用時: 約5〜10分
- CPU使用時: 約30〜60分

## 📊 出力結果

### 1. ディレクトリ構造

```
purify_examples/
├── triplets/          # クリーン・敵対的・浄化画像の比較 (20枚)
├── clean/             # クリーン画像 (20枚)
├── adversarial/       # FGSM攻撃画像 (20枚)
├── purified/          # DDPM浄化画像 (20枚)
├── cm_clean.png       # クリーン画像の混同行列
├── cm_adversarial.png # 敵対的画像の混同行列
├── cm_purified.png    # 浄化画像の混同行列
├── detailed_results.csv        # 全サンプルの詳細結果
└── summary_statistics.txt      # サマリー統計
```

### 2. 評価指標

**精度 (Accuracy)**:
- Clean Accuracy: クリーン画像での分類精度
- Adversarial Accuracy: FGSM攻撃後の分類精度
- Purified Accuracy: DDPM浄化後の分類精度
- Defense Improvement: 浄化による精度向上 (Purified - Adversarial)

**摂動ノルム**:
- L2ノルム: ピクセル空間での平均二乗誤差
- L∞ノルム: ピクセル空間での最大絶対誤差

**混同行列の指標**:
- Precision, Recall, F1スコア
- TN, FP, FN, TP

### 3. CSVファイル詳細

`detailed_results.csv` には以下の列が含まれます:

| 列名 | 説明 |
|------|------|
| true_label | 真のラベル (0=NORMAL, 1=PNEUMONIA) |
| pred_clean | クリーン画像の予測 |
| pred_adv | 敵対的画像の予測 |
| pred_purified | 浄化画像の予測 |
| l2_norm_adv | 敵対的摂動のL2ノルム |
| linf_norm_adv | 敵対的摂動のL∞ノルム |
| l2_norm_purified | 浄化による変化のL2ノルム |
| linf_norm_purified | 浄化による変化のL∞ノルム |
| attack_success | 攻撃成功フラグ (1=誤分類) |
| purify_success | 浄化成功フラグ (1=正分類) |
| defense_recovery | 防御成功フラグ (攻撃成功→浄化成功) |

## 🎯 期待される結果

医療画像データセット（ChestXray）での典型的な結果:

| 指標 | 期待値 |
|------|--------|
| Clean Accuracy | 0.90〜0.95 |
| Adversarial Accuracy (FGSM) | 0.20〜0.40 |
| Purified Accuracy | 0.60〜0.80 |
| Defense Improvement | +0.30〜+0.50 |
| Attack Success Rate | 0.60〜0.80 |
| Defense Success Rate | 0.50〜0.70 |

**注意**: X線画像の特性（高コントラスト、解剖学的構造）により、攻撃の影響や防御効果はDermMelとは異なる可能性があります。

## 🔍 パラメータのカスタマイズ

スクリプト内で以下のパラメータを変更可能:

```python
# FGSM攻撃の強度
epsilon_pixel = 8/255.0  # 標準: 8/255

# DDPM浄化パラメータ
start_t = 100      # 拡散開始時刻 (推奨: 50〜200)
T_purify = 50      # 逆拡散ステップ数 (推奨: 30〜100)

# 保存する画像数
MAX_IMAGES_TO_SAVE = 20  # トリプレット画像の数
```

## 📖 参考文献

**FGSM攻撃**:
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
- Paper: https://arxiv.org/abs/1412.6572

**DDPM拡散モデル**:
- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
- Paper: https://arxiv.org/abs/2006.11239

**拡散モデルによる防御**:
- Nie et al., "Diffusion Models for Adversarial Purification", ICML 2022
- Paper: https://arxiv.org/abs/2205.07460

## ⚠️ トラブルシューティング

### GPU メモリ不足

```python
# バッチサイズを減らす
test_loader = DataLoader(test_dataset, batch_size=16, ...)  # 32→16
```

### モデルファイルが見つからない

```bash
# ResNet50モデルの確認
ls -lh /mnt/data1/gotou/kaggle/chestxray/resnet50_models/resnet50_best.pth

# DDPMモデルの確認
ls -lh /mnt/data1/gotou/kaggle/chestxray/ddpm_out/ddpm_epoch100.pth
```

### データセットが見つからない

```bash
# ChestXrayデータの確認
ls /mnt/data1/Public/MedImages/CellData/chest_xray/test/
# 期待される出力: NORMAL  PNEUMONIA
```

## 📝 備考

- **医療画像の特性**: 胸部X線画像は解剖学的構造が重要なため、攻撃による構造の歪みや浄化による復元の様子を視覚的に確認することが重要です
- **クラス不均衡**: ChestXrayデータセットはPNEUMONIA画像が多いため、混同行列での詳細分析が有用です
- **正解画像のみ評価**: 元々正しく分類された画像のみを対象に攻撃と防御を評価しています
- **再現性**: 乱数シードは固定されていないため、実行ごとに若干の変動があります

## 🔗 関連ファイル

- `../resnet50.py`: ResNet50分類器の訓練スクリプト
- `../ddpm_train.py`: DDPM拡散モデルの訓練スクリプト
- `../README_resnet50.md`: ResNet50分類器の詳細ドキュメント
- `/mnt/data1/gotou/kaggle/README`: プロジェクト全体のREADME
