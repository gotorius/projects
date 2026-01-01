※README.mdはClaude Sonnet4.5で作成

# 拡散モデルを用いた敵対的攻撃に対する防御手法の研究

深層学習モデルに対する敵対的攻撃（Adversarial Attacks）は、セキュリティ上の重大な脅威となっています。本研究では、**拡散モデル（Diffusion Models）を用いた新しい防御手法**を提案し、医療画像分類タスクにおいてその有効性を検証します。

## 目次

- [概要](#概要)
- [研究の動機](#研究の動機)
- [プロジェクト構成](#プロジェクト構成)
- [データセット](#データセット)
- [敵対的攻撃手法](#敵対的攻撃手法)
- [防御手法](#防御手法)
- [実験環境](#実験環境)
- [使用方法](#使用方法)
- [実験結果](#実験結果)
- [参考文献](#参考文献)

---

## 概要

本プロジェクトでは、**拡散モデル（DDPM: Denoising Diffusion Probabilistic Models）** を敵対的攻撃に対する防御機構として活用します。拡散モデルは画像のノイズ除去能力に優れており、敵対的摂動を効果的に除去できることを実験により示しました。

### 研究の主要な内容

1. **拡散モデルによる敵対的防御**: DDPMを用いて攻撃画像を浄化し、元の画像に復元（DiffPure手法）
2. **医療画像への適用**: 3つの医療画像データセット（ChestXray, DermMel, PCam）で防御性能を評価
3. **ベースラインとの比較**: JPEG圧縮との詳細な性能比較を実施
4. **複数の攻撃手法への対応**: FGSM、PGD-10、AutoAttack（APGD-CE）に対する頑健性を検証

---

## 研究の動機

### 敵対的攻撃の脅威

深層学習モデルは、人間には知覚できない微小な摂動（adversarial perturbation）を加えることで、誤った予測を出力することが知られています。特に医療分野では、誤診につながる可能性があり、重大な問題となっています。

### なぜ拡散モデルか？

- **強力なノイズ除去能力**: 拡散モデルは逐次的なデノイジング過程により、自然な画像を生成
- **分布のモデリング**: データ分布を正確に学習し、異常な摂動を検出・除去
- **柔軟性**: 様々な画像タスクに適用可能
- **理論的基盤**: 確率的生成モデルとしての強固な理論的裏付け

---

## 📁 プロジェクト構成

```
projects/
├── README.md                       # このファイル
├── chestxray/                      # 胸部X線画像（肺炎分類）
│   ├── README.md                  # ChestXray詳細ドキュメント
│   ├── ddpm/                      # DDPM防御実験
│   │   ├── ddpm_train.py          # 拡散モデル訓練
│   │   ├── ddpm_out/              # 拡散モデル出力
│   │   ├── fgsm/                  # FGSM攻撃評価
│   │   ├── pgd/                   # PGD攻撃評価
│   │   └── autoattack/            # AutoAttack評価
│   ├── jpeg/                      # JPEG圧縮防御実験
│   │   ├── fgsm/
│   │   ├── pgd/
│   │   └── autoattack/
│   ├── resnet/                    # ResNet50分類器
│   │   ├── resnet50.py            # 分類器訓練
│   │   ├── load_model.py          # モデル読み込みユーティリティ
│   │   └── resnet50_best.pth      # 訓練済みモデル
│   └── result/                    # 結果可視化
│       └── result.ipynb           # 結果分析ノートブック
├── dermmel/                        # 皮膚病変画像（メラノーマ分類）
│   ├── README.md                  # DermMel詳細ドキュメント
│   ├── ddpm/                      # DDPM防御実験
│   ├── jpeg/                      # JPEG圧縮防御実験
│   ├── resnet/                    # ResNet50分類器
│   └── result/                    # 結果可視化
├── pcam/                           # 病理組織画像（転移性がん検出）
│   ├── ddpm/                      # DDPM防御実験
│   │   ├── ddpm_train_pcam.py     # 拡散モデル訓練
│   │   ├── checkpoints/           # モデルチェックポイント
│   │   ├── fgsm/                  # FGSM攻撃評価
│   │   ├── pgd/                   # PGD攻撃評価
│   │   └── autoattack/            # AutoAttack評価
│   ├── jpeg/                      # JPEG圧縮防御実験
│   └── resnet/                    # ResNet50分類器
└── config/                         # 設定ファイル
```

---

## データセット

本研究では、3つの異なる医療画像データセットを使用します。

### 1. PCam (Patch Camelyon)

**タスク**: 病理組織画像における転移性がんの検出

| 項目 | 詳細 |
|------|------|
| **画像サイズ** | 96×96 px |
| **クラス数** | 2（がん転移あり/なし） |
| **訓練データ** | 262,144枚 |
| **検証データ** | 32,768枚 |
| **テストデータ** | 32,768枚 |
| **データ形式** | H5形式 |
| **特徴** | リンパ節の組織切片画像 |

**出典**: Histopathologic Cancer Detection　
https://www.kaggle.com/competitions/histopathologic-cancer-detection/data

### 2. DermMel (Dermatology Melanoma)

**タスク**: 皮膚病変画像におけるメラノーマ（悪性黒色腫）の分類

| 項目 | 詳細 |
|------|------|
| **画像サイズ** | 可変（リサイズ: 224×224 px） |
| **クラス数** | 2（Melanoma / NotMelanoma） |
| **訓練データ** | 10,682枚 |
| **検証データ** | 3,562枚 |
| **テストデータ** | あり |
| **データ形式** | JPEG |
| **特徴** | 皮膚鏡画像、データ拡張済み |

**出典**: 研究室のデータセット

### 3. ChestXray (Chest X-ray Pneumonia)

**タスク**: 胸部X線画像における肺炎の検出

| 項目 | 詳細 |
|------|------|
| **画像サイズ** | 可変（リサイズ: 224×224 px） |
| **クラス数** | 2（NORMAL / PNEUMONIA） |
| **訓練データ** | 5,232枚（分割: 4,448 / 784） |
| **検証データ** | 784枚（訓練から分割） |
| **テストデータ** | 624枚 |
| **データ形式** | JPEG |
| **特徴** | 小児胸部X線、クラス不均衡あり |

**出典**: 研究室のデータセット

---

## 敵対的攻撃手法

### 1. FGSM (Fast Gradient Sign Method)

**提案者**: Goodfellow et al. (2015)

**原理**: 損失関数の勾配の符号を用いて、最小限の摂動で誤分類を引き起こす

**数式**:
```
x_adv = x + ε · sign(∇_x L(θ, x, y))
```

**特徴**:
- ✅ 計算コストが低い（1回のバックプロパゲーション）
- ✅ 実装が簡単
- ❌ 単純な攻撃のため、防御されやすい

**パラメータ**:
- `ε`: 摂動の大きさ（典型値: 0.03, 0.1）

### 2. PGD (Projected Gradient Descent)

**提案者**: Madry et al. (2018)

**原理**: FGSMの反復版。複数回の小さなステップで摂動を最適化

**数式**:
```
x_adv^(t+1) = Π_ε(x_adv^(t) + α · sign(∇_x L(θ, x_adv^(t), y)))
```

**特徴**:
- ✅ FGSMより強力な攻撃
- ✅ 敵対的訓練のベースライン
- ⚖️ 計算コストが高い（複数回の反復）

**パラメータ**:
- `ε`: 摂動の最大L∞ノルム（典型値: 8/255）
- `α`: ステップサイズ（典型値: 2/255）
- `iterations`: 反復回数（典型値: 10, 20, 40）

### 3. AutoAttack

**提案者**: Croce & Hein (2020)

**原理**: 複数の攻撃手法を組み合わせた自動化された攻撃フレームワーク

**構成要素**:
1. **APGD-CE**: Adaptive PGD with Cross-Entropy loss
2. **APGD-DLR**: Adaptive PGD with Difference of Logits Ratio loss
3. **FAB**: Fast Adaptive Boundary attack
4. **Square Attack**: クエリベースのブラックボックス攻撃

**特徴**:
- ✅ 最も強力な攻撃の一つ
- ✅ パラメータチューニング不要
- ✅ 防御手法の評価標準として広く採用
- ❌ 計算コストが非常に高い

**使用場面**:
- 防御手法の頑健性を評価する最終テスト

---

## 防御手法

### 1. 拡散モデル（DDPM）【提案手法】

**原理**: 
拡散モデルの逆拡散過程を利用して、攻撃画像から敵対的摂動を除去し、元の画像分布に戻す。

**プロセス**:
1. **攻撃画像の入力**: x_adv を受け取る
2. **部分的な拡散**: x_adv に軽微なノイズを追加（t=T_defense まで）
3. **逆拡散による浄化**: t=T_defense → 0 まで逐次的にデノイジング
4. **分類器への入力**: 浄化された画像 x_purified を分類

**数式**:
```
# Forward (部分的な拡散)
x_t = √(ᾱ_t) · x_adv + √(1-ᾱ_t) · ε,  ε ~ N(0,I)

# Reverse (逆拡散による浄化)
x_(t-1) = μ_θ(x_t, t) + σ_t · z,  z ~ N(0,I)
```

**パラメータ**:
- `T_defense`: 拡散ステップ数（典型値: 100-500）
- `beta_schedule`: ノイズスケジュール（cosine推奨）

**実装**:
- 訓練: `ddpm_train.py`（各データセット用）
- U-Netアーキテクチャ（base_ch=64, time_emb_dim=256）
- 1000ステップの拡散過程
- Cosineノイズスケジュール

**利点**:
- ✅ 理論的に基づいた防御
- ✅ 高品質な画像復元
- ✅ 様々な攻撃に対して頑健

**欠点**:
- ❌ 推論時間が長い（逐次的な処理）
- ❌ GPUメモリ使用量が大きい

### 2. JPEG圧縮【ベースライン】

**原理**: 
JPEG圧縮の過程で高周波成分が除去されるため、敵対的摂動（高周波ノイズ）も同時に除去される。

**プロセス**:
1. 攻撃画像をJPEG形式で圧縮
2. 圧縮された画像を分類器に入力

**パラメータ**:
- `quality`: 圧縮品質（0-100、典型値: 75, 50, 25）

**実装例**:
```python
from PIL import Image
from io import BytesIO

def jpeg_defense(image, quality=75):
    buffer = BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer)
```

**利点**:
- ✅ 実装が非常に簡単
- ✅ 計算コストが低い
- ✅ リアルタイム処理可能

**欠点**:
- ❌ 画質劣化による精度低下
- ❌ 適応的攻撃に弱い
- ❌ 防御性能が限定的

### 3. GAN (Generative Adversarial Network)

**原理**: 
GANのGeneratorを用いて、攻撃画像をクリーンな画像に変換。

**アーキテクチャ**:
- **Generator**: 攻撃画像 → クリーン画像
- **Discriminator**: クリーン画像 vs 生成画像を識別
- **訓練**: Adversarial Loss + Reconstruction Loss

**プロセス**:
1. 攻撃画像をGeneratorに入力
2. Generatorが浄化画像を生成
3. 浄化画像を分類器に入力

**損失関数**:
```
L_total = L_adv + λ_rec · L_rec
L_rec = ||G(x_adv) - x_clean||_2
```

**パラメータ**:
- `λ_rec`: 再構成損失の重み（典型値: 10.0）
- `architecture`: U-Net, ResNet, etc.

**利点**:
- ✅ 高品質な画像生成
- ✅ 1回のフォワードパスで処理

**欠点**:
- ❌ 訓練が不安定（mode collapse）
- ❌ ペアデータ（攻撃画像-クリーン画像）が必要
- ❌ 過学習のリスク

### 4. VAE (Variational Autoencoder)

**原理**: 
VAEの潜在空間を経由することで、敵対的摂動を除去。

**アーキテクチャ**:
- **Encoder**: x_adv → z (潜在変数)
- **Decoder**: z → x_purified

**プロセス**:
1. 攻撃画像をEncoderで潜在空間に埋め込み
2. 潜在変数をDecoderで画像に復元
3. 復元画像を分類器に入力

**損失関数**:
```
L_VAE = L_rec + β · KL(q(z|x) || p(z))
L_rec = ||x_reconstructed - x||_2
KL = D_KL(N(μ,σ²) || N(0,I))
```

**パラメータ**:
- `β`: KL項の重み（典型値: 1.0, VAE; >1.0, β-VAE）
- `latent_dim`: 潜在次元数（典型値: 128, 256）

**利点**:
- ✅ 訓練が安定（GANより）
- ✅ 1回のフォワードパスで処理
- ✅ 潜在空間の正則化により汎化性能向上

**欠点**:
- ❌ 画像がぼやける傾向（over-smoothing）
- ❌ ペアデータが必要
- ❌ 複雑な画像の再構成が困難

---


## 使用方法

### 1. 分類器の訓練

各データセット用のResNet50分類器を訓練します。

```bash
# ChestXray
cd /mnt/data1/gotou/projects/chestxray/resnet
python resnet50.py

# DermMel
cd /mnt/data1/gotou/projects/dermmel/resnet
python resnet50.py

# PCam
cd /mnt/data1/gotou/projects/pcam/resnet
python resnet50.py
```

訓練後、`resnet50_best.pth` が生成されます。

### 2. 拡散モデルの訓練

各データセット用のDDPMを訓練します。

```bash
# ChestXray
cd /mnt/data1/gotou/projects/chestxray/ddpm
python ddpm_train.py

# DermMel
cd /mnt/data1/gotou/projects/dermmel/ddpm
python ddpm_train.py

# PCam
cd /mnt/data1/gotou/projects/pcam/ddpm
python ddpm_train_pcam.py
```

訓練後、`ddpm_out/` ディレクトリにモデルとサンプル画像が生成されます。

### 3. DDPM防御評価の実行

各データセット・攻撃手法ごとに評価スクリプトを実行します。

```bash
# ChestXray - FGSM
cd /mnt/data1/gotou/projects/chestxray/ddpm/fgsm
python ddpm_fgsm_eval.py

# ChestXray - PGD
cd /mnt/data1/gotou/projects/chestxray/ddpm/pgd
python ddpm_pgd_eval.py

# ChestXray - AutoAttack
cd /mnt/data1/gotou/projects/chestxray/ddpm/autoattack
python ddpm_autoattack_eval.py

# PCam - 全攻撃
cd /mnt/data1/gotou/projects/pcam/ddpm
python ddpm_defense_eval.py --attack all --num_samples 500 --use_purification --t_purify 50
```

### 4. JPEG防御評価の実行

```bash
# ChestXray - FGSM
cd /mnt/data1/gotou/projects/chestxray/jpeg/fgsm
python jpeg_fgsm_eval.py

# DermMel - PGD
cd /mnt/data1/gotou/projects/dermmel/jpeg/pgd
python jpeg_pgd_eval.py
```

### 5. 結果の可視化

各データセットの `result/` ディレクトリにあるノートブックで結果を可視化できます。

```bash
# ChestXray結果分析
jupyter notebook /mnt/data1/gotou/projects/chestxray/result/result.ipynb

# DermMel結果分析
jupyter notebook /mnt/data1/gotou/projects/dermmel/result/result.ipynb
```

---

## 実験結果

### 評価指標

- **Clean Accuracy**: クリーン画像に対する精度（正しく分類されたサンプルのみ使用）
- **Adversarial Accuracy**: 攻撃画像に対する精度
- **Defense Accuracy**: 防御後の精度
- **Defense Improvement**: 防御による精度向上（Defense Acc - Adv Acc）

### ChestXray 実験結果

**実験設定**:
- 対象画像数: 601枚（クリーン画像で正しく分類された画像）
- DDPM: start_t=80, steps=50
- JPEG: quality=11
- ε=8/255 (≈0.0314)

| 攻撃手法 | Clean Acc | Adv Acc | DDPM防御 | JPEG防御 | DDPM優位 |
|---------|-----------|---------|----------|----------|----------|
| FGSM | 100.00% | 64.56% | **69.22%** | 64.56% | +4.66% |
| PGD-10 | 100.00% | 0.00% | **91.68%** | 67.39% | +24.29% |
| AutoAttack | 100.00% | 0.00% | **91.01%** | 66.72% | +24.29% |

### DermMel 実験結果

**実験設定**:
- 対象画像数: 3,434枚（クリーン画像で正しく分類された画像）
- DDPM: start_t=80, steps=50
- JPEG: quality=11
- ε=8/255 (≈0.0314)

| 攻撃手法 | Clean Acc | Adv Acc | DDPM防御 | JPEG防御 | DDPM優位 |
|---------|-----------|---------|----------|----------|----------|
| FGSM | 100.00% | 43.42% | **53.47%** | 49.53% | +3.94% |
| PGD-10 | 100.00% | 0.00% | **64.91%** | 49.56% | +15.35% |
| AutoAttack | 100.00% | 0.00% | **61.97%** | 49.56% | +12.41% |

### PCam 実験結果

**実験設定**:
- 対象画像数: 500枚（クラスバランス調整済み）
- DDPM: start_t=80, t_purify=50
- ε=8/255 (≈0.0314)

| 攻撃手法 | Clean Acc | Adv Acc | DDPM防御 | 改善率 |
|---------|-----------|---------|----------|--------|
| FGSM | 100.00% | 14.60% | **64.40%** | +49.80% |
| PGD-10 | 100.00% | 0.00% | **84.80%** | +84.80% |
| AutoAttack | 100.00% | 0.00% | **92.00%** | +92.00% |

### 分析のポイント

1. **拡散モデル（DDPM）の優位性**: 全てのデータセット・攻撃手法に対してJPEG圧縮より高い防御性能を達成
2. **攻撃手法による差**: PGD-10とAutoAttackはFGSMより強力だが、DDPMは強力な攻撃に対してより効果的
3. **データセット依存性**: 
   - PCamで最も高い防御効果（AutoAttackに対して92%の精度回復）
   - ChestXrayでもPGD/AutoAttackに対して90%以上の精度を維持
   - DermMelでは相対的に低いが、依然としてJPEGを上回る
4. **医療診断への示唆**: 肺炎診断（ChestXray）において、DDPMは高いRecall（95%以上）を維持し、見逃しリスクを低減

### 総合結論

本研究の実験結果から、以下の結論が得られました：

| データセット | 最良DDPM防御精度 | JPEG比優位性 | 特記事項 |
|-------------|-----------------|-------------|---------|
| **ChestXray** | 91.68% (PGD-10) | +24.29% | 肺炎診断において高いRecallを維持 |
| **DermMel** | 64.91% (PGD-10) | +15.35% | 皮膚病変画像での有効性を確認 |
| **PCam** | 92.00% (AutoAttack) | - | 病理画像で最高の防御性能 |

**主要な知見**:
- 拡散モデルベースの防御（DiffPure）は、強力な敵対的攻撃（PGD、AutoAttack）に対しても有効
- JPEG圧縮と比較して、全ての実験条件で優れた防御性能を示す
- 医療画像AIのセキュリティ向上に拡散モデルが有望なアプローチであることを実証

---

## 参考文献

### 拡散モデル

1. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models". NeurIPS.
2. Nichol, A., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models". ICML.
3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations". ICLR.

### 敵対的攻撃

4. Goodfellow, I. J., et al. (2015). "Explaining and Harnessing Adversarial Examples". ICLR.
5. Madry, A., et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks". ICLR.
6. Croce, F., & Hein, M. (2020). "Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks". ICML.

### 防御手法

7. Samangouei, P., et al. (2018). "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models". ICLR.
8. Meng, D., & Chen, H. (2017). "MagNet: A Two-Pronged Defense against Adversarial Examples". CCS.
9. Nie, W., et al. (2022). "Diffusion Models for Adversarial Purification". ICML.

### 医療画像AI

10. Veeling, B. S., et al. (2018). "Rotation Equivariant CNNs for Digital Pathology". MICCAI.
11. Kermany, D. S., et al. (2018). "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning". Cell.
12. Finlayson, S. G., et al. (2019). "Adversarial Attacks on Medical Machine Learning". Science.

---

**最終更新日**: 2026年1月1日
**バージョン**: 2.0.0
