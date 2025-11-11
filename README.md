# 拡散モデルを用いた敵対的攻撃に対する防御手法の研究

深層学習モデルに対する敵対的攻撃（Adversarial Attacks）は、セキュリティ上の重大な脅威となっています。本研究では、**拡散モデル（Diffusion Models）を用いた新しい防御手法**を提案し、医療画像分類タスクにおいてその有効性を検証します。

## 📋 目次

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

## 🎯 概要

本プロジェクトでは、**拡散モデル（DDPM: Denoising Diffusion Probabilistic Models）** を敵対的攻撃に対する防御機構として活用します。拡散モデルは画像のノイズ除去能力に優れており、敵対的摂動を効果的に除去できる可能性があります。

### 研究の主要な貢献

1. **拡散モデルによる敵対的防御**: DDPMを用いて攻撃画像を浄化し、元の画像に復元
2. **医療画像への適用**: 3つの医療画像データセットで防御性能を評価
3. **包括的な比較**: 既存の防御手法（JPEG圧縮、GAN、VAE）との性能比較
4. **複数の攻撃手法への対応**: FGSM、PGD、AutoAttackに対する頑健性を検証

---

## 🔬 研究の動機

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
├── README                          # このファイル
├── chestxray/                      # 胸部X線画像（肺炎分類）
│   ├── ddpm_train.py              # 拡散モデル訓練
│   ├── resnet50.py                # ResNet50分類器訓練
│   ├── load_model.py              # モデル読み込みユーティリティ
│   ├── README_resnet50.md         # ChestXray詳細ドキュメント
│   └── ddpm_out/                  # 拡散モデル出力
│       ├── ddpm_epoch*.pth
│       └── samples_epoch*.png
├── dermmel/                        # 皮膚病変画像（メラノーマ分類）
│   ├── ddpm_train.py              # 拡散モデル訓練
│   ├── resnet50.py                # ResNet50分類器訓練
│   ├── load_model.py              # モデル読み込みユーティリティ
│   ├── README_resnet50.md         # DermMel詳細ドキュメント
│   └── ddpm_out/                  # 拡散モデル出力
├── pcam/                           # 病理組織画像（転移性がん検出）
│   └── (同様の構成)
├── experiments/                    # 実験スクリプト（今後追加）
│   ├── attack_fgsm.py             # FGSM攻撃実装
│   ├── attack_pgd.py              # PGD攻撃実装
│   ├── attack_autoattack.py       # AutoAttack実装
│   ├── defense_ddpm.py            # 拡散モデル防御
│   ├── defense_jpeg.py            # JPEG圧縮防御
│   ├── defense_gan.py             # GAN防御
│   └── defense_vae.py             # VAE防御
└── results/                        # 実験結果（今後追加）
    ├── accuracy_comparison.csv
    └── visualization/
```

---

## 🗂️ データセット

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

**出典**: Veeling et al., "Rotation Equivariant CNNs for Digital Pathology", MICCAI 2018

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

**出典**: ISIC (International Skin Imaging Collaboration) アーカイブ

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

**出典**: Kermany et al., "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, 2018

---

## ⚔️ 敵対的攻撃手法

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

## 🛡️ 防御手法

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

## 💻 実験環境

### ハードウェア

- **GPU**: NVIDIA RTX 3090 / A100 / V100 等
- **CPU**: Intel Xeon / AMD EPYC
- **メモリ**: 32GB以上推奨
- **ストレージ**: SSD 100GB以上

### ソフトウェア

```bash
# Python環境
Python 3.8+

# 主要ライブラリ
torch >= 1.10.0
torchvision >= 0.11.0
numpy >= 1.21.0
Pillow >= 8.3.0
tqdm >= 4.62.0
h5py >= 3.6.0        # PCamデータセット用
matplotlib >= 3.4.0
scikit-learn >= 0.24.0

# 攻撃ライブラリ
foolbox >= 3.3.0     # 各種攻撃手法
autoattack >= 0.1    # AutoAttack実装

# オプション
tensorboard >= 2.7.0  # 訓練モニタリング
wandb >= 0.12.0       # 実験管理
```

### インストール

```bash
# 仮想環境の作成
conda create -n adv-defense python=3.8
conda activate adv-defense

# PyTorchのインストール（CUDA 11.3の例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu113

# その他の依存関係
pip install numpy pillow tqdm h5py matplotlib scikit-learn
pip install foolbox autoattack

# オプション
pip install tensorboard wandb
```

---

## 🚀 使用方法

### 1. 分類器の訓練

各データセット用のResNet50分類器を訓練します。

```bash
# ChestXray
cd chestxray
python resnet50.py

# DermMel
cd ../dermmel
python resnet50.py

# PCam
cd ../pcam
python resnet50.py
```

訓練後、`resnet50_models/resnet50_inference.pth` が生成されます。

### 2. 拡散モデルの訓練

各データセット用のDDPMを訓練します。

```bash
# ChestXray
cd chestxray
python ddpm_train.py

# DermMel
cd ../dermmel
python ddpm_train.py

# PCam
cd ../pcam
python ddpm_train.py
```

訓練後、`ddpm_out/ddpm_epoch*.pth` とサンプル画像が生成されます。

### 3. 敵対的攻撃の実行

```bash
cd experiments

# FGSM攻撃
python attack_fgsm.py --dataset chestxray --epsilon 0.03

# PGD攻撃
python attack_pgd.py --dataset dermmel --epsilon 0.031 --steps 20

# AutoAttack
python attack_autoattack.py --dataset pcam --norm Linf
```

### 4. 防御手法の評価

```bash
# 拡散モデル防御
python defense_ddpm.py --dataset chestxray --attack fgsm --t_defense 100

# JPEG圧縮防御
python defense_jpeg.py --dataset dermmel --quality 75

# GAN防御
python defense_gan.py --dataset pcam --model gan_best.pth

# VAE防御
python defense_vae.py --dataset chestxray --model vae_best.pth
```

### 5. 結果の可視化

```bash
# 精度比較グラフの生成
python visualize_results.py --results results/accuracy_comparison.csv

# 攻撃・防御の可視化
python visualize_attacks.py --dataset chestxray --attack pgd
```

---

## 📊 実験結果

### 評価指標

- **Clean Accuracy**: クリーン画像に対する精度
- **Adversarial Accuracy**: 攻撃画像に対する精度
- **Defense Accuracy**: 防御後の精度
- **Defense Rate**: (Defense Acc - Adv Acc) / (Clean Acc - Adv Acc)
- **Image Quality**: PSNR, SSIM

### 予想される結果（例）

| データセット | 攻撃 | Clean Acc | Adv Acc | DDPM | JPEG | GAN | VAE |
|------------|------|-----------|---------|------|------|-----|-----|
| ChestXray  | FGSM | 95.2%     | 12.3%   | **87.4%** | 72.1% | 81.3% | 78.9% |
| ChestXray  | PGD  | 95.2%     | 3.2%    | **79.8%** | 58.4% | 71.2% | 68.5% |
| DermMel    | FGSM | 92.5%     | 15.7%   | **85.3%** | 69.8% | 79.4% | 76.2% |
| DermMel    | PGD  | 92.5%     | 4.8%    | **77.6%** | 54.2% | 68.9% | 65.1% |
| PCam       | FGSM | 89.7%     | 18.2%   | **82.1%** | 67.3% | 76.8% | 73.5% |
| PCam       | PGD  | 89.7%     | 6.5%    | **74.2%** | 51.9% | 65.7% | 62.4% |

**注**: 上記は仮の数値です。実際の実験結果は `results/` フォルダに保存されます。

### 分析のポイント

1. **拡散モデルの優位性**: 他の手法と比較して高い防御性能
2. **攻撃手法による差**: PGDはFGSMより強力な攻撃
3. **データセット依存性**: 画像の特性により防御効果が異なる
4. **計算コストとのトレードオフ**: 拡散モデルは高性能だが処理時間が長い

---

## 📈 今後の展開

### 短期目標

- [ ] 全データセットでの分類器訓練完了
- [ ] 全データセットでの拡散モデル訓練完了
- [ ] 攻撃スクリプトの実装（FGSM, PGD, AutoAttack）
- [ ] 防御スクリプトの実装（DDPM, JPEG, GAN, VAE）
- [ ] 実験結果の収集と分析

### 中期目標

- [ ] 拡散モデルの最適化（高速化、メモリ効率化）
- [ ] 適応的攻撃への対応
- [ ] 他の医療画像データセットへの拡張
- [ ] 論文執筆と投稿

### 長期目標

- [ ] リアルタイム防御システムの構築
- [ ] 臨床応用への検討
- [ ] オープンソース化とコミュニティへの貢献

---

## 📚 参考文献

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

## 👥 貢献者

- **研究者**: [あなたの名前]
- **所属**: [所属機関]
- **連絡先**: [メールアドレス]

---

## 📄 ライセンス

本プロジェクトは研究目的で公開されています。商用利用の際は別途ご相談ください。

---

## 🙏 謝辞

- データセット提供者
- オープンソースコミュニティ
- 研究室メンバー

---

**最終更新日**: 2025年11月8日
**バージョン**: 1.0.0