# VAEモデルの概要

## 1. 使用目的

**MagNet方式の敵対的防御（Adversarial Defense）**

敵対的攻撃（Adversarial Attack）によって摂動を加えられた画像を、VAEで再構成（浄化）することで、摂動を除去し、分類器の精度を回復させる。

```
[敵対的画像] → [VAE再構成] → [浄化された画像] → [分類器] → [正しい予測]
```

### 参考論文
- **MagNet**: "MagNet: a Two-Pronged Defense against Adversarial Examples" (Meng & Chen, ACM CCS 2017)

---

## 2. VAEのタイプ

### 基本構造: **標準VAE（Unconditional VAE）**

- **Conditional VAEではない**: クラスラベルを条件として与えていない
- 入力画像のみから潜在変数を推定し、再構成する
- 目的が「画像の浄化」であるため、クラス情報は不要

### 拡張手法: **VAE-GAN ハイブリッド**

標準VAEに以下の拡張を加えている：

| 手法 | 説明 | 目的 |
|------|------|------|
| **Adversarial Loss** | Discriminatorによる敵対的損失 | より鮮明な再構成画像を生成 |
| **Perceptual Loss** | VGG16の特徴量でのL1損失 | 知覚的に類似した画像を生成 |
| **SSIM Loss** | 構造的類似性損失 | 構造情報を保持 |
| **Edge Loss** | Sobelフィルタによるエッジ損失 | エッジを保存 |
| **Feature Matching** | Discriminator特徴量のマッチング | 学習の安定化 |

---

## 3. アーキテクチャ詳細

### 3.1 全体構成

```
入力画像 (224×224×3)
    ↓
[Encoder] - ResBlockで5回ダウンサンプリング
    ↓
潜在空間 (512次元) - μとσを出力、Reparameterization Trick
    ↓
[Decoder] - ResBlockで5回アップサンプリング
    ↓
再構成画像 (224×224×3)
```

### 3.2 Encoder

```python
入力: 224×224×3 (RGB画像)
  ↓ Conv2d (3→64ch)
  ↓ ResBlock + AvgPool: 224→112 (64ch)
  ↓ ResBlock + AvgPool: 112→56  (128ch)
  ↓ ResBlock + AvgPool: 56→28   (256ch)
  ↓ ResBlock + AvgPool: 28→14   (512ch)
  ↓ ResBlock + AvgPool: 14→7    (512ch)
  ↓ Flatten + FC
出力: μ (512次元), log(σ²) (512次元)
```

### 3.3 Decoder

```python
入力: z (512次元の潜在変数)
  ↓ FC + Reshape: 7×7×512
  ↓ ResBlock + Upsample: 7→14   (512ch)
  ↓ ResBlock + Upsample: 14→28  (256ch)
  ↓ ResBlock + Upsample: 28→56  (128ch)
  ↓ ResBlock + Upsample: 56→112 (64ch)
  ↓ ResBlock + Upsample: 112→224 (64ch)
  ↓ Conv2d + Sigmoid
出力: 224×224×3 (再構成画像)
```

### 3.4 Discriminator (VAE-GAN用)

- Spectral Normalizationで安定化
- 5層の畳み込み層
- Hinge Lossを使用

---

## 4. 損失関数

### 総合損失

```
L_total = L_recon + β·L_KL + λ_perc·L_perceptual + λ_ssim·L_SSIM 
        + λ_edge·L_edge + λ_adv·L_adversarial + λ_fm·L_feature_matching
```

### 各損失の詳細

| 損失 | 式 | 重み | 役割 |
|------|-----|------|------|
| **再構成損失** | L1 + MSE | 1.0 | ピクセル単位の再構成 |
| **KL損失** | KL(q(z\|x) \|\| p(z)) | β=0.01 | 潜在空間の正則化 |
| **Perceptual損失** | VGG特徴量のL1 | 2.0 | 知覚的類似性 |
| **SSIM損失** | 1 - SSIM(x, x̂) | 2.0 | 構造的類似性 |
| **Edge損失** | Sobel特徴のL1 | 1.0 | エッジ保存 |
| **Adversarial損失** | -D(x̂) | 0.02 | 鮮明さ向上 |
| **Feature Matching** | D特徴量のL1 | 0.1 | 学習安定化 |

### β-VAEについて

- 標準VAEではβ=1だが、本実装ではβ=0.01（最大値）
- **β < 1の理由**: 再構成品質を優先するため
- KL項を小さくすることで、より忠実な再構成が可能

### Cyclical Annealing

- KL vanishing問題を緩和するため、βを周期的に変化
- 100エポック周期で0からβ_maxまで増加

---

## 5. 学習の工夫

| 技術 | 説明 |
|------|------|
| **EMA (Exponential Moving Average)** | パラメータの指数移動平均で安定した推論 |
| **Gradient Accumulation** | 実効バッチサイズを増加 |
| **Warmup + Cosine Annealing** | 学習率スケジューリング |
| **Spectral Normalization** | Discriminatorの安定化 |
| **Gradient Clipping** | 勾配爆発の防止 |

---

## 6. Conditional VAEとの比較

| 項目 | 本実装 (Unconditional VAE) | Conditional VAE |
|------|---------------------------|-----------------|
| クラス情報 | **使用しない** | Encoder/Decoderに入力 |
| 目的 | 画像の浄化（クラス非依存） | クラス条件付き生成 |
| 潜在空間 | 全クラス共通 | クラスごとに分離可能 |
| 適用場面 | 敵対的防御、ノイズ除去 | 条件付き画像生成 |

### なぜUnconditional VAEを選んだか

1. **MagNet方式の防御** ではクラス情報は不要
2. 敵対的画像の「浄化」が目的であり、特定クラスの生成ではない
3. 推論時にクラスラベルが不明でも動作する必要がある

---

## 7. GANとの違い

| 項目 | VAE (本実装) | GAN |
|------|-------------|-----|
| 潜在空間 | 確率分布（ガウス分布）として明示的にモデル化 | 暗黙的 |
| 学習目標 | 変分下限の最大化 | 敵対的学習 |
| モード崩壊 | 起きにくい | 起きやすい |
| 再構成 | 入力画像の再構成が可能 | 直接的な再構成は困難 |
| 画像品質 | ややぼやける傾向 | 鮮明だがアーティファクトの可能性 |

### 本実装での工夫

VAEの「ぼやけ」問題を解決するため、以下を導入：
- **Adversarial Loss**: GANのDiscriminatorを追加（VAE-GAN）
- **Perceptual Loss**: 高レベル特徴量での損失
- **SSIM Loss**: 構造的類似性の保持

---

## 8. ハイパーパラメータ

```python
# モデル設定
latent_dim = 512      # 潜在空間の次元
base_ch = 64          # 基本チャンネル数
img_size = 224        # 入力画像サイズ

# 損失の重み
beta_max = 0.01       # KL項の最大重み（再構成重視）
lambda_perceptual = 2.0
lambda_ssim = 2.0
lambda_edge = 1.0
lambda_adv = 0.02
lambda_fm = 0.1

# 学習設定
lr_vae = 1e-4
lr_disc = 1e-4
epochs = 300
batch_size = 8
```

---

## 9. 推論時の動作

```python
def reconstruct(self, x):
    """推論時: 平均値μのみを使用（ノイズなし）"""
    mu, _ = self.encoder(x)
    return self.decoder(mu)
```

- 学習時: Reparameterization Trick で z = μ + σ·ε（ε~N(0,1)）
- **推論時: z = μ のみ使用**（決定論的な再構成）

---

## 10. まとめ

本実装のVAEは以下の特徴を持つ：

1. **タイプ**: Unconditional VAE-GAN ハイブリッド
2. **目的**: MagNet方式の敵対的防御（画像浄化）
3. **特徴**:
   - クラス条件なし（Conditional VAEではない）
   - β-VAE（β<1）で再構成品質を優先
   - Discriminatorによる敵対的損失で鮮明化
   - Perceptual/SSIM/Edge損失で知覚的品質を向上
4. **推論**: 決定論的（μのみ使用）

---

## 参考文献

1. VAE: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014
2. β-VAE: Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017
3. VAE-GAN: Larsen et al., "Autoencoding beyond pixels using a learned similarity metric", ICML 2016
4. MagNet: Meng & Chen, "MagNet: a Two-Pronged Defense against Adversarial Examples", ACM CCS 2017
5. Perceptual Loss: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV 2016
6. SSIM: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity", IEEE TIP 2004
7. Spectral Normalization: Miyato et al., "Spectral Normalization for Generative Adversarial Networks", ICLR 2018
8. Cyclical Annealing: Fu et al., "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing", NAACL 2019
