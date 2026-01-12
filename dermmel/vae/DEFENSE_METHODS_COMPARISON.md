# 敵対的防御における Conditional vs Unconditional の比較

## 概要

敵対的防御手法には大きく分けて以下の種類がある：

1. **VAE (本実装) - Unconditional**
2. **DiffPure (Diffusion Model) - Conditional OR Unconditional**
3. **GAN-based Defense - Conditional**

---

## 1. VAE敵対的防御（本実装）- **Unconditional**

### 特徴
- **Unconditional VAE-GAN**
- クラスラベルを使用しない
- 入力画像のみから再構成

### 原理
```
[敵対的画像] → [VAE再構成] → [浄化された画像]
             (クラス非依存)
```

### 利点
- シンプルで汎用的
- 推論時にラベル不要
- どのクラスの画像でも処理可能

### 欠点
- クラス情報を活用できないため、最適な再構成ができない可能性

---

## 2. DiffPure (Diffusion Model) - **Conditional OR Unconditional**

DiffPureは導入されている**Diffusion Modelの種類**によって異なる：

### 2.1 Unconditional Diffusion版

**実装**: DDPM（Denoising Diffusion Probabilistic Models）

```python
# 学習時
[Clean Image] → Noiseを追加 → Reverse Process → [Clean Image]

# 敵対的防御時
[Adversarial Image] → Reverse Process → [Purified Image]
                    (クラス非依存)
```

**特徴**:
- クラスラベル不使用
- 純粋なノイズ除去
- 任意の画像に適用可能

### 2.2 Conditional Diffusion版

**実装**: Guided Diffusion（OpenAIのClassifier-Free Guidance対応）

```python
# 敵対的防御時
[Adversarial Image] + [Class Label] → Conditional Reverse Process → [Purified Image]

# クラス情報を活用:
- クラス固有のパターンで浄化
- より正確な再構成
```

**特徴**:
- クラスラベルを条件として使用
- Classifier-Free Guidance対応
- より高精度な浄化

### 2.3 DiffPureの実装状況

本ワークスペースのDiffPureを確認すると：

```python
# 参考: guided_diffusion/script_util.py
NUM_CLASSES = 1000  # ImageNetの場合

# UNetModelは条件付きサポート
class UNetModel(nn.Module):
    def __init__(self, ..., num_classes=None, ...):
        # num_classes != None → Conditional
        # num_classes = None → Unconditional
```

**実装判定**:
- `eval_sde_adv.py` では通常、条件情報を指定していない場合が多い
- DermMel（2クラス）のような小規模タスクでは、Unconditionalが一般的

---

## 3. GAN-based Defense - **通常Conditional**

### 代表例

#### a) **Adversarial Auto-Encoder (AAE)**
- Encoder-Decoder + Discriminator
- **Conditional** (class label条件付け)

#### b) **Conditional GAN + Defense**
```python
# 敵対的防御時
[Adversarial Image] + [Class Label] → Conditional Generator → [Purified Image]
                                         (Discriminatorで判別)
```

#### c) **本実装のVAE-GAN**（比較対象）
```python
# 本実装（Unconditional）
[Adversarial Image] → VAE Encoder → Latent Space → VAE Decoder → [Purified Image]
                      (クラス非依存)
```

### GAN-based Defenseの特徴

| 項目 | 説明 |
|------|------|
| **通常の形式** | Conditional GAN (cGAN) |
| **条件情報** | クラスラベルまたはワンホットベクトル |
| **学習** | Generator と Discriminator の敵対的学習 |
| **推論時** | ラベル必須 |
| **利点** | より鮮明な画像、クラス固有の浄化 |
| **欠点** | ラベル依存、未知クラスに対応困難 |

---

## 4. 比較表

| 特性 | VAE本実装 | DiffPure (条件なし) | DiffPure (条件付き) | GAN Defense |
|------|---------|------------------|-------------------|-----------|
| **タイプ** | Unconditional VAE-GAN | Unconditional Diffusion | Conditional Diffusion | Conditional GAN |
| **クラス情報** | ❌ 不使用 | ❌ 不使用 | ✅ 使用 | ✅ 使用 |
| **推論時ラベル** | 不要 | 不要 | 必須 | 必須 |
| **汎用性** | 高（任意クラス対応） | 高（任意クラス対応） | 中（訓練済みクラスのみ） | 中（訓練済みクラスのみ） |
| **精度** | 中 | 中～高 | 高 | 高 |
| **計算量** | 低 | **高** | **高** | 低 |
| **推論速度** | 高速 | **遅い**（逆拡散プロセス） | **遅い** | 高速 |

---

## 5. DermMelの文脈での選択

### なぜ本実装がUnconditionalか

1. **推論時の柔軟性**: ラベルが不明でも動作
2. **シンプル性**: 実装と訓練の簡潔さ
3. **実用性**: リアルタイム応用を想定

### 代替案との比較

| 方式 | メリット | デメリット |
|------|---------|----------|
| **VAE (本実装)** | 高速、シンプル | 精度はDiffusionより低い可能性 |
| **DiffPure** | 高精度 | **推論遅い**（拡散ステップ1000回） |
| **Conditional GAN** | 高速＋高精度 | ラベル依存、複雑 |

---

## 6. 推奨される組み合わせ

### DermMelのユースケース別

#### A) リアルタイム敵対的防御（推奨：本実装）
```
[敵対的画像] → [VAE浄化] → [分類器] → 予測
              ⏱️ 数ms          結果
```
✅ VAE (Unconditional) が最適

#### B) 最高精度重視
```
[敵対的画像] + [正解ラベル] → [DiffPure浄化] → [分類器]
                          ⏱️ 数秒（遅い）
```
✅ Conditional Diffusion が最適

#### C) 精度とスピードのバランス
```
[敵対的画像] + [Predicted Label] → [Conditional VAE] → [分類器]
                                ⏱️ 数ms～数十ms
```
✅ Conditional VAE-GAN（本実装の拡張案）

---

## 7. 将来の拡張案

### 案1: Conditional VAE版への拡張

```python
class ConditionalVAE(nn.Module):
    """クラス条件付きVAE"""
    def __init__(self, ..., num_classes=2):
        # Encoderに埋め込まれたクラス情報を入力
        # Decoderにもクラス情報を条件として渡す
        
    def forward(self, x, class_label):
        # x: 画像
        # class_label: クラスラベル（0 or 1）
        mu, logvar = self.encoder(x, class_label)
        ...
```

**メリット**:
- より正確な浄化
- 推論時にラベルがあれば活用可能
- 後方互換性（ラベルなしでも動作）

### 案2: DiffPureとの組み合わせ

```
[敵対的画像] 
  ├→ [Lightweight VAE] → 粗い浄化 (高速)
  └→ [DiffPure] → 細部調整 (精密)
```

段階的浄化で「速度」と「精度」のバランス

---

## 結論

### 本実装について
- **Unconditional VAE-GAN**は敵対的防御として最適な選択
- MagNet方式を踏襲した標準的なアプローチ
- シンプルさと実用性を重視

### 先生への説明ポイント
1. クラス条件を使わない理由：**推論時の柔軟性と実用性**
2. VAE-GANにした理由：**VAEの"ぼやけ"をGANで改善**
3. β<1にした理由：**再構成品質を優先（浄化目的）**
4. DiffPureより選ばれなかった理由：**推論速度（DiffPure は数秒かかる）**

---

## 参考資料

### 関連論文
1. **MagNet**: "MagNet: a Two-Pronged Defense against Adversarial Examples" (Meng & Chen, CCS 2017)
   - Autoencoder を使った敵対的防御の先駆け
   - Unconditional に敵対的画像を浄化

2. **DiffPure**: "Diffpure: Generative Diffusion Models as Generic Preprocessors for Improving Adversarial Robustness" (Nie et al., ICLR 2022)
   - 拡散モデルを使った防御
   - Conditional / Unconditional 両対応

3. **Guided Diffusion**: "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol, NeurIPS 2021)
   - クラス条件付き拡散モデル
   - Classifier-Free Guidance

4. **Conditional GAN**: "Conditional Generative Adversarial Nets" (Mirza & Osindero, CVPR 2014)
   - クラス条件付きGAN
   - 敵対的防御への応用例多数
