# VAE-GANハイブリッド設計 - なぜVAEにGANが必要か？

## 問題：VAEの「ぼやけ」問題

### VAE単独の場合

```
[クリーン画像] → [Encoder] → [潜在空間] → [Decoder] → [再構成画像]
                                          ↓
                                    ぼやけている！
```

**VAEの特徴**:
- 損失関数が L1/L2（ピクセル単位）
- **MSEを最小化しようとする** = 「平均的な」画像を生成
- 複数の可能性を「足し合わせた」結果がぼやける

**数式的理由**:
```
L_recon = ||x - x̂||²  （ピクセルごとのL2誤差）

複数の可能な画像x1, x2, ...に対して、
MSEを最小化すると: x̂ = (x1 + x2 + ...) / N
                    ↑
                「平均化された」ぼやけた画像
```

**視覚的例**:
```
元の画像: 顔（輪郭がはっきり）
  ↓ 複数の「起こりうる再構成」
  - 同じ顔が少し回転した画像
  - 少し異なる顔の角度
  ↓ MSEを最小化しようとする
再構成: 「平均化された顔」（ぼやけている）
```

---

## 解決策：Discriminatorを追加 → **VAE-GAN**

### VAE-GANのアーキテクチャ

```
[クリーン画像] 
  ↓
[Encoder] → [潜在空間] → [Decoder] → [再構成画像] ─→ ┐
                                                    ├→ [Discriminator]
[クリーン画像] ────────────────────────────────────→ ┘
  ↓
  「この再構成画像は本物に見える？」
```

### 訓練時の役割分担

**Discriminator の役割**:
```python
# コード: train_vae.py より

# 1. Discriminatorを訓練
d_real = disc(x)              # クリーン画像 → "本物"と判定
d_fake = disc(recon_x)        # 再構成画像 → "ニセモノ"と判定

# Hinge Loss (WGAN-GP風)
disc_loss = (F.relu(1.0 - d_real).mean() +    # 本物を正しく認識
             F.relu(1.0 + d_fake).mean()) # ニセモノを正しく認識

# 2. VAEを訓練
d_fake_for_g, features_fake = disc(recon_x, return_features=True)
adv_loss = -d_fake_for_g.mean()               # Discriminatorを騙す

# 敵対的損失を加える
total_loss = (recon_loss + 
             beta * kl_loss +
             ... +
             args.lambda_adv * adv_loss)  # ← Discriminatorの判定を使う
```

### VAE-GANの学習過程

```
イテレーション1:
[VAE] → [ぼやけた再構成] → [Discriminator] → "ニセモノ"と判定
                           ↓
                      VAEは「よりリアルな画像を生成」するように改良

イテレーション2:
[VAE] → [より鮮明な再構成] → [Discriminator] → より難しい判定
                           ↓
                      さらに改良...

最終的に:
[VAE] → [リアルで鮮明な再構成] → [Discriminator] → 「これは本物か？」
                                            と迷うレベル
```

---

## 敵対的防御との関係

### 敵対的防御で必要な画像品質

```
敵対的防御の目的: 「敵対的摂動を除去する」

敵対的画像: ノイズが混ぜられた画像
  ↓
浄化（再構成）が目的

浄化の品質が低い（ぼやけ）
  → 分類器の入力品質が低下
  → 防御効果が減少
```

### 防御効果の比較

```
例：DermMel（メラノーマ分類）

【VAE単独（ぼやけた再構成）】
敵対的画像 → VAEで再構成 → ぼやけた画像 → 分類器
                ↓
           ノイズは除去されたが、細部も失われた
           → 分類精度が回復しない

【VAE-GAN（鮮明な再構成）】
敵対的画像 → VAEで再構成 → 鮮明な画像 → 分類器
         （Discriminatorが監視）
                ↓
           ノイズは除去されて、細部も保持
           → 分類精度が回復する
```

---

## 実装コードの詳細説明

### Loss関数の構成

```python
# train_vae.py より

# 1. 再構成損失（画像をできるだけ忠実に再構成）
recon_loss = F.l1_loss(recon_x, x) + F.mse_loss(recon_x, x)

# 2. KL損失（潜在空間の正則化、β<<1で再構成重視）
kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# 3. 知覚的損失（VGG特徴量）→ 高レベルの特徴が近い
perceptual = perceptual_loss_fn(recon_x, x)

# 4. SSIM損失（構造的類似性）→ エッジやテクスチャを保持
ssim = ssim_loss_fn(recon_x, x)

# 5. エッジ損失（Sobel）→ 細部を保持
edge = edge_loss_fn(recon_x, x)

# 6. 敵対的損失（Discriminator）→ リアルに見える画像を生成
adv_loss = -d_fake_for_g.mean()

# 7. Feature Matching（Discriminator特徴量）→ 安定化
fm_loss = sum(F.l1_loss(f, r.detach()) 
              for f, r in zip(features_fake, features_real))

# 総合損失
total_loss = (recon_loss + 
             beta * kl_loss +
             λ_perceptual * perceptual +
             λ_ssim * ssim +
             λ_edge * edge +
             λ_adv * adv_loss +      ← Discriminator
             λ_fm * fm_loss)         ← Discriminator
```

### なぜこれらの損失が必要か？

| 損失 | 役割 | VAE単独 | VAE-GAN |
|------|------|--------|---------|
| **Reconstruction** | ピクセル再構成 | ✓ | ✓ |
| **KL** | 潜在空間正則化 | ✓ | ✓ |
| **Perceptual** | 高レベル特徴 | ✗ ぼやける | ✓ 改善 |
| **SSIM** | 構造保持 | ✗ ぼやける | ✓ 改善 |
| **Edge** | 細部保持 | ✗ ぼやける | ✓ 改善 |
| **Adversarial** | リアルさ | ✗ ぼやける | ✓ **重要** |
| **Feature Matching** | 安定化 | - | ✓ 安定化 |

---

## VAE vs GAN vs VAE-GAN の比較

### 原理の違い

```
【純粋なVAE】
最小化: L1/MSE誤差
結果: ぼやけた再構成

【純粋なGAN】
最小化: Discriminatorとの敵対
結果: リアルだがたまに不安定

【VAE-GAN（ハイブリッド）】
最小化: L1/MSE + 敵対 + 知覚的類似
結果: リアルで安定した再構成 ✓
```

### 敵対的防御での選択

```
【Defense-GAN（純粋GAN）】
- 利点: 超リアルな再構成
- 欠点: 訓練が不安定、テスト時最適化が遅い

【敵対的防御用VAE-GAN（本実装）】
- 利点: VAEの安定性 + GANのリアルさ
- 欠点: 計算量増加

DermMel選択: VAE-GAN ← 医療画像は安定性重視
```

---

## 敵対的防御での動作の流れ

### テスト時（推論時）

```python
# vae_autoattack_eval.py より

def reconstruct(x_adv):
    """再構成（推論時）"""
    # VAEを使用（Discriminatorは不要）
    mu, _ = encoder(x_adv)
    x_purified = decoder(mu)
    return x_purified
```

**重要**: 訓練時はDiscriminatorで監視して品質を高めるが、
テスト時（敵対的防御時）はVAEだけで浄化する

### 訓練時と推論時の違い

```
【訓練時】
[クリーン画像] → [VAE] → [再構成] → [Discriminator監視]
                                      ↓
                                  品質が高い再構成を学習

【推論時（敵対的防御）】
[敵対的画像] → [VAE] → [浄化画像] → [分類器]
           （Discriminatorは使わない）
```

---

## まとめ

### なぜVAEにGANを組み合わせるのか？

**1つの言葉で**: **VAEのぼやけをGANで解決するため**

### VAE単独の問題
```
L2誤差を最小化 → 複数の可能性を平均化 → ぼやけた画像
```

### GAN追加による改善
```
Discriminatorが「本物らしさ」を判定
→ VAEが「ノイズ除去 + 鮮明さ」の両方を学習
→ 高品質な再構成が可能
```

### 敵対的防御での効果
```
敵対的防御 = 画像の浄化 = 高品質な再構成が必須
→ VAE-GANが最適な選択
```

---

## 関連論文

1. **VAE-GAN**: "Autoencoding beyond pixels using a learned similarity metric"
   - Larsen et al., ICML 2016
   - VAEとGANの結合の初期提案

2. **Perceptual Loss**: "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
   - Johnson et al., ECCV 2016
   - 高レベル特徴を使った損失

3. **β-VAE**: "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
   - Higgins et al., ICLR 2017
   - KL項の調整（β<<1で再構成重視）

4. **Defense-GAN**: "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
   - Samangouei et al., ICLR 2018
   - 敵対的防御へのGANの応用

5. **Feature Matching**: "Improved Techniques for Training GANs"
   - Salimans et al., NeurIPS 2016
   - Discriminator特徴量を使った安定化
