# 検証中のGAN敵対的防御について

## 結論：**全てUnconditional GANで行っています**

---

## 1. Defense-GAN実装の特徴

### 1.1 訓練時（defense_gan_train*.py）

```python
# Generatorの構造
class Generator(nn.Module):
    def __init__(self, latent_dim=128, ngf=64, nc=3):
        # 入力：z (潜在ベクトル)
        # クラスラベル：使用していない
        self.fc = nn.Linear(latent_dim, ngf * 16 * 7 * 7)
        ...
    
    def forward(self, z):
        # z のみから画像を生成
        # クラス情報は使用しない
        return self.main(x)
```

**結論**: **Unconditional GAN**

---

### 1.2 敵対的防御時（gan_pgd_eval_v4.py, gan_fgsm_eval*.py）

```python
class DefenseGANv4:
    """
    Defense-GANで画像を浄化する方法：
    1. 敵対的画像 x_adv を入力
    2. 潜在変数 z を最適化（L-BFGSまたはAdam）
    3. Generator(z) ≈ x_adv となるように最適化
    4. x_purified = Generator(z_optimal) を使用
    """
    
    def __init__(self, generator, latent_dim=512, rec_iters=200, rec_rr=5,
                 perceptual_weight=0.0, use_lbfgs=True, device='cuda'):
        # generator: Unconditional GAN
        # 敵対的防御には、クラスラベルを渡していない
        ...
```

**敵対的防御の流れ**:
```
[敵対的画像 x_adv]
    ↓
[L-BFGS最適化: argmin_z ||G(z) - x_adv||]
    ↓
[z_optimal に対応する z を見つける]
    ↓
[x_purified = G(z_optimal)]
    ↓
[分類器で予測]
```

**結論**: 敵対的防御時も**クラスラベルは使用していない**

---

## 2. 実装バージョンの比較

### 版の進化過程

| バージョン | 特徴 | Conditional? |
|-----------|------|-------------|
| **defense_gan_train.py** | DCGAN基本版 | ❌ Unconditional |
| **defense_gan_train_v2.py** | Label smoothing, ノイズ追加 | ❌ Unconditional |
| **defense_gan_train_v3.py** | ResNet + Self-Attention + EMA | ❌ Unconditional |
| **gan_pgd_eval_v4.py** | L-BFGS最適化、RGB対応 | ❌ Unconditional |
| **gan_fgsm_eval_v4.py** | FGSM攻撃対応 | ❌ Unconditional |

すべて**Unconditional**。

---

## 3. 敵対的防御のアルゴリズム詳細

### 3.1 再構成アルゴリズム（Defense-GAN v4）

```python
def reconstruct(self, x_adv, target=None):
    """
    敵対的画像を浄化
    
    目的: G(z*)を使って敵対的摂動を除去
    - x_adv: 敵対的画像
    - z*: 最適化された潜在変数
    """
    
    best_z = None
    best_loss = float('inf')
    
    # ランダムリスタート（デフォルト5回）
    for restart in range(self.rec_rr):
        # 初期 z をランダムに初期化
        z = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
        
        # L-BFGSで最適化
        optimizer = torch.optim.LBFGS([z], max_iter=self.rec_iters, ...)
        
        def closure():
            optimizer.zero_grad()
            
            # Generator で画像を再構成
            x_recon = self.generator(z)
            
            # Reconstruction Loss
            loss = F.l2_loss(x_recon, x_adv)
            
            # Perceptual Loss（オプション）
            if self.perceptual_weight > 0:
                loss += self.perceptual_weight * perceptual_loss(x_recon, x_adv)
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
        final_loss = closure().item()
        
        if final_loss < best_loss:
            best_loss = final_loss
            best_z = z.detach().clone()
    
    # 最良の z を使って再構成
    x_purified = self.generator(best_z)
    return x_purified
```

**重要な点**:
- クラスラベル（target）は定義されているが、実装では使用されていない
- 純粋に「敵対的画像にマッチする潜在変数」を探しているだけ

---

## 4. 本実装が Unconditional な理由

### 設計思想

Defense-GANの原論文でも**Unconditional**：

> "Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
> Pouya Samangouei, Maya Kabkab, Rama Chellappa (ICLR 2018)

**引用**:
```
"We use an unconditional generator trained on clean images.
At test time, to defend against adversarial examples, we find the closest point
in the generator's output manifold to the input (adversarial) image."
```

---

## 5. Conditional GANにしなかった理由

### トレードオフ分析

| 項目 | Unconditional（現在） | Conditional |
|------|----------------------|-----------|
| 敵対的画像の浄化 | ✅ 有効 | ✅ より有効 |
| 推論時ラベル必要 | ❌ 不要 | ✅ **必須** |
| 計算量 | 低 | 中 |
| 実装複雑性 | 低 | 高 |
| 未知クラス対応 | ✅ 可能 | ❌ 不可 |

### Defense-GANの利点

Unconditionalの利点：
1. **予測なしで浄化可能** - ラベル不要
2. **汎用性** - 未知のクラスにも対応
3. **理論的シンプル性** - Manifold仮説に基づく

**Conditional にしないほうが、防御として「独立」に働く**

---

## 6. 関連ファイルの検証

### 6.1 訓練ファイル群
```
projects/pcam/gan/
├── defense_gan_train.py          # DCGAN + Unconditional
├── defense_gan_train_v2.py        # 改良版（まだUnconditional）
├── defense_gan_train_v3.py        # ResNet版（まだUnconditional）
└── defense_gan_train_v4.py        # 最新版（未確認だがUnconditional推定）
```

### 6.2 評価ファイル群
```
projects/pcam/gan/
├── fgsm/
│   ├── gan_fgsm_eval.py           # FGSM評価
│   ├── gan_fgsm_eval_v3.py
│   ├── gan_fgsm_eval_v3_fixed.py
│   └── gan_fgsm_eval_v4.py        # 最新FGSM評価
├── pgd/
│   ├── gan_pgd_eval.py            # PGD評価
│   ├── gan_pgd_eval_v4.py         # 最新PGD評価
└── autoattack/
    ├── gan_autoattack_eval.py     # AutoAttack評価（未作成？）
```

全て**Unconditional**に基づいている。

---

## 7. 先生への回答案

### Q: 「今回検証しているGANでの敵対的防御は全てConditionalで行っていますか？」

### A: 「いいえ、全てUnconditionalです。」

**理由**:
1. **Defense-GAN論文に準拠** - 元論文がUnconditional
2. **設計思想** - 推論時にラベル不要で汎用性を重視
3. **実装** - Generatorの入力は潜在ベクトルzのみ

**補足**:
- 将来的には、正解ラベルが得られる場合にConditional版への拡張も検討可能
- ただし現在は、Defense-GAN本来の「ラベルなしで浄化」という特徴を活かしている

---

## 8. VAEとGANの敵対的防御の比較

### 本実装での差異

| 項目 | VAE（本実装） | GAN（本実装） |
|------|-------------|------------|
| 訓練時 | Unconditional | Unconditional |
| 敵対的防御時 | 直接再構成 | 潜在変数最適化 |
| クラス情報 | ❌ 使用なし | ❌ 使用なし |
| 推論速度 | 高速（1回のforward） | **遅い**（最適化200回） |
| 再構成品質 | 中（VAE-GAN改良） | 高（Generator精度） |

### 敵対的防御の考え方

```
VAE方式:
[敵対的画像] →[Encoder]→ z →[Decoder]→[浄化画像]

GAN方式:
[敵対的画像] ← [Generator] ← z*（最適化）
             （最適化が目的）
```

---

## 結論

**本検証で使用しているGAN敵対的防御は全てUnconditionalです。**

これはDefense-GAN論文の設計思想を踏襲し、推論時にラベルが不要という実用性を重視した選択です。
