# DDPM, GAN, VAE敵対的防御 - Conditional/Unconditional 検証結果

## 結論：**全てUnconditionalです** ✓

---

## 1. 詳細検証結果

### 1.1 VAE（DermMel実装）

**タイプ**: **Unconditional VAE-GAN** ✓

```python
# train_vae.py より
class VAE(nn.Module):
    def __init__(self, img_channels=3, base_ch=64, latent_dim=512):
        # 入力: RGB画像のみ
        # クラスラベル: 使用なし
        self.encoder = Encoder(img_channels, base_ch, latent_dim)
        self.decoder = Decoder(img_channels, base_ch, latent_dim)
```

敵対的防御時：
```python
def reconstruct(self, x):
    """推論時: 平均値μのみを使用"""
    mu, _ = self.encoder(x)  # ← ラベル使用なし
    return self.decoder(mu)
```

---

### 1.2 GAN（PCam検証）

**タイプ**: **Unconditional GAN** ✓

```python
# defense_gan_train_v3.py より
class Generator(nn.Module):
    def __init__(self, latent_dim=512, ngf=64, nc=3):
        # 入力: 潜在ベクトル z のみ
        # クラスラベル: 使用なし
        self.fc = nn.Linear(latent_dim, ngf * 8 * 7 * 7)
    
    def forward(self, z):
        # z のみから画像を生成
        return self.decoder(z)
```

敵対的防御時：
```python
# gan_pgd_eval_v4.py より
def reconstruct(self, x_adv):
    """潜在変数を最適化してx_advに最適にマッチする z を見つける"""
    # L-BFGS最適化: argmin_z ||G(z) - x_adv||
    # ← ラベル使用なし（潜在変数最適化のみ）
    return self.generator(z_optimal)
```

---

### 1.3 DDPM（DiffPure実装）

**タイプ**: **Unconditional Diffusion** ✓

```python
# guided_diffusion/script_util.py より
def model_and_diffusion_defaults():
    res = dict(
        image_size=64,
        class_cond=False,  # ← デフォルトで False！
        ...
    )
```

```python
# diffpure_guided.py より
class GuidedDiffusion(torch.nn.Module):
    def __init__(self, args, config, device=None):
        # モデル読み込み
        model, diffusion = create_model_and_diffusion(**model_config)
        # model_config['class_cond']=False → Unconditional
        model.load_state_dict(
            torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', ...)
        )
```

敵対的防御時：
```python
# eval_sde_adv.py より
def forward(self, x):
    # 拡散モデルで敵対的画像を浄化
    x_re = self.runner.image_editing_sample(
        (x - 0.5) * 2,
        bs_id=counter,
        tag=self.tag
    )
    # ← ラベル使用なし
    return self.classifier((x_re + 1) * 0.5)
```

**重要**: `256x256_diffusion_uncond.pt` という**Unconditional モデル**を使用

---

## 2. 三者比較表

| 項目 | VAE | GAN | DDPM |
|------|-----|-----|------|
| **基本構造** | Unconditional | Unconditional | Unconditional |
| **敵対的防御方式** | 直接再構成 | 潜在変数最適化 | 逆拡散プロセス |
| **クラスラベル** | ❌ 不使用 | ❌ 不使用 | ❌ 不使用 |
| **推論速度** | 高速（ms） | 遅い（秒） | **非常に遅い**（数秒～分） |
| **再構成品質** | 中 | 高 | **最高** |
| **実装複雑度** | 低 | 中 | **高** |

---

## 3. Conditional対応の有無

### VAE
- **訓練時**: Unconditional
- **拡張案**: Conditional VAEへの拡張も可能だが、実装していない

### GAN
- **訓練時**: Unconditional
- **理由**: Defense-GAN論文（ICLR 2018）の設計思想に準拠
- **実装**: ラベル埋め込み層なし

### DDPM
- **訓練時**: Unconditional
- **コード内に Conditional オプション有り**: 
  ```python
  class_cond=False  # デフォルト
  ```
  ただし、DiffPureの敵対的防御では`class_cond=False`で固定
- **理由**: 敵対的防御には「ラベル非依存」が重要

---

## 4. なぜ全てUnconditionalなのか？

### 敵対的防御の設計哲学

敵対的防御において、Unconditionalが選ばれる理由：

1. **推論時の柔軟性**
   - ラベル不要 → 分類器の予測結果に依存しない
   - より「独立した」防御になる

2. **汎用性**
   - 訓練データにないクラスにも対応可能
   - モデルの適応性向上

3. **理論的堅牢性**
   - Manifold仮説（MagNet, Defense-GAN）
   - 「クリーン画像のManifold」に画像を投影することが目的
   - ラベルは不要

4. **先制攻撃への耐性**
   - 敵対者がラベル情報を知っていても無関係
   - ラベルを条件として使わないため、Label-Only攻撃に強い

---

## 5. Conditional版への拡張の可能性

### もし Conditional にしたら？

```
現在（Unconditional）:
[敵対的画像] → [浄化] → [分類器の予測]
            (ラベル非依存)

拡張案（Conditional）:
[敵対的画像] + [正解ラベル] → [浄化] → [分類器の予測]
                            (より精密)
```

**メリット**:
- より精密な浄化が可能
- クラス固有の特徴を活かせる

**デメリット**:
- ラベル依存 → 予測ラベルが間違うと悪化
- 現在のUnconditionalの汎用性が失われる
- Label-Only攻撃に脆弱になる可能性

### 本実装の判断
**Unconditionalを採用** = **安全性と汎用性を重視**

---

## 6. 先生への最終回答

### Q: 「DDPM, GAN, VAEの敵対的防御についてUnConditionalという認識であっていますか？」

### A: **はい、その認識で正しいです。**

**理由**:
1. **VAE（DermMel）**: Unconditional VAE-GAN
2. **GAN（PCam）**: Unconditional GAN（Defense-GAN論文準拠）
3. **DDPM（DiffPure）**: Unconditional Diffusion（`uncond.pt`を使用）

全て**クラスラベルを条件として使用していません**。

**設計思想**: 敵対的防御において、ラベル非依存であることが「堅牢性」と「汎用性」を確保する上で重要です。

---

## 7. コード証拠

### VAE
```
ファイル: /mnt/data1/gotou/projects/dermmel/vae/train_vae.py
- Encoder/Decoder: ラベルパラメータなし
```

### GAN
```
ファイル: /mnt/data1/gotou/projects/pcam/gan/defense_gan_train_v3.py
- Generator.__init__: 入力は z のみ
- Forward: z から画像を生成

ファイル: /mnt/data1/gotou/projects/pcam/gan/pgd/gan_pgd_eval_v4.py
- 敵対的防御時: L-BFGS で z を最適化
```

### DDPM
```
ファイル: /mnt/data1/gotou/DiffPure/guided_diffusion/script_util.py
- class_cond=False（デフォルト）
- num_classes=(NUM_CLASSES if class_cond else None)

ファイル: /mnt/data1/gotou/DiffPure/diffpure_guided.py
- モデルロード: '256x256_diffusion_uncond.pt'
  ↑ "uncond" = Unconditional
```

---

## 結論

**DDPM, GAN, VAE 敵対的防御：全て Unconditional** ✓

これは設計上の意図的な選択であり、敵対的防御の「汎用性」「独立性」「堅牢性」を確保するためのものです。
