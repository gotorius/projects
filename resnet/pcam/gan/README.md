# Defense-GAN for PCam Dataset

Defense-GAN を用いた PCam (PatchCamelyon) データセットに対する敵対的防御の実装です。

## Reference

```
"Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models"
Pouya Samangouei, Maya Kabkab, Rama Chellappa
ICLR 2018
```

## 概要

Defense-GAN は、GAN の生成器を使って入力画像を「浄化」することで、敵対的摂動を除去する防御手法です。

### 浄化プロセス

1. **入力画像 x に対する最適化**: 入力画像に最も近い画像を生成できる潜在変数 z* を勾配降下法で探索
2. **再構成**: z* から生成された画像 G(z*) を分類器への入力として使用

敵対的摂動は GAN の学習時に見ていないノイズパターンであるため、再構成によって自然と除去されます。

## ファイル構成

```
gan/
├── defense_gan_train.py   # GAN訓練コード
├── defense_gan_eval.py    # 敵対的防御評価コード
├── README.md              # このファイル
└── checkpoints/           # モデル保存先
    └── best_model.pth
```

## 使用方法

### 1. GAN の訓練

```bash
python defense_gan_train.py \
    --data_dir /mnt/data1/Public/MedImages/PCam_ImageFolder/train \
    --save_dir /mnt/data1/gotou/projects/pcam/gan/checkpoints \
    --epochs 100 \
    --batch_size 64 \
    --latent_dim 128 \
    --ngf 64 \
    --ndf 64 \
    --gpu_id 0
```

#### 訓練オプション

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--data_dir` | PCam train | 訓練データパス |
| `--save_dir` | checkpoints | モデル保存先 |
| `--epochs` | 100 | エポック数 |
| `--batch_size` | 64 | バッチサイズ |
| `--latent_dim` | 128 | 潜在空間の次元 |
| `--ngf` | 64 | Generator基本チャンネル数 |
| `--ndf` | 64 | Discriminator基本チャンネル数 |
| `--lr_g` | 2e-4 | Generatorの学習率 |
| `--lr_d` | 2e-4 | Discriminatorの学習率 |
| `--gpu_id` | 0 | 使用するGPU ID |

### 2. 敵対的防御の評価

#### 全攻撃 (FGSM, PGD, AutoAttack) で評価

```bash
python defense_gan_eval.py \
    --attack all \
    --use_defense \
    --num_samples 100 \
    --gan_ckpt /mnt/data1/gotou/projects/pcam/gan/checkpoints/best_model.pth \
    --clf_ckpt /mnt/data1/gotou/kaggle/checkpoints/best_resnet50_pcam.pth \
    --gpu 0
```

#### FGSM のみで評価

```bash
python defense_gan_eval.py \
    --attack fgsm \
    --use_defense \
    --epsilon 0.03137 \
    --num_samples 100 \
    --gpu 0
```

#### PGD のみで評価

```bash
python defense_gan_eval.py \
    --attack pgd \
    --use_defense \
    --epsilon 0.03137 \
    --pgd_steps 20 \
    --pgd_alpha 0.00784 \
    --num_samples 100 \
    --gpu 0
```

#### AutoAttack のみで評価

```bash
python defense_gan_eval.py \
    --attack autoattack \
    --use_defense \
    --epsilon 0.03137 \
    --num_samples 100 \
    --gpu 0
```

#### 評価オプション

| オプション | デフォルト | 説明 |
|------------|-----------|------|
| `--attack` | all | 攻撃タイプ (fgsm, pgd, autoattack, all) |
| `--epsilon` | 8/255 | 摂動の大きさ |
| `--use_defense` | False | Defense-GAN を有効にする |
| `--rec_iters` | 200 | 再構成の勾配降下イテレーション数 |
| `--rec_rr` | 10 | ランダムリスタート回数 |
| `--rec_lr` | 0.01 | 再構成の学習率 |
| `--pgd_steps` | 20 | PGDのステップ数 |
| `--pgd_alpha` | 2/255 | PGDのステップサイズ |
| `--num_samples` | 100 | 評価サンプル数 |
| `--batch_size` | 16 | バッチサイズ |
| `--gpu` | 0 | 使用するGPU ID |

## 出力

### 訓練出力

- `checkpoints/best_model.pth` - 最終モデル
- `checkpoints/gan_epochXXXX.pth` - エポックごとのチェックポイント
- `checkpoints/samples/` - 生成画像サンプル
- `checkpoints/training_loss.png` - ロスのプロット
- `checkpoints/training_history.json` - 訓練履歴

### 評価出力

```
gan/
├── fgsm/
│   └── eps0.0314_YYYYMMDD_HHMMSS/
│       ├── fgsm_results.json
│       └── config.json
├── pgd/
│   └── eps0.0314_YYYYMMDD_HHMMSS/
│       ├── pgd_results.json
│       └── config.json
└── autoattack/
    └── eps0.0314_YYYYMMDD_HHMMSS/
        ├── autoattack_results.json
        └── config.json
```

### 結果JSON形式

```json
{
  "clean_acc": 0.89,
  "clean_defended_acc": 0.85,
  "adv_acc_no_defense": 0.12,
  "adv_defended_acc": 0.65,
  "defense_improvement": 0.53,
  "time": 123.45,
  "confusion_matrices": {
    "clean": {"tn": 45, "fp": 5, "fn": 6, "tp": 44, ...},
    "adv_no_defense": {...},
    "clean_defended": {...},
    "adv_defended": {...}
  }
}
```

## Defense-GAN のハイパーパラメータ

Defense-GAN の性能は以下のパラメータに依存します：

| パラメータ | 推奨値 | 説明 |
|------------|--------|------|
| `rec_iters` | 200-500 | 多いほど良い再構成だが遅い |
| `rec_rr` | 10-20 | ランダムリスタートで局所解を回避 |
| `rec_lr` | 0.01-0.1 | 小さすぎると収束が遅い |

## 注意事項

1. **訓練時間**: GAN の訓練には時間がかかります（PCamで約2-4時間/100epochs）
2. **評価時間**: Defense-GAN の再構成は計算コストが高いため、評価に時間がかかります
3. **メモリ使用量**: 複数のランダムリスタートを使用するため、メモリ使用量が大きくなる場合があります
4. **AutoAttack**: `autoattack` ライブラリが必要です

## インストール

```bash
pip install torch torchvision tqdm matplotlib scikit-learn

# AutoAttack (オプション)
pip install git+https://github.com/fra31/auto-attack
```

## モデルアーキテクチャ

### Generator (DCGAN-based)

```
Input: z ∈ R^128
      ↓ Linear
(ngf*16) × 7 × 7
      ↓ ConvTranspose2d
(ngf*8) × 14 × 14
      ↓ ConvTranspose2d
(ngf*4) × 28 × 28
      ↓ ConvTranspose2d
(ngf*2) × 56 × 56
      ↓ ConvTranspose2d
(ngf) × 112 × 112
      ↓ ConvTranspose2d
3 × 224 × 224
```

### Discriminator (DCGAN-based)

```
Input: 3 × 224 × 224
      ↓ Conv2d
(ndf) × 112 × 112
      ↓ Conv2d
(ndf*2) × 56 × 56
      ↓ Conv2d
(ndf*4) × 28 × 28
      ↓ Conv2d
(ndf*8) × 14 × 14
      ↓ Conv2d
(ndf*16) × 7 × 7
      ↓ Conv2d
1 × 1 × 1
```
