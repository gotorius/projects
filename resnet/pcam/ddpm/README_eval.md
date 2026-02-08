# PCam DiffPure Defense Evaluation

DDPMベースの敵対的防御（DiffPure）をPCamデータセットで評価するスクリプト群です。

## 📁 ファイル構成

```
pcam/ddpm/
├── ddpm_defense_eval.py     # メイン評価スクリプト
├── run_all_attacks.sh        # 3つの攻撃を連続実行
├── run_single_attack.sh      # 個別攻撃実行
└── README_eval.md            # このファイル
```

## 🚀 使い方

### 1. すべての攻撃を一度に実行（推奨）

```bash
cd /mnt/data1/gotou/projects/pcam/ddpm
chmod +x run_all_attacks.sh
./run_all_attacks.sh
```

これにより、FGSM、PGD、AutoAttackの3つが順次実行されます。

### 2. 個別の攻撃を実行

```bash
chmod +x run_single_attack.sh

# FGSM (100サンプル、t_purify=250)
./run_single_attack.sh fgsm 100 250

# PGD (50サンプル、t_purify=200)
./run_single_attack.sh pgd 50 200

# AutoAttack (30サンプル、t_purify=250)
./run_single_attack.sh autoattack 30 250
```

### 3. Pythonスクリプトを直接実行

```bash
# すべての攻撃
python ddpm_defense_eval.py --attack all --num_samples 100 --use_purification --t_purify 250

# FGSM のみ
python ddpm_defense_eval.py --attack fgsm --num_samples 100 --use_purification --t_purify 250

# 防御なしで攻撃のみ評価
python ddpm_defense_eval.py --attack fgsm --num_samples 100
```

## ⚙️ 主要パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `--attack` | 攻撃タイプ (fgsm/pgd/autoattack/all) | all |
| `--num_samples` | 評価サンプル数 | 100 |
| `--epsilon` | 摂動の大きさ | 8/255 |
| `--t_purify` | 浄化のタイムステップ (0で浄化なし) | 250 |
| `--use_purification` | DiffPure防御を有効化 | False |
| `--pgd_alpha` | PGDステップサイズ | 2/255 |
| `--pgd_steps` | PGD反復回数 | 20 |
| `--gpu` | GPU ID | 0 |

## 📊 出力

各実行後、以下が保存されます：

```
eval_results/
└── fgsm_eps0.0314_20251207_120000/
    ├── config.json              # 実行設定
    ├── fgsm_results.json        # FGSM結果
    ├── pgd_results.json         # PGD結果
    ├── autoattack_results.json  # AutoAttack結果
    └── full_log.txt             # 実行ログ
```

### 結果の内容

- **clean_acc**: クリーン画像の精度
- **clean_purified_acc**: クリーン画像 + DiffPure の精度
- **adv_acc_no_defense**: 攻撃後の精度（防御なし）
- **adv_defended_acc**: 攻撃後 + DiffPure の精度
- **defense_improvement**: 防御による精度向上
- **confusion_matrices**: 混同行列

## 🔬 評価の流れ

各攻撃について、以下の4ステップを評価：

1. **クリーン画像**: 攻撃なし、防御なし → ベースライン精度
2. **クリーン + DiffPure**: 攻撃なし、防御あり → 浄化の副作用を確認
3. **攻撃のみ**: 攻撃あり、防御なし → 攻撃の効果を確認
4. **攻撃 + 防御**: 攻撃あり、防御あり → 防御の効果を確認

## 📝 注意事項

### サンプル数について
- **AutoAttack**: 非常に時間がかかるため、30-50サンプル推奨
- **PGD**: 中程度の時間、50-100サンプル推奨
- **FGSM**: 高速、100-500サンプル可能

### t_purify の選択
- **t=0**: 浄化なし（防御なし）
- **t=100-200**: 軽い浄化（高速、やや効果的）
- **t=250-500**: 中程度の浄化（バランス）
- **t=500+**: 強い浄化（遅い、効果的だが画質劣化の可能性）

## 🎯 期待される結果

```
Attack       | Clean | Attack | Defense | Improvement
-------------|-------|--------|---------|------------
FGSM         | 0.95  | 0.30   | 0.75    | +0.45
PGD          | 0.95  | 0.10   | 0.60    | +0.50
AutoAttack   | 0.95  | 0.05   | 0.50    | +0.45
```

※ 実際の値は訓練されたモデルの品質により変動

## 🐛 トラブルシューティング

### AutoAttack が動かない
```bash
pip install git+https://github.com/fra31/auto-attack
```

### CUDA out of memory
- `--batch_size` を小さくする（デフォルト16 → 8 or 4）
- `--num_samples` を減らす
- `--t_purify` を小さくする

### 実行時間が長すぎる
- `--num_samples` を減らす（100 → 50 → 30）
- `--t_purify` を小さくする（250 → 150 → 100）
- AutoAttackをスキップして FGSM/PGD のみ実行

## 📚 参考文献

- **DiffPure**: Nie, W., et al. "Diffusion Models for Adversarial Purification." ICML 2022.
- **DDPM**: Ho, J., et al. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
- **AutoAttack**: Croce, F., & Hein, M. "Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks." ICML 2020.
