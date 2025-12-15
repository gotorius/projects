# PCam DDPM防御評価 - クイックスタート

## 🚀 実行方法

### 1回で全て実行（推奨）

```bash
cd /mnt/data1/gotou/projects/pcam/ddpm
bash run_all_attacks.sh
```

これで以下が自動実行されます：
1. 正解サンプル500枚の準備（初回のみ）
2. FGSM攻撃 + DDPM防御の評価
3. PGD攻撃 + DDPM防御の評価
4. AutoAttack + DDPM防御の評価

## 📊 評価される内容

各攻撃で4つの精度を測定：
- ✅ クリーン画像の精度
- ✅ クリーン画像をDDPM浄化後の精度
- ❌ 敵対的画像の精度（防御なし）
- 🛡️ 敵対的画像をDDPM浄化後の精度（防御あり）

追加メトリクス：
- 混同行列
- L2ノルム
- 防御改善度

## 📁 結果の場所

```
/mnt/data1/gotou/projects/pcam/ddpm/
├── fgsm/results/          # FGSM結果
├── pgd/results/           # PGD結果
├── autoattack/results/    # AutoAttack結果
└── logs/                  # 実行ログ
```

## ⚙️ パラメータ（デフォルト）

- DDPM浄化: `t_purify=50`, `start_t=80`
- 攻撃強度: `epsilon=0.031` (8/255)
- テストサンプル: 500枚（全て同じ）

## 個別実行

```bash
# FGSM のみ
cd /mnt/data1/gotou/projects/pcam/ddpm/fgsm
python ddpm_fgsm_eval.py

# PGD のみ
cd /mnt/data1/gotou/projects/pcam/ddpm/pgd
python ddpm_pgd_eval.py

# AutoAttack のみ
cd /mnt/data1/gotou/projects/pcam/ddpm/autoattack
python ddpm_autoattack_eval.py
```

## ⏱️ 実行時間の目安

- FGSM: 約15分
- PGD: 約20分
- AutoAttack: 約2-3時間

合計: 約3-4時間

## 必要な環境

- PyTorch
- torchvision
- autoattack (`pip install git+https://github.com/fra31/auto-attack`)
- scikit-learn
- tqdm

## トラブルシューティング

**autoattackがない場合:**
```bash
pip install git+https://github.com/fra31/auto-attack
```

**GPU メモリ不足:**
各スクリプトに `--batch_size 8` を追加

**キャッシュファイルがない:**
自動で作成されます（初回のみ数分かかります）
