# DermMel敵対的防御実験結果サマリー

## 実験概要

- **データセット**: DermMel validation set (Melanoma vs NotMelanoma)
- **対象画像数**: 3,434枚 (クリーン画像で正しく分類された画像のみ)
- **分類器**: ResNet50 (`/mnt/data1/gotou/projects/dermmel/resnet/resnet50_best.pth`)
- **防御手法**: 
  - DDPM (start_t=80, steps=50)
  - JPEG圧縮 (quality=11)
- **攻撃手法**:
  - FGSM (ε=8/255=0.0314)
  - PGD-10 (ε=8/255, α=2/255, 10ステップ, random_start=True)
  - AutoAttack (APGD-CE, ε=8/255)

---

## 結果サマリー

### 1. FGSM攻撃に対する防御

| 指標 | クリーン | 敵対的例 | DDPM防御 | JPEG防御 |
|------|----------|----------|----------|----------|
| **精度** | 100.00% | 43.42% | **53.47%** | 49.53% |
| **改善率** | - | - | **+10.05%** | +6.12% |
| **攻撃成功率** | - | 56.58% | - | - |
| **防御成功率** | - | - | 17.76% | 10.81% |

**Perturbation Norms (Adversarial vs Clean):**
- L2: mean=12.15, L∞: mean=0.0314

**Perturbation Norms (Purified vs Clean):**
- DDPM: L2=9.21±1.70, L∞=0.207±0.056
- JPEG: L2=11.46±1.80, L∞=0.230±0.065

---

### 2. PGD-10攻撃に対する防御

| 指標 | クリーン | 敵対的例 | DDPM防御 | JPEG防御 |
|------|----------|----------|----------|----------|
| **精度** | 100.00% | 0.00% | **64.91%** | 49.56% |
| **改善率** | - | - | **+64.91%** | +49.56% |

**Perturbation Norms (Adversarial vs Clean):**
- L2: mean=8.02±0.12, L∞: mean=0.0314

**Perturbation Norms (Purified vs Clean):**
- DDPM: L2=8.96±1.69, L∞=0.203±0.055
- JPEG: L2=10.64±1.65, L∞=0.220±0.066

**混同行列 (DDPM防御):**
- TN:1631, FP:70, FN:1135, TP:598
- Precision: 90.82%, Recall: 27.41%, F1: 42.11%, Specificity: 97.18%

**混同行列 (JPEG防御):**
- TN:1701, FP:0, FN:1732, TP:1
- Precision: 0.00%, Recall: 0.00%, F1: 0.00%

---

### 3. AutoAttack (APGD-CE)に対する防御

| 指標 | クリーン | 敵対的例 | DDPM防御 | JPEG防御 |
|------|----------|----------|----------|----------|
| **精度** | 100.00% | 0.00% | **61.97%** | 49.56% |
| **改善率** | - | - | **+61.97%** | +49.56% |

**Perturbation Norms (Adversarial vs Clean):**
- DDPM実験: L2=8.69±0.28, L∞=0.0314
- JPEG実験: L2=8.69±0.28, L∞=0.0314

**Perturbation Norms (Purified vs Clean):**
- DDPM: L2=9.21±1.70, L∞=0.207±0.056
- JPEG: L2=10.97±1.63, L∞=0.223±0.067

**混同行列 (DDPM防御):**
- TN:1653, FP:48, FN:1258, TP:475
- Precision: 90.82%, Recall: 27.41%, F1: 42.11%, Specificity: 97.18%

**混同行列 (JPEG防御):**
- TN:1701, FP:0, FN:1732, TP:1
- Precision: 0.00%, Recall: 0.00%, F1: 0.00%

---

## 総合評価

### 防御性能比較 (精度改善率)

```
攻撃手法      | DDPM防御  | JPEG防御  | 優位性
-------------|-----------|-----------|--------
FGSM         | +10.05%   | +6.12%    | DDPM
PGD-10       | +64.91%   | +49.56%   | DDPM
AutoAttack   | +61.97%   | +49.56%   | DDPM
```

### 主要な知見

1. **DDPM防御の優位性**
   - 全ての攻撃手法に対してDDPM防御がJPEG圧縮よりも高い防御性能を示した
   - 特にPGD-10とAutoAttackに対して約60-65%の精度を達成

2. **攻撃強度の違い**
   - FGSM: 最も弱い攻撃 (敵対的精度43.42%)
   - PGD-10 & AutoAttack: 非常に強力 (敵対的精度0.00%)

3. **JPEG圧縮の限界**
   - PGD-10とAutoAttackに対して49.56%の精度
   - 混同行列から、ほとんどがNotMelanomaと予測される傾向(FP=0, FN=1732)
   - True Positiveがほぼ0で、Melanoma検出に失敗

4. **DDPM防御の特徴**
   - より高いSpecificity (97.18%)を維持
   - RecallとF1スコアは低いが、JPEG圧縮よりは改善
   - 摂動ノルムはJPEGより小さく、より原画像に近い復元

5. **Perturbation分析**
   - DDPMの復元画像: L2≈9, L∞≈0.2
   - JPEGの復元画像: L2≈11, L∞≈0.22
   - DDPMの方が原画像により近い復元を実現

---

## 結論

DermMelデータセットにおける敵対的防御実験において、**DDPM (start_t=80, steps=50)** はJPEG圧縮(quality=11)よりも一貫して優れた防御性能を示しました。特に強力なPGD-10およびAutoAttack攻撃に対して約62-65%の精度を達成し、原画像により近い復元を実現しています。

ただし、両防御手法ともMelanoma検出のRecallが低いという課題があり、医療診断における偽陰性(見逃し)のリスクが高い点には注意が必要です。
