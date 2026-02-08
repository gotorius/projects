# Grand X-Ray Slam: Division A

## Overview
14種類の胸部X線疾患の多ラベル分類コンペティション

- **参加期程**：August 20, 2025 - October 10, 2025
- **賞金**：$1,500 (Division A)
- **評価指標**：14種類の条件ごとのAUC（ROC曲線下面積）の平均値
- **タスク**：多ラベル分類（各画像は複数のラベルを持つ可能性あり）

## Dataset
- **訓練セット**：107,374枚（約138GB）
- **テストセット**：46,233枚（約60GB）
- **形式**：PNG/JPG形式の胸部X線画像（de-identified）

## 14種類の検出対象疾患
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Enlarged Cardiomediastinum
6. Fracture
7. Lung Lesion
8. Lung Opacity
9. No Finding
10. Pleural Effusion
11. Pleural Other
12. Pneumonia
13. Pneumothorax
14. Support Devices

## Submission Format
CSV形式: Image_name, Atelectasis, Cardiomegaly, ..., Support Devices
各列は0-1の確率値

例：
```
00000005_001_001.jpg,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
```

## Rules
- ✓ チーム規模：1人～4人
- ✗ 外部データ：使用不可
- ✓ 利用目的：コンペティション＆研究のみ

## ディレクトリ構成
```
GrandX-Ray/
├── README.md (このファイル)
├── requirements.txt (Python依存関係)
├── notebooks/ (分析・検証用ノートブック)
├── data/
│   ├── train/ (訓練データ)
│   ├── test/ (テストデータ)
│   ├── train_labels.csv (訓練用ラベルCSV)
│   └── sample_submission.csv (提出フォーマット参考)
├── models/ (学習済みモデル保存先)
├── submissions/ (提出ファイル)
├── src/ (本体コード)
│   ├── preprocessing.py
│   ├── models.py
│   ├── utils.py
│   └── main.py
└── logs/ (ログファイル)
```

## 準備完了後のステップ
1. Kaggleから公式データセットをダウンロード
2. データセット構造の確認
3. EDA（探索的データ分析）
4. 前処理パイプライン開発
5. ベースラインモデル構築
