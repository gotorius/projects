#!/bin/bash

# PCam DDPM防御評価: FGSM, PGD, AutoAttackを連続実行
# 実行方法: bash run_all_attacks.sh

set -e  # エラーが発生したら停止

echo "=================================="
echo "PCam DDPM Defense Evaluation"
echo "=================================="
echo "Running FGSM, PGD, and AutoAttack sequentially"
echo ""

# 共通パラメータ
EPSILON=0.031
T_PURIFY=50
START_T=80
GPU=0

# ログファイル
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="/mnt/data1/gotou/projects/pcam/ddpm/logs"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/run_all_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# ========== ステップ0: 正解サンプルの準備 ==========
echo "[Step 0/4] Preparing correct samples (500 images)..."
echo "------------------------------------------------------"
if [ ! -f "/mnt/data1/gotou/projects/pcam/ddpm/correct_samples_500.pt" ]; then
    echo "Cached samples not found. Creating..."
    python /mnt/data1/gotou/projects/pcam/ddpm/prepare_correct_samples.py 2>&1 | tee -a $LOG_FILE
    echo "Cached samples created!"
else
    echo "Cached samples already exist. Skipping..."
fi
echo ""

# ========== ステップ1: FGSM攻撃 ==========
echo "[Step 1/4] Running FGSM Attack..."
echo "------------------------------------------------------"
echo "  Epsilon: $EPSILON"
echo "  DDPM params: t_purify=$T_PURIFY, start_t=$START_T"
echo ""

cd /mnt/data1/gotou/projects/pcam/ddpm/fgsm
python ddpm_fgsm_eval.py \
    --epsilon $EPSILON \
    --t_purify $T_PURIFY \
    --start_t $START_T \
    --gpu $GPU \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "FGSM completed!"
echo ""

# ========== ステップ2: PGD攻撃 ==========
echo "[Step 2/4] Running PGD Attack..."
echo "------------------------------------------------------"
echo "  Epsilon: $EPSILON, Alpha: 0.01, Steps: 10"
echo "  DDPM params: t_purify=$T_PURIFY, start_t=$START_T"
echo ""

cd /mnt/data1/gotou/projects/pcam/ddpm/pgd
python ddpm_pgd_eval.py \
    --epsilon $EPSILON \
    --alpha 0.01 \
    --steps 10 \
    --t_purify $T_PURIFY \
    --start_t $START_T \
    --gpu $GPU \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "PGD completed!"
echo ""

# ========== ステップ3: AutoAttack ==========
echo "[Step 3/4] Running AutoAttack..."
echo "------------------------------------------------------"
echo "  Epsilon: $EPSILON, Version: standard"
echo "  DDPM params: t_purify=$T_PURIFY, start_t=$START_T"
echo ""

cd /mnt/data1/gotou/projects/pcam/ddpm/autoattack
python ddpm_autoattack_eval.py \
    --epsilon $EPSILON \
    --version standard \
    --n_examples 500 \
    --t_purify $T_PURIFY \
    --start_t $START_T \
    --gpu $GPU \
    2>&1 | tee -a $LOG_FILE

echo ""
echo "AutoAttack completed!"
echo ""

# ========== 完了 ==========
echo "=================================="
echo "All evaluations completed!"
echo "=================================="
echo ""
echo "Results saved in:"
echo "  - FGSM:       /mnt/data1/gotou/projects/pcam/ddpm/fgsm/results/"
echo "  - PGD:        /mnt/data1/gotou/projects/pcam/ddpm/pgd/results/"
echo "  - AutoAttack: /mnt/data1/gotou/projects/pcam/ddpm/autoattack/results/"
echo ""
echo "Log file: $LOG_FILE"
echo ""

echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
