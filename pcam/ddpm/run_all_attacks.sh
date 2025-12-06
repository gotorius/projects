#!/bin/bash
# Run all attacks sequentially for PCam DiffPure evaluation

echo "=============================================="
echo "PCam DiffPure Defense Evaluation"
echo "Starting all attacks: FGSM, PGD, AutoAttack"
echo "=============================================="
echo ""

# Set parameters
NUM_SAMPLES=100
T_PURIFY=250
EPSILON=0.03137  # 8/255
GPU_ID=0

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/mnt/data1/gotou/projects/pcam/ddpm/eval_results/all_attacks_${TIMESTAMP}"

echo "Configuration:"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Purification timestep: ${T_PURIFY}"
echo "  Epsilon: ${EPSILON}"
echo "  GPU: ${GPU_ID}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Save this script for reference
cp $0 ${OUTPUT_DIR}/run_script.sh

# Run all attacks in one go
echo "=============================================="
echo "Running ALL attacks together..."
echo "=============================================="
python ddpm_defense_eval.py \
    --attack all \
    --num_samples ${NUM_SAMPLES} \
    --epsilon ${EPSILON} \
    --t_purify ${T_PURIFY} \
    --use_purification \
    --gpu ${GPU_ID} \
    --output_dir ${OUTPUT_DIR} \
    2>&1 | tee ${OUTPUT_DIR}/full_log.txt

echo ""
echo "=============================================="
echo "All attacks completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
