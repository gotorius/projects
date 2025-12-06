#!/bin/bash
# Run individual attacks for PCam DiffPure evaluation

# Usage examples:
#   ./run_single_attack.sh fgsm 100 250
#   ./run_single_attack.sh pgd 50 200
#   ./run_single_attack.sh autoattack 30 250

ATTACK_TYPE=${1:-fgsm}          # Default: fgsm
NUM_SAMPLES=${2:-100}           # Default: 100
T_PURIFY=${3:-250}              # Default: 250
EPSILON=${4:-0.03137}           # Default: 8/255
GPU_ID=${5:-0}                  # Default: GPU 0

echo "=============================================="
echo "PCam DiffPure Evaluation: ${ATTACK_TYPE^^}"
echo "=============================================="
echo "Configuration:"
echo "  Attack: ${ATTACK_TYPE}"
echo "  Samples: ${NUM_SAMPLES}"
echo "  Purification timestep: ${T_PURIFY}"
echo "  Epsilon: ${EPSILON}"
echo "  GPU: ${GPU_ID}"
echo ""

python ddpm_defense_eval.py \
    --attack ${ATTACK_TYPE} \
    --num_samples ${NUM_SAMPLES} \
    --epsilon ${EPSILON} \
    --t_purify ${T_PURIFY} \
    --use_purification \
    --gpu ${GPU_ID}

echo ""
echo "=============================================="
echo "${ATTACK_TYPE^^} evaluation completed!"
echo "=============================================="
