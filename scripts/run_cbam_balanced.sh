#!/bin/bash
# Balanced CBAM config — try to recover accuracy while keeping Severe F1.
# Target: Severe F1 ~0.28-0.30 (slightly below v2 0.333) BUT Mean Acc ~76-78%
# instead of v2's 68.6%. Net Pareto improvement if achieved.
#
# Hyperparam delta vs run_cbam_v2bug.sh:
# - oversample-factor: 5 -> 3 (less aggressive minority boost)
# - focal-gamma: 1.8 -> 1.5 (gentler focal)
# - HFlip swap labels: keep BUG mode (--no-hflip-swap-labels) for regularization
#
# Usage on Vast.ai: bash scripts/run_cbam_balanced.sh

set -e

mkdir -p checkpoints/v3_20260503/cbam_balanced experiments/v3_20260503

python3 train_rsna_attention.py \
    --epochs 25 \
    --batch-size 64 \
    --lr 1e-3 \
    --focal-gamma 1.5 \
    --use-focal \
    --use-uncertainty true \
    --class-weight-mode sqrt \
    --augmentation medium \
    --oversample-factor 3 \
    --no-hflip-swap-labels \
    --save-dir checkpoints/v3_20260503/cbam_balanced 2>&1 | tee experiments/v3_20260503/run_cbam_balanced.log
