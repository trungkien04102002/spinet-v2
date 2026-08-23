#!/bin/bash
# Reproduce v2 fresh_cbam working config (HFlip BUG mode + class-weight=sqrt + gamma=1.8 + 25 epochs).
# Use this to recover Severe F1 ~0.333 that v3 retrain regressed on.
#
# Usage on Vast.ai: bash scripts/run_cbam_v2bug.sh

set -e

mkdir -p checkpoints/v3_20260503/cbam experiments/v3_20260503

python3 train_rsna_attention.py \
    --epochs 25 \
    --batch-size 64 \
    --lr 1e-3 \
    --focal-gamma 1.8 \
    --use-focal \
    --use-uncertainty true \
    --class-weight-mode sqrt \
    --augmentation medium \
    --oversample-factor 5 \
    --no-hflip-swap-labels \
    --save-dir checkpoints/v3_20260503/cbam 2>&1 | tee experiments/v3_20260503/run_cbam_v2bug.log
