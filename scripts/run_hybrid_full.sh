#!/bin/bash
# Phase 2: Hybrid full RSNA retrain (main paper result for Theme 3 — label-space extension).
# Uses v2 fresh_cbam as CBAM-branch input (Plan B consistency).
#
# Usage on Vast.ai: bash scripts/run_hybrid_full.sh

set -e

mkdir -p checkpoints/v3_20260503/hybrid experiments/v3_20260503

V3_CKPT="checkpoints/v3_20260503/cbam/best_model_attention.pth"
V2_CKPT="checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth"

if [ -f "$V3_CKPT" ]; then
    CBAM_CKPT="$V3_CKPT"
    echo "Using v3 retrained CBAM: $V3_CKPT"
elif [ -f "$V2_CKPT" ]; then
    CBAM_CKPT="$V2_CKPT"
    echo "v3 not found, falling back to v2: $V2_CKPT"
else
    echo "ERROR: No CBAM checkpoint found"
    echo "  Tried: $V3_CKPT"
    echo "  Tried: $V2_CKPT"
    exit 1
fi

python3 train_rsna_hybrid.py \
    --cbam-checkpoint "$CBAM_CKPT" \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --no-hflip-swap-labels \
    --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_full.log
