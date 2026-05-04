#!/bin/bash
# Phase 2: Hybrid full RSNA retrain (main paper result for Theme 3 — label-space extension).
# Uses v2 fresh_cbam as CBAM-branch input (Plan B consistency).
#
# Usage on Vast.ai: bash scripts/run_hybrid_full.sh

set -e

mkdir -p checkpoints/v3_20260503/hybrid experiments/v3_20260503

CBAM_CKPT="checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth"
if [ ! -f "$CBAM_CKPT" ]; then
    echo "ERROR: CBAM checkpoint not found at $CBAM_CKPT"
    echo "Either scp from local or use checkpoints/v3_20260503/cbam/best_model_attention.pth if v3 retrain is done."
    exit 1
fi

python3 train_rsna_hybrid.py \
    --cbam-checkpoint "$CBAM_CKPT" \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-dir checkpoints/v3_20260503/hybrid 2>&1 | tee experiments/v3_20260503/run_hybrid_full.log
