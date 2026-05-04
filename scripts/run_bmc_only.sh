#!/bin/bash
# Phase 1 ablation: BMC-only RSNA retrain.
#
# Same architecture as Hybrid full but with --ablate-branch biomedclip_only:
# CBAM 3D features are zeroed at forward time, so the model effectively learns
# from BiomedCLIP features alone (passed through the same fusion MLP).
#
# All other args MATCH scripts/run_hybrid_full.sh exactly so the comparison
# isolates the contribution of the CBAM branch (BMC-only vs Hybrid full).
#
# Usage on Vast.ai: bash scripts/run_bmc_only.sh

set -e

mkdir -p checkpoints/v3_20260503/bmc_only experiments/v3_20260503

V3_CKPT="checkpoints/v3_20260503/cbam/best_model_attention.pth"
V2_CKPT="checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth"

if [ -f "$V3_CKPT" ]; then
    CBAM_CKPT="$V3_CKPT"
    echo "Using v3 retrained CBAM (init weights only — branch zeroed at forward): $V3_CKPT"
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
    --ablate-branch biomedclip_only \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --focal-gamma 2.0 \
    --class-weight-mode sqrt \
    --oversample-factor 3 \
    --supcon-weight 0.1 \
    --no-hflip-swap-labels \
    --save-dir checkpoints/v3_20260503/bmc_only 2>&1 | tee experiments/v3_20260503/run_bmc_only.log
