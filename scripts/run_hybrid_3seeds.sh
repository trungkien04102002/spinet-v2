#!/bin/bash
# =============================================================================
# Run RSNA Hybrid (CBAM + BiomedCLIP) with 3 seeds for paper mean±std.
#
# Usage:
#   bash scripts/run_hybrid_3seeds.sh
#
# Output:
#   checkpoints/v3_20260503/hybrid/best_model_hybrid.pth         # seed=42 (already exists from v3)
#   checkpoints/v3_20260503/hybrid/best_model_hybrid_seed123.pth
#   checkpoints/v3_20260503/hybrid/best_model_hybrid_seed456.pth
#   experiments/v3_20260503/hybrid/best_metrics_hybrid_seed{42,123,456}.json
#
# Cost: ~6h on Vast.ai 1× RTX 4090, ~$2 total.
#
# Note: seed=42 run is already done from v3 (commit ed705ef). This script
#       only adds seed=123 and seed=456 runs (~4h total).
# =============================================================================
set -euo pipefail

# Paths
CBAM_CKPT="${CBAM_CKPT:-checkpoints/v3_20260503/cbam/best_model_attention.pth}"
SAVE_DIR="${SAVE_DIR:-checkpoints/v3_20260503/hybrid}"

if [ ! -f "$CBAM_CKPT" ]; then
    echo "ERROR: CBAM checkpoint not found at $CBAM_CKPT"
    echo "Either train CBAM first (scripts/run_cbam_v2bug.sh) or set CBAM_CKPT env var."
    exit 1
fi

# v2 hybrid_fixed_e7 reference args (from project_v3_retrain_workflow memory):
#   focal_gamma=2.0, class_weight=sqrt, oversample=3, supcon_weight=0.1,
#   --no-hflip-swap-labels, epochs=20, batch=32, lr=1e-4

mkdir -p "$SAVE_DIR"

for SEED in 123 456; do
    echo ""
    echo "=========================================="
    echo " Hybrid retrain — seed=$SEED"
    echo "=========================================="

    python3 train_rsna_hybrid.py \
        --cbam-checkpoint "$CBAM_CKPT" \
        --save-dir "$SAVE_DIR" \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-4 \
        --weight-decay 1e-4 \
        --focal-gamma 2.0 \
        --class-weight-mode sqrt \
        --oversample-factor 3 \
        --supcon-weight 0.1 \
        --no-hflip-swap-labels \
        --seed "$SEED" \
        2>&1 | tee "$SAVE_DIR/run_hybrid_seed${SEED}.log"

    echo "  ✓ seed=$SEED done"
    echo "  Metrics: $SAVE_DIR/best_metrics_hybrid_seed${SEED}.json"
    echo "  Checkpoint: $SAVE_DIR/best_model_hybrid_seed${SEED}.pth"
done

echo ""
echo "=========================================="
echo " 3-seed runs complete"
echo "=========================================="
echo "Aggregate (mean ± std) with:"
echo "  python3 scripts/aggregate_seeds.py --pattern 'best_metrics_hybrid*.json' --dir $SAVE_DIR"
