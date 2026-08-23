#!/bin/bash
# Phase 4: SPIDER Hybrid retrain (v3 — uses v3 RSNA Hybrid + CBAM ckpts as init).
#
# Reproduces v2 best_metrics_hybrid_spider_unfreeze.json args:
#   epochs=20, batch_size=16, lr=1e-5, unfreeze-cbam, loss=weighted
#
# Usage on Vast.ai: bash scripts/run_spider_hybrid.sh

set -e

mkdir -p checkpoints/v3_spider experiments/v3_spider

HYBRID_CKPT="checkpoints/v3_20260503/hybrid/best_model_hybrid.pth"
CBAM_CKPT="checkpoints/v3_20260503/cbam/best_model_attention.pth"

for f in "$HYBRID_CKPT" "$CBAM_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing v3 ckpt at $f"
        exit 1
    fi
done
echo "Using v3 RSNA Hybrid init: $HYBRID_CKPT"
echo "Using v3 RSNA CBAM init:   $CBAM_CKPT"

python3 train_spider_hybrid.py \
    --hybrid-checkpoint "$HYBRID_CKPT" \
    --cbam-checkpoint "$CBAM_CKPT" \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-5 \
    --weight-decay 1e-4 \
    --unfreeze-cbam \
    --loss weighted \
    --num-workers 4 \
    --checkpoint-dir checkpoints/v3_spider \
    --metrics-dir experiments/v3_spider 2>&1 | tee experiments/v3_spider/run_hybrid.log

echo ""
echo "✓ Hybrid SPIDER v3 done."
echo "  Ckpt:    checkpoints/v3_spider/best_model_hybrid_spider*.pth"
echo "  Metrics: experiments/v3_spider/best_metrics_hybrid_spider*.{json,txt}"
