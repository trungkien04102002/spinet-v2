#!/bin/bash
# Phase 4: SPIDER Baseline retrain (v3 — uses v3 RSNA baseline ckpt as init).
#
# Reproduces v2 best_metrics_baseline.json args:
#   epochs=15, batch_size=32, lr=1e-3, freeze-backbone, loss=weighted
#
# Usage on Vast.ai: bash scripts/run_spider_baseline.sh

set -e

mkdir -p checkpoints/v3_spider experiments/v3_spider

V3_CKPT="checkpoints/v3_20260503/baseline/best_model.pth"
V2_CKPT="checkpoints/fresh_baseline/best_model_baseline_full_e25.pth"

if [ -f "$V3_CKPT" ]; then
    RSNA_CKPT="$V3_CKPT"
    echo "Using v3 RSNA baseline init: $V3_CKPT"
elif [ -f "$V2_CKPT" ]; then
    RSNA_CKPT="$V2_CKPT"
    echo "v3 baseline ckpt not found, falling back to v2: $V2_CKPT"
else
    echo "ERROR: No RSNA baseline checkpoint found"
    echo "  Tried: $V3_CKPT"
    echo "  Tried: $V2_CKPT"
    exit 1
fi

python3 train_spider.py \
    --model baseline \
    --rsna-checkpoint "$RSNA_CKPT" \
    --freeze-backbone \
    --epochs 15 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --loss weighted \
    --num-workers 4 \
    --checkpoint-dir checkpoints/v3_spider \
    --metrics-dir experiments/v3_spider 2>&1 | tee experiments/v3_spider/run_baseline.log

echo ""
echo "✓ Baseline SPIDER v3 done."
echo "  Ckpt:    checkpoints/v3_spider/best_model_baseline.pth"
echo "  Metrics: experiments/v3_spider/best_metrics_baseline.{json,txt}"
