#!/bin/bash
# Phase 4: SPIDER Baseline retrain (v3 — uses v3 RSNA baseline ckpt as init).
#
# Reproduces v2 best_metrics_baseline.json args:
#   epochs=15, batch_size=32, lr=1e-3, freeze-backbone, loss=weighted
#
# Usage on Vast.ai: bash scripts/run_spider_baseline.sh

set -e

mkdir -p checkpoints/v3_spider experiments/v3_spider

RSNA_CKPT="checkpoints/v3_20260503/baseline/best_model.pth"

if [ ! -f "$RSNA_CKPT" ]; then
    echo "ERROR: missing v3 RSNA baseline ckpt at $RSNA_CKPT"
    exit 1
fi
echo "Using v3 RSNA baseline init: $RSNA_CKPT"

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
