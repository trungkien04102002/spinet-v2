#!/bin/bash
# Phase 4: SPIDER CBAM retrain (v3 — uses v3 RSNA CBAM ckpt as init).
#
# Reproduces v2 best_metrics_cbam.json args:
#   epochs=20, batch_size=32, lr=1e-3, freeze-backbone, loss=weighted
#
# Usage on Vast.ai: bash scripts/run_spider_cbam.sh

set -e

mkdir -p checkpoints/v3_spider experiments/v3_spider

RSNA_CKPT="checkpoints/v3_20260503/cbam/best_model_attention.pth"

if [ ! -f "$RSNA_CKPT" ]; then
    echo "ERROR: missing v3 RSNA CBAM ckpt at $RSNA_CKPT"
    exit 1
fi
echo "Using v3 RSNA CBAM init: $RSNA_CKPT"

python3 train_spider.py \
    --model cbam \
    --rsna-checkpoint "$RSNA_CKPT" \
    --freeze-backbone \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --loss weighted \
    --num-workers 4 2>&1 | tee experiments/v3_spider/run_cbam.log

mv checkpoints_spider/best_model_cbam.pth checkpoints/v3_spider/best_model_cbam.pth 2>/dev/null || true
mv experiments/spider_phase4/best_metrics_cbam.json experiments/v3_spider/best_metrics_cbam.json 2>/dev/null || true
mv experiments/spider_phase4/best_metrics_cbam.txt experiments/v3_spider/best_metrics_cbam.txt 2>/dev/null || true
echo ""
echo "✓ CBAM SPIDER v3 done."
echo "  Ckpt:    checkpoints/v3_spider/best_model_cbam.pth"
echo "  Metrics: experiments/v3_spider/best_metrics_cbam.{json,txt}"
