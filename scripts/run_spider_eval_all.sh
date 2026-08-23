#!/bin/bash
# Phase 4 post-hoc: compute AUC/AUPRC for all 3 SPIDER v3 ckpts.
# Run after run_spider_baseline.sh + run_spider_cbam.sh + run_spider_hybrid.sh done.
#
# Usage on Vast.ai: bash scripts/run_spider_eval_all.sh

set -e

mkdir -p experiments/v3_spider

BASELINE_CKPT="checkpoints/v3_spider/best_model_baseline.pth"
CBAM_CKPT="checkpoints/v3_spider/best_model_cbam.pth"
HYBRID_CKPT="checkpoints/v3_spider/best_model_hybrid_spider_unfreeze.pth"
RSNA_CBAM="checkpoints/v3_20260503/cbam/best_model_attention.pth"

# Fallback Hybrid name if --unfreeze-cbam not active
if [ ! -f "$HYBRID_CKPT" ]; then
    HYBRID_CKPT="checkpoints/v3_spider/best_model_hybrid_spider.pth"
fi

echo "=== Pre-flight checks ==="
for f in "$BASELINE_CKPT" "$CBAM_CKPT" "$HYBRID_CKPT" "$RSNA_CBAM"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    else
        echo "✓ $f"
    fi
done

echo ""
echo "=== [1/3] Eval Baseline SPIDER v3 ==="
python3 eval_spider_auc.py \
    --model baseline \
    --checkpoint "$BASELINE_CKPT" \
    --output experiments/v3_spider/auc_auprc_baseline.json 2>&1 | tee experiments/v3_spider/run_eval_baseline.log

echo ""
echo "=== [2/3] Eval CBAM SPIDER v3 ==="
python3 eval_spider_auc.py \
    --model cbam \
    --checkpoint "$CBAM_CKPT" \
    --output experiments/v3_spider/auc_auprc_cbam.json 2>&1 | tee experiments/v3_spider/run_eval_cbam.log

echo ""
echo "=== [3/3] Eval Hybrid SPIDER v3 ==="
python3 eval_spider_auc.py \
    --model hybrid \
    --checkpoint "$HYBRID_CKPT" \
    --cbam-checkpoint "$RSNA_CBAM" \
    --output experiments/v3_spider/auc_auprc_hybrid.json 2>&1 | tee experiments/v3_spider/run_eval_hybrid.log

echo ""
echo "=== Summary ==="
echo "Output JSON:"
ls -la experiments/v3_spider/auc_auprc_*.json
echo ""
echo "Quick view (mean across 4 conditions):"
for j in experiments/v3_spider/auc_auprc_baseline.json experiments/v3_spider/auc_auprc_cbam.json experiments/v3_spider/auc_auprc_hybrid.json; do
    name=$(basename "$j" .json | sed 's/auc_auprc_//')
    python3 -c "
import json
d = json.load(open('$j'))
m = d['macro_across_conditions']
print(f'  {\"$name\":<10} F1 {m[\"f1\"]:.3f} | Acc {m[\"accuracy\"]:.3f} | AUC {m[\"auc\"]:.3f} | AUPRC {m[\"auprc\"]:.3f}')
" 2>/dev/null
done
