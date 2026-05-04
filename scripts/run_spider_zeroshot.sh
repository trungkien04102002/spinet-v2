#!/bin/bash
# Phase 2: SPIDER zero-shot evaluation using Hybrid v3 checkpoint.
#
# No SPIDER training — load RSNA-trained Hybrid, encode SPIDER 8-disease
# text prompts via frozen BiomedCLIP, cosine-sim → predict.
#
# Usage on Vast.ai: bash scripts/run_spider_zeroshot.sh

set -e

HYBRID_CKPT="checkpoints/v3_20260503/hybrid/best_model_hybrid.pth"
CBAM_CKPT="checkpoints/v3_20260503/cbam/best_model_attention.pth"
SPIDER_TEST="rsna_preprocessed_spider/spider_zeroshot_test.csv"
VOLUMES_DIR="rsna_preprocessed_spider"
OUTPUT="experiments/v3_20260503/spider_zeroshot_v3.csv"

# Pre-flight checks
echo "=== Pre-flight checks ==="
for f in "$HYBRID_CKPT" "$CBAM_CKPT" "$SPIDER_TEST"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f"
        exit 1
    else
        echo "✓ $f"
    fi
done

if [ ! -d "$VOLUMES_DIR" ]; then
    echo "ERROR: missing volumes dir $VOLUMES_DIR"
    exit 1
else
    echo "✓ $VOLUMES_DIR/ ($(ls "$VOLUMES_DIR" | wc -l) entries)"
fi

mkdir -p "$(dirname "$OUTPUT")"

echo ""
echo "=== Running zero-shot SPIDER evaluation ==="
python3 eval_zeroshot_spider.py \
    --hybrid-checkpoint "$HYBRID_CKPT" \
    --cbam-checkpoint "$CBAM_CKPT" \
    --spider-test "$SPIDER_TEST" \
    --volumes-dir "$VOLUMES_DIR" \
    --output "$OUTPUT" \
    --prompt-template med \
    --batch-size 8 \
    --slice-strategy static 2>&1 | tee experiments/v3_20260503/run_spider_zeroshot.log

echo ""
echo "=== Output ==="
echo "Per-sample CSV: $OUTPUT"
echo "Summary JSON:   ${OUTPUT%.csv}_summary.json"
echo ""
echo "Quick view:"
python3 -c "import json; d=json.load(open('${OUTPUT%.csv}_summary.json')); print(json.dumps(d, indent=2))" 2>/dev/null || echo "(summary JSON not yet generated)"
