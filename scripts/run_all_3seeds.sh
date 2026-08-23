#!/bin/bash
# =============================================================================
# Full 3-seed runs for Baseline + CBAM + Hybrid + SPIDER (Hybrid only).
#
# Usage:
#   bash scripts/run_all_3seeds.sh
#
# Cost: ~25h on Vast.ai 1× RTX 4090, ~$8 total.
# Cheaper alternative: scripts/run_hybrid_3seeds.sh (Hybrid only, ~$2).
#
# Already done at seed=42 (v3 results, commit ed705ef): all 4 configs.
# This script only adds seed=123 and seed=456 runs.
# =============================================================================
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-checkpoints/v3_20260503}"
SPIDER_DIR="${SPIDER_DIR:-checkpoints/v3_spider}"

mkdir -p "$ROOT_DIR/baseline" "$ROOT_DIR/cbam" "$ROOT_DIR/hybrid" "$ROOT_DIR/bmc_only" "$SPIDER_DIR"

run_seed() {
    local SEED=$1

    echo ""
    echo "=========================================="
    echo " ALL CONFIGS — seed=$SEED"
    echo "=========================================="

    # 1. Baseline RSNA (~16 min on Vast.ai)
    echo "[1/4] Baseline RSNA seed=$SEED"
    python3 train_rsna_baseline.py \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-3 \
        --save-dir "$ROOT_DIR/baseline" \
        --seed "$SEED" \
        2>&1 | tee "$ROOT_DIR/baseline/run_baseline_seed${SEED}.log"

    # 2. CBAM RSNA (v2bug args, ~56 min)
    echo "[2/4] CBAM RSNA seed=$SEED"
    python3 train_rsna_attention.py \
        --epochs 25 \
        --batch-size 32 \
        --lr 1e-3 \
        --focal-gamma 1.8 \
        --class-weight-mode sqrt \
        --oversample-factor 5 \
        --no-hflip-swap-labels \
        --save-dir "$ROOT_DIR/cbam" \
        --seed "$SEED" \
        2>&1 | tee "$ROOT_DIR/cbam/run_cbam_seed${SEED}.log"

    CBAM_SEED_CKPT="$ROOT_DIR/cbam/best_model_attention_seed${SEED}.pth"

    # 3. Hybrid full RSNA (v2 hybrid_fixed_e7 args, ~25 min)
    echo "[3/4] Hybrid RSNA seed=$SEED"
    python3 train_rsna_hybrid.py \
        --cbam-checkpoint "$CBAM_SEED_CKPT" \
        --save-dir "$ROOT_DIR/hybrid" \
        --epochs 20 \
        --batch-size 32 \
        --lr 1e-4 \
        --focal-gamma 2.0 \
        --class-weight-mode sqrt \
        --oversample-factor 3 \
        --supcon-weight 0.1 \
        --no-hflip-swap-labels \
        --seed "$SEED" \
        2>&1 | tee "$ROOT_DIR/hybrid/run_hybrid_seed${SEED}.log"

    HYBRID_SEED_CKPT="$ROOT_DIR/hybrid/best_model_hybrid_seed${SEED}.pth"

    # 4. SPIDER Hybrid (transfer + frozen, ~3.5h)
    echo "[4/4] SPIDER Hybrid seed=$SEED"
    python3 train_spider.py \
        --model hybrid \
        --rsna-checkpoint "$HYBRID_SEED_CKPT" \
        --freeze-backbone \
        --epochs 15 \
        --batch-size 16 \
        --lr 1e-3 \
        --checkpoint-dir "$SPIDER_DIR" \
        --metrics-dir "experiments/v3_spider" \
        --seed "$SEED" \
        2>&1 | tee "$SPIDER_DIR/run_spider_hybrid_seed${SEED}.log"

    echo "  ✓ seed=$SEED done"
}

for SEED in 123 456; do
    run_seed "$SEED"
done

echo ""
echo "=========================================="
echo " All 3-seed runs complete"
echo "=========================================="
echo "Aggregate with:"
echo "  python3 scripts/aggregate_seeds.py --pattern 'best_metrics_*.json' --dir $ROOT_DIR"
