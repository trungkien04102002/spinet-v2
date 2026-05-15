#!/bin/bash
# =============================================================================
# run_spider_8lbl_3seeds.sh — SPIDER 8-label 4 configs × 2 additional seeds
#
# Note: seed=42 already exists from 2026-05-15 run (commit 37d9ab7+551718f).
# This script adds seed=123 and seed=456 for mean ± std reporting.
#
# REQUIRES: RSNA seed-matched checkpoints (from run_rsna_3seeds_v3.sh):
#   - checkpoints/v3_20260503/cbam/best_model_attention_seed{123,456}.pth
#   - checkpoints/v3_20260503/hybrid/best_model_hybrid_seed{123,456}.pth
#
# Configs:
#   1. Baseline      (train_spider.py --model baseline,  uses CBAM seed-matched)
#   2. CBAM          (train_spider.py --model cbam,      uses CBAM seed-matched)
#   3. BMC-only      (train_spider_hybrid.py --ablate-branch biomedclip_only)
#   4. Hybrid (full) (train_spider_hybrid.py)
#
# Output filenames carry _seed{N} suffix (seed=42 outputs keep original names).
#
# Cost: ~75 min per seed × 2 seeds = ~2.5h on RTX 4090, ~$1.5 Vast.ai
#
# Usage on Vast.ai (AFTER RSNA 3-seed done):
#   bash scripts/run_spider_8lbl_3seeds.sh 2>&1 | tee experiments/v3_spider_8lbl/run_3seeds.log
# =============================================================================

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints/v3_spider_8lbl}"
METRICS_DIR="${METRICS_DIR:-experiments/v3_spider_8lbl}"
RSNA_DIR="${RSNA_DIR:-checkpoints/v3_20260503}"
SEEDS=(${SEEDS:-123 456})

mkdir -p "$CKPT_DIR" "$METRICS_DIR"

echo "=============================================================="
echo "SPIDER 8-label 3-seed runs (adding seeds: ${SEEDS[*]})"
echo "Started: $(date)"
echo "=============================================================="

# ---------- SAFETY PREVIEW ----------
# Filenames for seed=42 carry NO suffix; other seeds get _seed{N}.
echo ""
echo "============== SAFETY PREVIEW =============="
echo ""
echo "EXISTING seed=42 outputs (WILL BE PRESERVED):"
ls -1 "$CKPT_DIR"/best_model_baseline.pth 2>/dev/null || echo "  (no seed=42 baseline)"
ls -1 "$CKPT_DIR"/best_model_cbam.pth 2>/dev/null || echo "  (no seed=42 cbam)"
ls -1 "$CKPT_DIR"/best_model_hybrid_spider_frozen.pth 2>/dev/null || echo "  (no seed=42 hybrid)"
ls -1 "$CKPT_DIR"/best_model_hybrid_spider_frozen_biomedclip_only.pth 2>/dev/null || echo "  (no seed=42 bmc-only)"
echo ""
echo "NEW files that will be created (one per seed):"
for S in "${SEEDS[@]}"; do
    echo "  $CKPT_DIR/best_model_baseline_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_cbam_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_hybrid_spider_frozen_biomedclip_only_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_hybrid_spider_frozen_seed${S}.pth"
done
echo ""
echo "REQUIRED RSNA seed-matched checkpoints (must exist before running):"
for S in "${SEEDS[@]}"; do
    if [[ -f "$RSNA_DIR/cbam/best_model_attention_seed${S}.pth" ]]; then
        echo "  OK  $RSNA_DIR/cbam/best_model_attention_seed${S}.pth"
    else
        echo "  MISSING  $RSNA_DIR/cbam/best_model_attention_seed${S}.pth"
    fi
    if [[ -f "$RSNA_DIR/hybrid/best_model_hybrid_seed${S}.pth" ]]; then
        echo "  OK  $RSNA_DIR/hybrid/best_model_hybrid_seed${S}.pth"
    else
        echo "  MISSING  $RSNA_DIR/hybrid/best_model_hybrid_seed${S}.pth"
    fi
done
echo ""
echo "Already-completed runs in this batch will be SKIPPED (idempotent)."
echo ""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1 set; exiting without running."
    exit 0
fi
echo "Starting in 10 seconds (Ctrl+C to abort)..."
sleep 10
echo "============================================"
echo ""

run_one_seed() {
    local SEED=$1
    local CBAM_CKPT="$RSNA_DIR/cbam/best_model_attention_seed${SEED}.pth"
    local HYBRID_CKPT="$RSNA_DIR/hybrid/best_model_hybrid_seed${SEED}.pth"

    # Sanity check — RSNA seed-matched checkpoints must exist
    if [[ ! -f "$CBAM_CKPT" ]]; then
        echo "ERROR: $CBAM_CKPT not found."
        echo "Run scripts/run_rsna_3seeds_v3.sh first."
        exit 1
    fi
    if [[ ! -f "$HYBRID_CKPT" ]]; then
        echo "ERROR: $HYBRID_CKPT not found."
        echo "Run scripts/run_rsna_3seeds_v3.sh first."
        exit 1
    fi

    echo ""
    echo "##############################################################"
    echo "# SPIDER 8-label sweep — seed=$SEED"
    echo "##############################################################"

    # -------- 1. Baseline SPIDER --------
    if [[ -f "$CKPT_DIR/best_model_baseline_seed${SEED}.pth" ]]; then
        echo "[1/4] Baseline SPIDER seed=$SEED already done, skipping"
    else
        echo "[1/4] Baseline SPIDER seed=$SEED ..."
        python3 train_spider.py --model baseline \
            --rsna-checkpoint "$CBAM_CKPT" \
            --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_baseline_seed${SEED}.log"
    fi

    # -------- 2. CBAM SPIDER --------
    if [[ -f "$CKPT_DIR/best_model_cbam_seed${SEED}.pth" ]]; then
        echo "[2/4] CBAM SPIDER seed=$SEED already done, skipping"
    else
        echo "[2/4] CBAM SPIDER seed=$SEED ..."
        python3 train_spider.py --model cbam \
            --rsna-checkpoint "$CBAM_CKPT" \
            --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_cbam_seed${SEED}.log"
    fi

    # -------- 3. BMC-only SPIDER --------
    if [[ -f "$CKPT_DIR/best_model_hybrid_spider_frozen_biomedclip_only_seed${SEED}.pth" ]]; then
        echo "[3/4] BMC-only SPIDER seed=$SEED already done, skipping"
    else
        echo "[3/4] BMC-only SPIDER seed=$SEED ..."
        python3 train_spider_hybrid.py \
            --hybrid-checkpoint "$HYBRID_CKPT" \
            --cbam-checkpoint "$CBAM_CKPT" \
            --ablate-branch biomedclip_only \
            --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_bmc_only_seed${SEED}.log"
    fi

    # -------- 4. Hybrid (full) SPIDER --------
    if [[ -f "$CKPT_DIR/best_model_hybrid_spider_frozen_seed${SEED}.pth" ]]; then
        echo "[4/4] Hybrid SPIDER seed=$SEED already done, skipping"
    else
        echo "[4/4] Hybrid (full) SPIDER seed=$SEED ..."
        python3 train_spider_hybrid.py \
            --hybrid-checkpoint "$HYBRID_CKPT" \
            --cbam-checkpoint "$CBAM_CKPT" \
            --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_hybrid_seed${SEED}.log"
    fi

    echo ""
    echo "  --- SPIDER seed=$SEED done at $(date) ---"
}

for SEED in "${SEEDS[@]}"; do
    run_one_seed "$SEED"
done

echo ""
echo "=============================================================="
echo "All SPIDER 8-label 3-seed runs complete"
echo "Finished: $(date)"
echo "=============================================================="
echo ""
echo "Outputs:"
ls -lh "$CKPT_DIR"/best_model_*.pth 2>/dev/null
echo ""
echo "Next: aggregate via"
echo "  python3 scripts/aggregate_seeds.py spider \\"
echo "      $METRICS_DIR/best_metrics_baseline.json \\"
echo "      $METRICS_DIR/best_metrics_baseline_seed123.json \\"
echo "      $METRICS_DIR/best_metrics_baseline_seed456.json"
echo "  (repeat for cbam, hybrid_spider_frozen, hybrid_spider_frozen_biomedclip_only)"
