#!/bin/bash
# =============================================================================
# run_spider_8lbl_3seeds_sharedrsna.sh
#
# SPIDER 8-label 3-seed variant that REUSES seed=42 RSNA checkpoints.
# Use this when RSNA 3-seed is not feasible (disk constraint, time, etc.).
#
# Variance source: SPIDER head/projection training only (not RSNA backbone).
# Defensible for paper: "RSNA fixed at v3 seed=42 checkpoints; SPIDER trained
# with 3 seeds to assess transfer-learning variance."
#
# Configs (per seed):
#   1. Baseline (train_spider.py --model baseline)
#   2. CBAM     (train_spider.py --model cbam)
#   3. BMC-only (train_spider_hybrid.py --ablate-branch biomedclip_only)
#   4. Hybrid   (train_spider_hybrid.py)
#
# Requires (existing seed=42 RSNA ckpts):
#   checkpoints/v3_20260503/cbam/best_model_attention.pth
#   checkpoints/v3_20260503/hybrid/best_model_hybrid.pth
#
# Cost: ~2.5h on RTX 4090, ~$1.5 Vast.ai.
#
# Usage:
#   bash scripts/run_spider_8lbl_3seeds_sharedrsna.sh 2>&1 | tee experiments/v3_spider_8lbl/run_3seeds_shared.log
# =============================================================================

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-checkpoints/v3_spider_8lbl}"
METRICS_DIR="${METRICS_DIR:-experiments/v3_spider_8lbl}"
RSNA_CBAM_CKPT="${RSNA_CBAM_CKPT:-checkpoints/v3_20260503/cbam/best_model_attention.pth}"
RSNA_HYBRID_CKPT="${RSNA_HYBRID_CKPT:-checkpoints/v3_20260503/hybrid/best_model_hybrid.pth}"
SEEDS=(${SEEDS:-123 456})

mkdir -p "$CKPT_DIR" "$METRICS_DIR"

# Sanity check
if [[ ! -f "$RSNA_CBAM_CKPT" ]]; then
    echo "ERROR: $RSNA_CBAM_CKPT not found"
    exit 1
fi
if [[ ! -f "$RSNA_HYBRID_CKPT" ]]; then
    echo "ERROR: $RSNA_HYBRID_CKPT not found"
    exit 1
fi

echo "=============================================================="
echo "SPIDER 8-label 3-seed (SHARED RSNA seed=42 checkpoints)"
echo "Adding seeds: ${SEEDS[*]}"
echo "Started: $(date)"
echo "=============================================================="

# Safety preview
echo ""
echo "============== SAFETY PREVIEW =============="
echo ""
echo "EXISTING seed=42 SPIDER outputs (PRESERVED):"
ls -1 "$CKPT_DIR"/best_model_baseline.pth 2>/dev/null
ls -1 "$CKPT_DIR"/best_model_cbam.pth 2>/dev/null
ls -1 "$CKPT_DIR"/best_model_hybrid_spider_frozen.pth 2>/dev/null
ls -1 "$CKPT_DIR"/best_model_hybrid_spider_frozen_biomedclip_only.pth 2>/dev/null
echo ""
echo "NEW files (one per new seed):"
for S in "${SEEDS[@]}"; do
    echo "  $CKPT_DIR/best_model_baseline_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_cbam_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_hybrid_spider_frozen_seed${S}.pth"
    echo "  $CKPT_DIR/best_model_hybrid_spider_frozen_seed${S}_biomedclip_only.pth"
done
echo ""
echo "Shared RSNA seed=42 ckpts (read-only):"
echo "  CBAM:   $RSNA_CBAM_CKPT"
echo "  Hybrid: $RSNA_HYBRID_CKPT"
echo ""
echo "Idempotent: already-completed runs will be SKIPPED."
echo ""
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: preview only. Exiting."
    exit 0
fi
echo "Starting in 10 seconds (Ctrl+C to abort)..."
sleep 10
echo "============================================"
echo ""

run_one_seed() {
    local SEED=$1
    echo ""
    echo "##############################################################"
    echo "# SPIDER 8-label sweep — seed=$SEED"
    echo "##############################################################"

    # 1. Baseline SPIDER
    if [[ -f "$CKPT_DIR/best_model_baseline_seed${SEED}.pth" ]]; then
        echo "[1/4] Baseline seed=$SEED already done, skipping"
    else
        echo "[1/4] Baseline SPIDER seed=$SEED ..."
        python3 train_spider.py --model baseline \
            --rsna-checkpoint "$RSNA_CBAM_CKPT" \
            --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_baseline_seed${SEED}.log"
    fi

    # 2. CBAM SPIDER
    if [[ -f "$CKPT_DIR/best_model_cbam_seed${SEED}.pth" ]]; then
        echo "[2/4] CBAM seed=$SEED already done, skipping"
    else
        echo "[2/4] CBAM SPIDER seed=$SEED ..."
        python3 train_spider.py --model cbam \
            --rsna-checkpoint "$RSNA_CBAM_CKPT" \
            --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_cbam_seed${SEED}.log"
    fi

    # 3. BMC-only SPIDER
    if [[ -f "$CKPT_DIR/best_model_hybrid_spider_frozen_seed${SEED}_biomedclip_only.pth" ]]; then
        echo "[3/4] BMC-only seed=$SEED already done, skipping"
    else
        echo "[3/4] BMC-only SPIDER seed=$SEED ..."
        python3 train_spider_hybrid.py \
            --hybrid-checkpoint "$RSNA_HYBRID_CKPT" \
            --cbam-checkpoint "$RSNA_CBAM_CKPT" \
            --ablate-branch biomedclip_only \
            --epochs 15 --batch-size 16 --lr 1e-3 \
            --checkpoint-dir "$CKPT_DIR" \
            --metrics-dir "$METRICS_DIR" \
            --seed "$SEED" \
            2>&1 | tee "$METRICS_DIR/run_bmc_only_seed${SEED}.log"
    fi

    # 4. Hybrid (full) SPIDER
    if [[ -f "$CKPT_DIR/best_model_hybrid_spider_frozen_seed${SEED}.pth" ]]; then
        echo "[4/4] Hybrid seed=$SEED already done, skipping"
    else
        echo "[4/4] Hybrid (full) SPIDER seed=$SEED ..."
        python3 train_spider_hybrid.py \
            --hybrid-checkpoint "$RSNA_HYBRID_CKPT" \
            --cbam-checkpoint "$RSNA_CBAM_CKPT" \
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
echo "All SPIDER 8-label 3-seed runs (shared RSNA) complete"
echo "Finished: $(date)"
echo "=============================================================="
ls -lh "$CKPT_DIR"/best_model_*seed*.pth 2>/dev/null
