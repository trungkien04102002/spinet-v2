#!/bin/bash
# =============================================================================
# run_rsna_3seeds_v3.sh — Run RSNA 4 configs × 2 additional seeds (123, 456)
#
# Note: seed=42 already exists from v3_20260503 retrain (commit ed705ef).
# This script adds seed=123 and seed=456 to enable mean ± std reporting.
#
# Configs:
#   1. Baseline      (train_rsna_baseline.py)
#   2. CBAM          (train_rsna_attention.py)
#   3. BMC-only      (train_rsna_hybrid.py --ablate-branch biomedclip_only)
#   4. Hybrid (full) (train_rsna_hybrid.py)
#
# Output (per seed):
#   checkpoints/v3_20260503/baseline/best_model_seed{N}.pth + metrics
#   checkpoints/v3_20260503/cbam/best_model_attention_seed{N}.pth + metrics
#   checkpoints/v3_20260503/bmc_only/best_model_hybrid_biomedclip_only_seed{N}.pth
#   checkpoints/v3_20260503/hybrid/best_model_hybrid_seed{N}.pth + metrics
#
# Cost: ~3.5h per seed × 2 seeds = ~7h on RTX 4090, ~$3-4 Vast.ai
#
# Usage on Vast.ai:
#   bash scripts/run_rsna_3seeds_v3.sh 2>&1 | tee experiments/v3_20260503/run_3seeds.log
# =============================================================================

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-checkpoints/v3_20260503}"
METRICS_DIR="${METRICS_DIR:-experiments/v3_20260503}"
SEEDS=(${SEEDS:-123 456})

mkdir -p "$ROOT_DIR/baseline" "$ROOT_DIR/cbam" \
         "$ROOT_DIR/bmc_only" "$ROOT_DIR/hybrid" \
         "$METRICS_DIR"

echo "=============================================================="
echo "RSNA 3-seed runs (adding seeds: ${SEEDS[*]})"
echo "Started: $(date)"
echo "=============================================================="

# ---------- SAFETY PREVIEW ----------
# This script ONLY adds new seed runs. It never overwrites seed=42 files.
# Filenames carry _seed{N} suffix when seed != 42 (logic in trainers).
echo ""
echo "============== SAFETY PREVIEW =============="
echo ""
echo "EXISTING seed=42 outputs (WILL BE PRESERVED):"
ls -1 "$ROOT_DIR"/baseline/best_model*.pth 2>/dev/null | grep -v seed || echo "  (none)"
ls -1 "$ROOT_DIR"/cbam/best_model_attention*.pth 2>/dev/null | grep -v seed || echo "  (none)"
ls -1 "$ROOT_DIR"/bmc_only/best_model*.pth 2>/dev/null | grep -v seed || echo "  (none)"
ls -1 "$ROOT_DIR"/hybrid/best_model_hybrid*.pth 2>/dev/null | grep -v seed || echo "  (none)"
echo ""
echo "NEW files that will be created (one per seed):"
for S in "${SEEDS[@]}"; do
    echo "  $ROOT_DIR/baseline/best_model_seed${S}.pth"
    echo "  $ROOT_DIR/cbam/best_model_attention_seed${S}.pth"
    echo "  $ROOT_DIR/bmc_only/best_model_hybrid_biomedclip_only_seed${S}.pth"
    echo "  $ROOT_DIR/hybrid/best_model_hybrid_seed${S}.pth"
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
    echo ""
    echo "##############################################################"
    echo "# RSNA full sweep — seed=$SEED"
    echo "##############################################################"

    # -------- 1. Baseline RSNA --------
    if [[ -f "$ROOT_DIR/baseline/best_model_seed${SEED}.pth" ]]; then
        echo "[1/4] Baseline seed=$SEED already done, skipping"
    else
        echo "[1/4] Baseline RSNA seed=$SEED ..."
        python3 train_rsna_baseline.py \
            --epochs 20 --batch-size 32 --lr 1e-3 \
            --save-dir "$ROOT_DIR/baseline" \
            --seed "$SEED" \
            2>&1 | tee "$ROOT_DIR/baseline/run_baseline_seed${SEED}.log"
    fi

    # -------- 2. CBAM RSNA --------
    CBAM_CKPT="$ROOT_DIR/cbam/best_model_attention_seed${SEED}.pth"
    if [[ -f "$CBAM_CKPT" ]]; then
        echo "[2/4] CBAM seed=$SEED already done, skipping"
    else
        echo "[2/4] CBAM RSNA seed=$SEED (15 epochs - intentionally undertrained vs v3 seed=42 which used 25 epochs)..."
        python3 train_rsna_attention.py \
            --epochs 15 --batch-size 32 --lr 1e-3 \
            --focal-gamma 1.8 \
            --class-weight-mode sqrt \
            --oversample-factor 5 \
            --no-hflip-swap-labels \
            --save-dir "$ROOT_DIR/cbam" \
            --seed "$SEED" \
            2>&1 | tee "$ROOT_DIR/cbam/run_cbam_seed${SEED}.log"
    fi

    # -------- 3. BMC-only RSNA --------
    BMC_CKPT="$ROOT_DIR/bmc_only/best_model_hybrid_biomedclip_only_seed${SEED}.pth"
    if [[ -f "$BMC_CKPT" ]]; then
        echo "[3/4] BMC-only seed=$SEED already done, skipping"
    else
        echo "[3/4] BMC-only RSNA seed=$SEED ..."
        python3 train_rsna_hybrid.py \
            --cbam-checkpoint "$CBAM_CKPT" \
            --save-dir "$ROOT_DIR/bmc_only" \
            --ablate-branch biomedclip_only \
            --epochs 20 --batch-size 32 --lr 1e-4 \
            --focal-gamma 2.0 \
            --class-weight-mode sqrt \
            --oversample-factor 3 \
            --supcon-weight 0.1 \
            --no-hflip-swap-labels \
            --seed "$SEED" \
            2>&1 | tee "$ROOT_DIR/bmc_only/run_bmc_only_seed${SEED}.log"
    fi

    # -------- 4. Hybrid (full) RSNA --------
    HYBRID_CKPT="$ROOT_DIR/hybrid/best_model_hybrid_seed${SEED}.pth"
    if [[ -f "$HYBRID_CKPT" ]]; then
        echo "[4/4] Hybrid seed=$SEED already done, skipping"
    else
        echo "[4/4] Hybrid (full) RSNA seed=$SEED ..."
        python3 train_rsna_hybrid.py \
            --cbam-checkpoint "$CBAM_CKPT" \
            --save-dir "$ROOT_DIR/hybrid" \
            --epochs 20 --batch-size 32 --lr 1e-4 \
            --focal-gamma 2.0 \
            --class-weight-mode sqrt \
            --oversample-factor 3 \
            --supcon-weight 0.1 \
            --no-hflip-swap-labels \
            --seed "$SEED" \
            2>&1 | tee "$ROOT_DIR/hybrid/run_hybrid_seed${SEED}.log"
    fi

    echo ""
    echo "  --- seed=$SEED done at $(date) ---"
}

for SEED in "${SEEDS[@]}"; do
    run_one_seed "$SEED"
done

echo ""
echo "=============================================================="
echo "All RSNA 3-seed runs complete"
echo "Finished: $(date)"
echo "=============================================================="
echo ""
echo "Next: aggregate via"
echo "  python3 scripts/aggregate_seeds.py rsna \\"
echo "      $ROOT_DIR/baseline/best_metrics_*.json \\"
echo "      $ROOT_DIR/cbam/best_metrics_attention*.json \\"
echo "      $ROOT_DIR/hybrid/best_metrics_hybrid*.json"
