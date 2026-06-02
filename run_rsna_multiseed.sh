#!/bin/bash
# =============================================================================
# run_rsna_multiseed.sh — RSNA 2024 ablation, 4 configs x extra seeds {123,456}
#
# Seed 42 is ALREADY done (current paper Table 1). This script adds seeds 123
# and 456 so the RSNA table can be reported as mean +/- std over 3 seeds,
# matching the SPIDER protocol (run_spider_8lbl.sh).
#
# The 4 configs (same mapping that produced the seed-42 numbers):
#   1. SpineNetV2 (Base) -> train_rsna_baseline.py
#   2. CBAM-only         -> train_rsna_attention.py        (also the cbam ckpt
#                                                            consumed by 3 & 4)
#   3. Hybrid (full)     -> train_rsna_hybrid.py --ablate-branch none
#   4. BMC-only          -> train_rsna_hybrid.py --ablate-branch biomedclip_only
#
# Each training run auto-writes <prefix>_best_metrics.json/.txt + *_log.csv
# (all 9 metrics incl. AUC/AUPRC) into its --save-dir, tagged by seed.
#
# Usage on Vast.ai:
#   chmod +x run_rsna_multiseed.sh
#   tmux new -s rsna3
#   ./run_rsna_multiseed.sh 2>&1 | tee experiments/v3_rsna_multiseed/run_all.log
#   # detach: Ctrl+B then D ;  reattach: tmux attach -t rsna3
#
# Prerequisites:
#   - export PYTHONPATH=$PYTHONPATH:$(pwd)
#   - rsna_preprocessed/ present (volumes + {train,test}_metadata.csv)
#   - ~/.spinenet/weights/ckpt1.pt present (pretrained 3D ResNet34 backbone)
#
# Est. runtime ~ half a day on one RTX 4090 (8 training runs), cost ~$15.
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

SEEDS="123 456"                      # 42 already done; change here to add more
BASE_DIR=checkpoints/v3_20260503     # same store as the seed-42 run
LOG_DIR=experiments/v3_rsna_multiseed
mkdir -p "$LOG_DIR" \
         "$BASE_DIR/baseline" "$BASE_DIR/cbam" "$BASE_DIR/hybrid" "$BASE_DIR/bmc_only"

# ---- sanity checks -----------------------------------------------------------
if [[ ! -d rsna_preprocessed ]]; then
    echo "ERROR: rsna_preprocessed/ not found. Run ./2_download_preprocessed.sh first." >&2
    exit 1
fi
if [[ ! -f "$HOME/.spinenet/weights/ckpt1.pt" ]]; then
    echo "ERROR: ~/.spinenet/weights/ckpt1.pt not found. Run ./3_download_weights.sh." >&2
    exit 1
fi

echo "=============================================================="
echo "RSNA multi-seed ablation — seeds: $SEEDS"
echo "Started: $(date)"
echo "=============================================================="

for S in $SEEDS; do
    echo ""
    echo "##############################################################"
    echo "# SEED $S"
    echo "##############################################################"

    CBAM_CKPT="$BASE_DIR/cbam/best_model_attention_seed${S}.pth"

    # ---------- 1. SpineNetV2 baseline ----------
    echo ""
    echo "[seed $S | 1/4] SpineNetV2 baseline ..."
    python3 train_rsna_baseline.py \
        --epochs 25 --batch-size 64 --lr 1e-3 \
        --seed "$S" \
        --save-dir "$BASE_DIR/baseline" \
        2>&1 | tee "$LOG_DIR/run_baseline_seed${S}.log"

    # ---------- 2. CBAM-only (also produces the cbam ckpt for 3 & 4) ----------
    echo ""
    echo "[seed $S | 2/4] CBAM-only ..."
    python3 train_rsna_attention.py \
        --epochs 25 --batch-size 64 --lr 1e-3 \
        --focal-gamma 1.8 --use-focal --use-uncertainty true \
        --class-weight-mode sqrt --augmentation medium --oversample-factor 5 \
        --no-hflip-swap-labels \
        --seed "$S" \
        --save-dir "$BASE_DIR/cbam" \
        2>&1 | tee "$LOG_DIR/run_cbam_seed${S}.log"

    if [[ ! -f "$CBAM_CKPT" ]]; then
        echo "ERROR: expected CBAM checkpoint $CBAM_CKPT not found after training." >&2
        echo "       (hybrid + bmc_only need it). Aborting seed $S." >&2
        exit 1
    fi

    # ---------- 3. Hybrid (full) ----------
    echo ""
    echo "[seed $S | 3/4] Hybrid (full) ..."
    python3 train_rsna_hybrid.py \
        --cbam-checkpoint "$CBAM_CKPT" \
        --ablate-branch none \
        --epochs 20 --batch-size 32 --lr 1e-4 \
        --focal-gamma 2.0 --class-weight-mode sqrt --oversample-factor 3 \
        --supcon-weight 0.1 --no-hflip-swap-labels \
        --seed "$S" \
        --save-dir "$BASE_DIR/hybrid" \
        2>&1 | tee "$LOG_DIR/run_hybrid_seed${S}.log"

    # ---------- 4. BMC-only (drop CBAM at forward) ----------
    echo ""
    echo "[seed $S | 4/4] BMC-only ..."
    python3 train_rsna_hybrid.py \
        --cbam-checkpoint "$CBAM_CKPT" \
        --ablate-branch biomedclip_only \
        --epochs 20 --batch-size 32 --lr 1e-4 \
        --focal-gamma 2.0 --class-weight-mode sqrt --oversample-factor 3 \
        --supcon-weight 0.1 --no-hflip-swap-labels \
        --seed "$S" \
        --save-dir "$BASE_DIR/bmc_only" \
        2>&1 | tee "$LOG_DIR/run_bmc_only_seed${S}.log"

    echo "[seed $S] done: $(date)"
done

echo ""
echo "=============================================================="
echo "All seeds done: $(date)"
echo "Aggregate the 3-seed table with:"
echo "  python3 scripts/aggregate_rsna_seeds.py"
echo "=============================================================="
