#!/bin/bash
# =============================================================================
# run_rsna_multiseed.sh — RSNA 2024 ablation, multi-seed, runnable in stages.
#
# 4 configs (same mapping that produced the seed-42 paper numbers):
#   baseline -> SpineNetV2        (train_rsna_baseline.py)        [independent]
#   cbam     -> CBAM-only         (train_rsna_attention.py)       [independent;
#                                   also produces the CBAM ckpt hybrid/bmc need]
#   hybrid   -> Hybrid (full)     (train_rsna_hybrid.py --ablate-branch none)
#   bmc      -> BMC-only          (train_rsna_hybrid.py --ablate-branch biomedclip_only)
#
# DEPENDENCY: hybrid & bmc load the CBAM checkpoint of the SAME seed, so 'cbam'
# must finish before 'hybrid'/'bmc' for that seed. The script enforces this.
#
# Each run auto-writes <prefix>_best_metrics.json/.txt + *_log.csv (all 9
# metrics incl. AUC/AUPRC) into its --save-dir, tagged by seed, plus a per-run
# tee log under experiments/v3_rsna_multiseed/.
#
# ---- USAGE ------------------------------------------------------------------
#   chmod +x run_rsna_multiseed.sh
#   export PYTHONPATH=$PYTHONPATH:$(pwd)
#   tmux new -s rsna3
#
#   # run a subset of configs (default = all four):
#   ./run_rsna_multiseed.sh baseline cbam      # e.g. TODAY  (independent + dep source)
#   ./run_rsna_multiseed.sh hybrid bmc         # e.g. TOMORROW (need cbam done)
#   ./run_rsna_multiseed.sh                    # all four
#
#   # choose seeds (default below). 42 is the canonical paper seed: by default
#   # we do NOT re-run it (so existing seed-42 checkpoints are not overwritten).
#   SEEDS="123 456" ./run_rsna_multiseed.sh baseline cbam
#
# Prereqs on the box: rsna_preprocessed/ , ~/.spinenet/weights/ckpt1.pt
# Est. ~1h/run on RTX 4090; 8 runs (2 seeds x 4 configs) ~ half a day, ~$15.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

SEEDS="${SEEDS:-123 456}"            # override: SEEDS="42 123 456" ./run_...
BASE_DIR=checkpoints/v3_20260503
LOG_DIR=experiments/v3_rsna_multiseed

# which configs to run (positional args); default = all four, canonical order
WANT=("$@"); [[ ${#WANT[@]} -eq 0 ]] && WANT=(baseline cbam hybrid bmc)
want() { for c in "${WANT[@]}"; do [[ "$c" == "$1" ]] && return 0; done; return 1; }

mkdir -p "$LOG_DIR" \
         "$BASE_DIR/baseline" "$BASE_DIR/cbam" "$BASE_DIR/hybrid" "$BASE_DIR/bmc_only"

# ---- sanity checks ----------------------------------------------------------
[[ -d rsna_preprocessed ]] || { echo "ERROR: rsna_preprocessed/ missing (./2_download_preprocessed.sh)"; exit 1; }
ls "$HOME/.spinenet/weights"/*.pt >/dev/null 2>&1 || { echo "ERROR: no *.pt backbone in ~/.spinenet/weights/ (./3_download_weights.sh)"; exit 1; }

echo "=============================================================="
echo "RSNA multi-seed  | seeds: $SEEDS  | configs: ${WANT[*]}"
echo "Started: $(date)"
echo "=============================================================="

for S in $SEEDS; do
    echo ""; echo "###################### SEED $S ######################"
    CBAM_CKPT="$BASE_DIR/cbam/best_model_attention_seed${S}.pth"
    [[ "$S" == "42" ]] && CBAM_CKPT="$BASE_DIR/cbam/best_model_attention.pth"

    # ---------- baseline (SpineNetV2) ----------
    if want baseline; then
        echo ""; echo "[seed $S] baseline ..."
        python3 train_rsna_baseline.py \
            --epochs 25 --batch-size 64 --lr 1e-3 \
            --seed "$S" --save-dir "$BASE_DIR/baseline" \
            2>&1 | tee "$LOG_DIR/run_baseline_seed${S}.log"
    fi

    # ---------- cbam (CBAM-only + produces the CBAM ckpt) ----------
    if want cbam; then
        echo ""; echo "[seed $S] cbam ..."
        python3 train_rsna_attention.py \
            --epochs 25 --batch-size 64 --lr 1e-3 \
            --focal-gamma 1.8 --use-focal --use-uncertainty true \
            --class-weight-mode sqrt --augmentation medium --oversample-factor 5 \
            --no-hflip-swap-labels \
            --seed "$S" --save-dir "$BASE_DIR/cbam" \
            2>&1 | tee "$LOG_DIR/run_cbam_seed${S}.log"
    fi

    # ---------- hybrid / bmc need the CBAM ckpt of THIS seed ----------
    if want hybrid || want bmc; then
        if [[ ! -f "$CBAM_CKPT" ]]; then
            echo "ERROR: $CBAM_CKPT not found." >&2
            echo "       hybrid/bmc need the CBAM checkpoint of seed $S." >&2
            echo "       Run 'cbam' for this seed first:  ./run_rsna_multiseed.sh cbam" >&2
            exit 1
        fi
    fi

    if want hybrid; then
        echo ""; echo "[seed $S] hybrid (full) ..."
        python3 train_rsna_hybrid.py \
            --cbam-checkpoint "$CBAM_CKPT" --ablate-branch none \
            --epochs 20 --batch-size 32 --lr 1e-4 \
            --focal-gamma 2.0 --class-weight-mode sqrt --oversample-factor 3 \
            --supcon-weight 0.1 --no-hflip-swap-labels \
            --seed "$S" --save-dir "$BASE_DIR/hybrid" \
            2>&1 | tee "$LOG_DIR/run_hybrid_seed${S}.log"
    fi

    if want bmc; then
        echo ""; echo "[seed $S] bmc_only ..."
        python3 train_rsna_hybrid.py \
            --cbam-checkpoint "$CBAM_CKPT" --ablate-branch biomedclip_only \
            --epochs 20 --batch-size 32 --lr 1e-4 \
            --focal-gamma 2.0 --class-weight-mode sqrt --oversample-factor 3 \
            --supcon-weight 0.1 --no-hflip-swap-labels \
            --seed "$S" --save-dir "$BASE_DIR/bmc_only" \
            2>&1 | tee "$LOG_DIR/run_bmc_only_seed${S}.log"
    fi
    echo "[seed $S] stage done: $(date)"
done

echo ""; echo "=============================================================="
echo "Done: $(date).  Configs run: ${WANT[*]}  | seeds: $SEEDS"
echo "When all 4 configs x all seeds are done, aggregate with:"
echo "  python3 scripts/aggregate_rsna_seeds.py --seeds 42 123 456"
echo "=============================================================="
