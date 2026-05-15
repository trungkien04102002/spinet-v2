#!/bin/bash
# =============================================================================
# run_spider_8lbl_resume34.sh — Run ONLY Run 3 (BMC-only) + Run 4 (Hybrid)
# Use this when Runs 1 (baseline) + 2 (cbam) already completed successfully
# from a previous run_spider_8lbl.sh invocation.
#
# Usage on Vast.ai:
#   chmod +x run_spider_8lbl_resume34.sh
#   bash run_spider_8lbl_resume34.sh 2>&1 | tee -a experiments/v3_spider_8lbl/run_all.log
# =============================================================================

set -e

CKPT_DIR=checkpoints/v3_spider_8lbl
METRICS_DIR=experiments/v3_spider_8lbl
RSNA_CBAM_CKPT=checkpoints/v3_20260503/cbam/best_model_attention.pth
RSNA_HYBRID_CKPT=checkpoints/v3_20260503/hybrid/best_model_hybrid.pth

mkdir -p $CKPT_DIR $METRICS_DIR

# Sanity: ensure Run 1+2 already produced their outputs (we should NOT overwrite)
if [[ ! -f $CKPT_DIR/best_model_baseline.pth ]]; then
    echo "WARNING: $CKPT_DIR/best_model_baseline.pth not found."
    echo "         Run 1 (baseline) may not have completed. Continue anyway? (Ctrl+C to abort)"
    sleep 5
fi
if [[ ! -f $CKPT_DIR/best_model_cbam.pth ]]; then
    echo "WARNING: $CKPT_DIR/best_model_cbam.pth not found."
    echo "         Run 2 (cbam) may not have completed. Continue anyway? (Ctrl+C to abort)"
    sleep 5
fi

if [[ ! -f $RSNA_CBAM_CKPT ]]; then
    echo "ERROR: $RSNA_CBAM_CKPT not found."
    exit 1
fi
if [[ ! -f $RSNA_HYBRID_CKPT ]]; then
    echo "ERROR: $RSNA_HYBRID_CKPT not found."
    exit 1
fi

echo "=============================================================="
echo "SPIDER 8-label RESUME from Run 3"
echo "(Runs 1+2 must already be in $CKPT_DIR/)"
echo "Started: $(date)"
echo "=============================================================="

# ---------- 3. BMC-only ----------
echo ""
echo "[3/4] BMC-only 8-label..."
echo "--------------------------------------------------------------"
python3 train_spider_hybrid.py \
    --hybrid-checkpoint $RSNA_HYBRID_CKPT \
    --cbam-checkpoint $RSNA_CBAM_CKPT \
    --ablate-branch biomedclip_only \
    --epochs 15 --batch-size 16 --lr 1e-3 \
    --checkpoint-dir $CKPT_DIR \
    --metrics-dir $METRICS_DIR

# ---------- 4. Hybrid (full) ----------
echo ""
echo "[4/4] Hybrid (full) 8-label..."
echo "--------------------------------------------------------------"
python3 train_spider_hybrid.py \
    --hybrid-checkpoint $RSNA_HYBRID_CKPT \
    --cbam-checkpoint $RSNA_CBAM_CKPT \
    --epochs 15 --batch-size 16 --lr 1e-3 \
    --checkpoint-dir $CKPT_DIR \
    --metrics-dir $METRICS_DIR

echo ""
echo "=============================================================="
echo "Runs 3 + 4 DONE."
echo "Finished: $(date)"
echo "=============================================================="
echo ""
echo "All 4 outputs should be in:"
ls -lh $CKPT_DIR/*.pth 2>/dev/null
echo ""
ls $METRICS_DIR/best_metrics_*.txt 2>/dev/null
