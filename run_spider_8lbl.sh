#!/bin/bash
# =============================================================================
# run_spider_8lbl.sh — SPIDER 8-label retrain × 4 configs (seed=42 default)
#
# Usage on Vast.ai:
#   chmod +x run_spider_8lbl.sh
#   tmux new -s spider8
#   ./run_spider_8lbl.sh 2>&1 | tee experiments/v3_spider_8lbl/run_all.log
#   # Detach: Ctrl+B then D
#
# Prerequisites:
#   - git pull origin biomedclip-integration (code 8-label)
#   - checkpoints/v3_20260503/cbam/best_model_attention.pth exists
#   - checkpoints/v3_20260503/hybrid/best_model_hybrid.pth exists
#
# Output: checkpoints/v3_spider_8lbl/ + experiments/v3_spider_8lbl/
#   (won't overwrite existing 4-label results in v3_spider/)
#
# Total runtime: ~2-3h on RTX 4090, cost ~$5 on Vast.ai
# =============================================================================

set -e   # exit on error

CKPT_DIR=checkpoints/v3_spider_8lbl
METRICS_DIR=experiments/v3_spider_8lbl
RSNA_CBAM_CKPT=checkpoints/v3_20260503/cbam/best_model_attention.pth
RSNA_HYBRID_CKPT=checkpoints/v3_20260503/hybrid/best_model_hybrid.pth

mkdir -p $CKPT_DIR $METRICS_DIR

# Sanity check
if [[ ! -f $RSNA_CBAM_CKPT ]]; then
    echo "ERROR: $RSNA_CBAM_CKPT not found. Copy from Desktop/v3_check first."
    exit 1
fi
if [[ ! -f $RSNA_HYBRID_CKPT ]]; then
    echo "ERROR: $RSNA_HYBRID_CKPT not found. Copy from Desktop/v3_check first."
    exit 1
fi

echo "=============================================================="
echo "SPIDER 8-label retrain — 4 configs (seed=42, single run)"
echo "Started: $(date)"
echo "=============================================================="

# ---------- 1. SpineNetV2 baseline ----------
echo ""
echo "[1/4] SpineNetV2 baseline 8-label..."
echo "--------------------------------------------------------------"
python3 train_spider.py --model baseline \
    --rsna-checkpoint $RSNA_CBAM_CKPT \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
    --checkpoint-dir $CKPT_DIR \
    --metrics-dir $METRICS_DIR

# ---------- 2. CBAM ----------
echo ""
echo "[2/4] CBAM 8-label..."
echo "--------------------------------------------------------------"
python3 train_spider.py --model cbam \
    --rsna-checkpoint $RSNA_CBAM_CKPT \
    --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
    --checkpoint-dir $CKPT_DIR \
    --metrics-dir $METRICS_DIR

# ---------- 3. BMC-only ----------
echo ""
echo "[3/4] BMC-only 8-label..."
echo "--------------------------------------------------------------"
python3 train_spider_hybrid.py \
    --hybrid-checkpoint $RSNA_HYBRID_CKPT \
    --cbam-checkpoint $RSNA_CBAM_CKPT \
    --ablate-branch biomedclip_only --freeze-backbone \
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
    --freeze-backbone \
    --epochs 15 --batch-size 16 --lr 1e-3 \
    --checkpoint-dir $CKPT_DIR \
    --metrics-dir $METRICS_DIR

echo ""
echo "=============================================================="
echo "ALL 4 CONFIGS DONE."
echo "Finished: $(date)"
echo "=============================================================="
echo ""
echo "Results saved to:"
echo "  - Checkpoints: $CKPT_DIR/"
echo "  - Metrics:     $METRICS_DIR/"
echo ""
echo "Quick summary:"
ls -lh $CKPT_DIR/*.pth 2>/dev/null
echo ""
echo "Best metrics files:"
ls $METRICS_DIR/best_metrics_*.txt 2>/dev/null
echo ""
echo "Pull to local with:"
echo "  scp -P <port> -r root@<vast_ip>:~/spinet-v2/$CKPT_DIR ./checkpoints/"
echo "  scp -P <port> -r root@<vast_ip>:~/spinet-v2/$METRICS_DIR ./experiments/"
