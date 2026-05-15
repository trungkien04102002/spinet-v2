#!/bin/bash
# =============================================================================
# run_all_3seeds_v3.sh — Master: RSNA 3-seed -> SPIDER 8-label 3-seed
#
# Runs the full mean +/- std sweep end-to-end:
#   Stage 1: RSNA 4 configs x [seed=123, 456]   (~4h)
#   Stage 2: SPIDER 8-label 4 configs x [seed=123, 456]  (~2.5h)
# Total wall-clock: ~6.5h on RTX 4090, ~$3-4 Vast.ai.
#
# seed=42 outputs are PRESERVED throughout (different filenames).
#
# Usage on Vast.ai:
#   tmux new -s allseeds
#   bash scripts/run_all_3seeds_v3.sh 2>&1 | tee experiments/run_all_3seeds.log
#   # Detach: Ctrl+B then D
#
# Resume after disconnect:
#   tmux attach -t allseeds
#
# Idempotent: re-running will skip already-done seeds.
# DRY_RUN=1 to preview without training.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-experiments}"

mkdir -p "$LOG_DIR"

echo "##############################################################"
echo "# MASTER 3-seed sweep: RSNA + SPIDER 8-label"
echo "# Started: $(date)"
echo "##############################################################"
echo ""

# Run RSNA first (SPIDER depends on RSNA seed-matched ckpts)
echo ""
echo "===================== STAGE 1/2: RSNA ========================"
bash "$SCRIPT_DIR/run_rsna_3seeds_v3.sh" \
    2>&1 | tee "$LOG_DIR/run_rsna_3seeds.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "DRY_RUN=1: Stage 1 preview done. Will preview Stage 2 next..."
    echo ""
fi

# Then SPIDER
echo ""
echo "===================== STAGE 2/2: SPIDER ======================"
bash "$SCRIPT_DIR/run_spider_8lbl_3seeds.sh" \
    2>&1 | tee "$LOG_DIR/run_spider_3seeds.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo ""
    echo "DRY_RUN=1: All stages previewed. No training executed."
    exit 0
fi

echo ""
echo "##############################################################"
echo "# MASTER 3-seed sweep COMPLETE"
echo "# Finished: $(date)"
echo "##############################################################"
echo ""

# Final verification
echo "=== RSNA 3-seed outputs ==="
ls -lh checkpoints/v3_20260503/baseline/best_model_seed*.pth 2>/dev/null
ls -lh checkpoints/v3_20260503/cbam/best_model_attention_seed*.pth 2>/dev/null
ls -lh checkpoints/v3_20260503/bmc_only/best_model_*seed*.pth 2>/dev/null
ls -lh checkpoints/v3_20260503/hybrid/best_model_hybrid_seed*.pth 2>/dev/null

echo ""
echo "=== SPIDER 3-seed outputs ==="
ls -lh checkpoints/v3_spider_8lbl/best_model_*seed*.pth 2>/dev/null

echo ""
echo "Next: aggregate via scripts/aggregate_seeds.py (run locally)."
echo "      See scripts/aggregate_seeds.py header for usage."
