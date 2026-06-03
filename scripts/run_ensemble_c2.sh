#!/usr/bin/env bash
# =============================================================================
# run_ensemble_c2.sh — run the trivial-ensemble control (reviewer concern C2)
# for one seed, auto-building the checkpoint paths so you don't type them.
#
# Compares: SpineNetV2 baseline, BMC-only, their softmax average (Ensemble),
# and the learned-fusion Hybrid — all on the RSNA val split. If Ensemble is
# below Hybrid on F1/Severe-F1, the learned fusion adds value beyond averaging.
#
# Usage (on the GPU box, has CUDA + data):
#   bash scripts/run_ensemble_c2.sh            # seed 123 (default)
#   bash scripts/run_ensemble_c2.sh 456        # seed 456
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

SEED="${1:-123}"
B="checkpoints/v3_20260503"

# filename tag: "" for seed 42, "_seed<S>" otherwise
if [ "$SEED" = "42" ]; then
  BASE="$B/baseline/best_model.pth"
  CBAM="$B/cbam/best_model_attention.pth"
  HYB="$B/hybrid/best_model_hybrid.pth"
  BMC="$B/bmc_only/best_model_hybrid_biomedclip_only.pth"
else
  BASE="$B/baseline/best_model_seed${SEED}.pth"
  CBAM="$B/cbam/best_model_attention_seed${SEED}.pth"
  HYB="$B/hybrid/best_model_hybrid_seed${SEED}.pth"
  BMC="$B/bmc_only/best_model_hybrid_seed${SEED}_biomedclip_only.pth"
fi
OUT="experiments/v3_rsna_multiseed/trivial_ensemble_seed${SEED}.json"

echo "Seed $SEED — using:"
fail=0
for f in "$BASE" "$CBAM" "$HYB" "$BMC"; do
  if [ -s "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; fail=1; fi
done
if [ "$fail" = 1 ]; then
  echo
  echo "Some checkpoints are missing. Check exact names with:"
  echo "  ls $B/baseline/ $B/cbam/ $B/hybrid/ $B/bmc_only/"
  exit 1
fi

echo
echo "Running trivial-ensemble (no training)..."
mkdir -p "$(dirname "$OUT")"
python3 scripts/trivial_ensemble_rsna.py \
  --baseline-ckpt   "$BASE" \
  --bmc-ckpt        "$BMC" \
  --cbam-checkpoint "$CBAM" \
  --hybrid-ckpt     "$HYB" \
  --seed "$SEED" \
  --output "$OUT"

echo
echo "Done. Result JSON: $OUT"
