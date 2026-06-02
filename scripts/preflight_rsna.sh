#!/bin/bash
# =============================================================================
# preflight_rsna.sh — verify the box is ready to run run_rsna_multiseed.sh
#
# Checks data, backbone weights, scripts, CUDA/GPU, python deps, model imports,
# disk space, and (optionally) the seed-42 metric files needed for aggregation.
# Prints a ✓/✗ for each item and a final verdict. Read-only; changes nothing.
#
# Usage:
#   source spinenet-venv/bin/activate
#   bash scripts/preflight_rsna.sh
# =============================================================================

cd "$(dirname "$0")/.."          # repo root
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

FAIL=0
ok(){ echo "  [OK]   $1"; }
no(){ echo "  [MISS] $1"; FAIL=1; }

echo "=============================================="
echo " PREFLIGHT — RSNA multi-seed"
echo " repo: $(pwd)"
echo "=============================================="

echo "-- data & weights --"
[ -d rsna_preprocessed/volumes ] && ok "rsna_preprocessed/volumes/" \
    || no "rsna_preprocessed/volumes/  -> ./2_download_preprocessed.sh <id>"
ls rsna_preprocessed/*metadata*.csv >/dev/null 2>&1 && ok "metadata csv" \
    || no "rsna_preprocessed metadata csv missing"
[ -f "$HOME/.spinenet/weights/ckpt1.pt" ] && ok "~/.spinenet/weights/ckpt1.pt" \
    || no "ckpt1.pt backbone  -> ./3_download_weights.sh <id>"

echo "-- scripts --"
[ -f run_rsna_multiseed.sh ]            && ok "run_rsna_multiseed.sh"        || no "run_rsna_multiseed.sh (git pull)"
[ -f scripts/aggregate_rsna_seeds.py ]  && ok "scripts/aggregate_rsna_seeds.py" || no "aggregate script (git pull)"
[ -f scripts/trivial_ensemble_rsna.py ] && ok "scripts/trivial_ensemble_rsna.py" || no "ensemble script (git pull)"

echo "-- CUDA / GPU --"
python3 - <<'PY'
try:
    import torch
    print(f"  [OK]   torch {torch.__version__}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  [OK]   CUDA available — {p.name}, {p.total_memory/1e9:.0f} GB VRAM")
    else:
        print("  [MISS] CUDA available = False (instance driver too old / wrong torch build)")
        raise SystemExit(3)
except SystemExit:
    raise
except Exception as e:
    print(f"  [MISS] torch import failed: {e}")
    raise SystemExit(3)
PY
[ $? -ne 0 ] && FAIL=1

echo "-- python deps --"
python3 -c "import open_clip, transformers, sklearn, SimpleITK, pandas, numpy" 2>/dev/null \
    && ok "open_clip / transformers / sklearn / SimpleITK / pandas / numpy" \
    || no "missing deps -> pip install -r requirements.txt"

echo "-- model imports (paths the run/ensemble need) --"
python3 -c "from train_rsna_hybrid import build_text_database; from spinenet.models.grading_hybrid import SpineNetHybrid; from spinenet.models.grading_baseline import GradingModelBaseline" 2>/dev/null \
    && ok "train_rsna_hybrid + grading models import" \
    || no "import error (check PYTHONPATH=\$(pwd))"

echo "-- seed-42 metric files (only needed to aggregate the 3-seed table here) --"
B=checkpoints/v3_20260503
for f in baseline/baseline_best_metrics.json cbam/attention_best_metrics.json \
         hybrid/hybrid_best_metrics.json bmc_only/hybrid_biomedclip_only_best_metrics.json; do
    [ -f "$B/$f" ] && echo "  [OK]   $f" \
        || echo "  [warn] $f  (optional: tar xf v3_results_1042.tar $B/$f  — or aggregate on local)"
done

echo "-- disk --"
df -h . | awk 'NR==2{print "  free: "$4" / total: "$2}'

echo "=============================================="
if [ "$FAIL" = 1 ]; then
    echo " RESULT: NOT READY — fix [MISS] lines above."
    exit 1
else
    echo " RESULT: READY — run:  ./run_rsna_multiseed.sh baseline cbam"
    exit 0
fi
