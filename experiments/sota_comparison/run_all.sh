#!/usr/bin/env bash
#
# One-shot GPU run script for the external SOTA comparison.
#
# Trains BOTH external baseline models (brendanartley, transformer) for
# seed 42 by default (2 runs, sequential, one GPU), then builds the comparison
# table. Override seeds with e.g. SOTA_SEEDS="42 123 456" for a 3-seed sweep.
#
# WALL-CLOCK CAVEAT: both models train FROM SCRATCH (no pretrained weights,
# torchvision resnet18(weights=None)) for 30 epochs each on the full RSNA
# preprocessed dataset. On an RTX-4090 this is expected to take roughly
# 20-40 minutes PER RUN (model/data-loader dependent) -> ~40-80 min for the
# default 2 runs. Single GPU, sequential (no parallelism) for predictable mem.
#
# Usage (run ONCE, from the repo root, on the RTX-4090 box):
#   bash experiments/sota_comparison/run_all.sh                 # seed 42 only
#   SOTA_SEEDS="42 123 456" bash experiments/sota_comparison/run_all.sh
#
set -uo pipefail  # NOTE: no -e — a failed run must not kill the remaining runs

export PYTHONPATH=/Users/kienha/spinet-v2

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${THIS_DIR}/../.." && pwd)"
LOG_DIR="${THIS_DIR}/logs"
mkdir -p "${LOG_DIR}"

MODELS=(brendanartley transformer)
# Default: seed 42 only (per decision to not run full 3-seed sweeps for new work).
# Override: SOTA_SEEDS="42 123 456" bash run_all.sh
read -r -a SEEDS <<< "${SOTA_SEEDS:-42}"

declare -a RESULTS=()  # "model seed status"

cd "${REPO_ROOT}"

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log_file="${LOG_DIR}/${model}_seed${seed}.log"
        echo "============================================================"
        echo "Training model=${model} seed=${seed}  (log: ${log_file})"
        echo "============================================================"

        python experiments/sota_comparison/train_sota.py \
            --model "${model}" --seed "${seed}" \
            --epochs 30 --batch-size 32 --lr 1e-3 \
            2>&1 | tee "${log_file}"
        run_status=${PIPESTATUS[0]}

        if [ "${run_status}" -eq 0 ]; then
            echo "[OK] model=${model} seed=${seed}"
            RESULTS+=("${model} ${seed} OK")
        else
            echo "[FAILED] model=${model} seed=${seed} (exit code ${run_status}) -- continuing with remaining runs"
            RESULTS+=("${model} ${seed} FAILED(exit=${run_status})")
        fi
    done
done

echo
echo "============================================================"
echo "Run summary (6 expected: 2 models x 3 seeds)"
echo "============================================================"
n_ok=0
n_fail=0
for r in "${RESULTS[@]}"; do
    echo "  ${r}"
    case "${r}" in
        *" OK") n_ok=$((n_ok + 1)) ;;
        *) n_fail=$((n_fail + 1)) ;;
    esac
done
echo "------------------------------------------------------------"
echo "Succeeded: ${n_ok} / $(( ${#MODELS[@]} * ${#SEEDS[@]} ))   Failed: ${n_fail}"
echo "============================================================"
echo

echo "Building comparison table..."
python experiments/sota_comparison/make_table.py
table_status=$?

if [ "${table_status}" -eq 0 ]; then
    echo "Comparison table written to experiments/sota_comparison/comparison_table.md (+ .tex)"
else
    echo "[WARN] make_table.py exited with code ${table_status} -- inspect output above."
fi

exit 0
