#!/usr/bin/env bash
# =============================================================================
# check_rsna_runs.sh — status of RSNA multi-seed runs (checkpoints + metrics + logs)
#
# Works on the GPU box (to see what finished) AND on local (to verify a copy).
# Read-only. Prints a DONE/-- matrix per (config, seed) + the log files.
#
# Usage:
#   bash scripts/check_rsna_runs.sh                 # seeds 42 123 456
#   bash scripts/check_rsna_runs.sh 123 456         # only these seeds
#   BASE_DIR=checkpoints/v3_20260503 bash scripts/check_rsna_runs.sh
# =============================================================================

cd "$(dirname "$0")/.."
BASE_DIR="${BASE_DIR:-checkpoints/v3_20260503}"
LOG_DIR="${LOG_DIR:-experiments/v3_rsna_multiseed}"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(42 123 456)

# tag for a seed: "" if 42 else "_seed<S>"
tag(){ [ "$1" = "42" ] && echo "" || echo "_seed$1"; }

# echo "<pth_path>|<json_path>" for a (config, seed)
paths(){
  local cfg="$1" s="$2" t; t="$(tag "$s")"
  case "$cfg" in
    baseline) echo "$BASE_DIR/baseline/best_model${t}.pth|$BASE_DIR/baseline/baseline${t}_best_metrics.json";;
    cbam)     echo "$BASE_DIR/cbam/best_model_attention${t}.pth|$BASE_DIR/cbam/attention${t}_best_metrics.json";;
    hybrid)   echo "$BASE_DIR/hybrid/best_model_hybrid${t}.pth|$BASE_DIR/hybrid/hybrid${t}_best_metrics.json";;
    bmc)      local b; [ "$s" = "42" ] && b="_biomedclip_only" || b="_seed${s}_biomedclip_only"
              echo "$BASE_DIR/bmc_only/best_model_hybrid${b}.pth|$BASE_DIR/bmc_only/hybrid${b}_best_metrics.json";;
  esac
}

echo "=================================================================="
echo " RSNA run status   base: $BASE_DIR   seeds: ${SEEDS[*]}"
echo "=================================================================="
printf "%-9s %-6s %-10s %-8s\n" "config" "seed" "ckpt.pth" "metrics"
echo "------------------------------------------------------------------"
done=0; total=0
for s in "${SEEDS[@]}"; do
  for cfg in baseline cbam hybrid bmc; do
    total=$((total+1))
    IFS='|' read -r pth json <<< "$(paths "$cfg" "$s")"
    if [ -s "$pth" ]; then sz=$(du -h "$pth" 2>/dev/null | cut -f1); pstat="$sz"; else pstat="--"; fi
    if [ -s "$json" ]; then jstat="OK"; else jstat="--"; fi
    [ "$pstat" != "--" ] && [ "$jstat" = "OK" ] && { mark="DONE"; done=$((done+1)); } || mark=""
    printf "%-9s %-6s %-10s %-8s %s\n" "$cfg" "$s" "$pstat" "$jstat" "$mark"
  done
  echo "------------------------------------------------------------------"
done
echo "Completed (ckpt+metrics): $done / $total"
echo
echo "-- logs in $LOG_DIR --"
ls -lh "$LOG_DIR"/*.log 2>/dev/null || echo "  (no logs found)"
