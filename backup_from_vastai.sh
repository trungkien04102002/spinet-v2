#!/usr/bin/env bash
#
# Backup important files from a Vast.ai instance to local before stop/destroy.
#
# Usage:
#   ./backup_from_vastai.sh <PORT> <USER@HOST> [SUFFIX]
#
# Example:
#   ./backup_from_vastai.sh 39036 root@220.130.209.122
#   ./backup_from_vastai.sh 39036 root@220.130.209.122 hybrid_run3
#
# If SUFFIX is omitted, a timestamp YYYYMMDD_HHMMSS is appended to every file.
# All files land under:
#   /Users/kienha/spinet-v2/checkpoints/fresh/      (.pth files)
#   /Users/kienha/spinet-v2/experiments/fresh_cbam/ (metrics.txt, log.csv)

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <PORT> <USER@HOST> [SUFFIX]"
    echo "Example: $0 39036 root@220.130.209.122"
    exit 1
fi

PORT="$1"
HOST="$2"
SUFFIX="${3:-$(date +%Y%m%d_%H%M%S)}"

LOCAL_CKPT_DIR="/Users/kienha/spinet-v2/checkpoints/fresh"
LOCAL_EXP_DIR="/Users/kienha/spinet-v2/experiments/fresh_cbam"
REMOTE_BASE="/root/spinet-v2/checkpoints"

mkdir -p "$LOCAL_CKPT_DIR" "$LOCAL_EXP_DIR"

echo "=================================================================="
echo "Backup from Vast.ai → local Mac"
echo "  Host:   $HOST"
echo "  Port:   $PORT"
echo "  Suffix: $SUFFIX"
echo "=================================================================="

# Pairs of (remote_path, local_dir, prefix) — script renames file as prefix_SUFFIX.ext
COPY_PAIRS=(
    # Baseline (vanilla SpineNetV2)
    "$REMOTE_BASE/best_model.pth                  $LOCAL_CKPT_DIR  baseline_pth"
    "$REMOTE_BASE/baseline_best_metrics.txt       $LOCAL_EXP_DIR   baseline_metrics_txt"
    "$REMOTE_BASE/baseline_best_metrics.json      $LOCAL_EXP_DIR   baseline_metrics_json"
    "$REMOTE_BASE/baseline_log.csv                $LOCAL_EXP_DIR   baseline_log_csv"

    # CBAM + class weights
    "$REMOTE_BASE/best_model_attention.pth        $LOCAL_CKPT_DIR  attention_pth"
    "$REMOTE_BASE/attention_best_metrics.txt      $LOCAL_EXP_DIR   attention_metrics_txt"
    "$REMOTE_BASE/attention_best_metrics.json     $LOCAL_EXP_DIR   attention_metrics_json"
    "$REMOTE_BASE/attention_log.csv               $LOCAL_EXP_DIR   attention_log_csv"

    # Hybrid (CBAM + BiomedCLIP)
    "$REMOTE_BASE/hybrid/best_model_hybrid.pth    $LOCAL_CKPT_DIR  hybrid_pth"
    "$REMOTE_BASE/hybrid/hybrid_best_metrics.txt  $LOCAL_EXP_DIR   hybrid_metrics_txt"
    "$REMOTE_BASE/hybrid/hybrid_best_metrics.json $LOCAL_EXP_DIR   hybrid_metrics_json"
    "$REMOTE_BASE/hybrid/hybrid_log.csv           $LOCAL_EXP_DIR   hybrid_log_csv"
)

success=0
skipped=0
failed=0

for pair in "${COPY_PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair
    REMOTE_PATH="$1"
    LOCAL_DIR="$2"
    PREFIX="$3"
    EXT="${REMOTE_PATH##*.}"
    LOCAL_PATH="$LOCAL_DIR/${PREFIX}_${SUFFIX}.${EXT}"

    # Check existence on remote first (fast, fails silently if file not there)
    if ! ssh -p "$PORT" -o ConnectTimeout=5 -o BatchMode=yes "$HOST" "test -f $REMOTE_PATH" 2>/dev/null; then
        echo "  ⊘ skip (not found): $REMOTE_PATH"
        skipped=$((skipped+1))
        continue
    fi

    echo "  ⇩ $REMOTE_PATH → $LOCAL_PATH"
    if scp -P "$PORT" -q "$HOST:$REMOTE_PATH" "$LOCAL_PATH"; then
        success=$((success+1))
    else
        echo "    ✗ failed"
        failed=$((failed+1))
    fi
done

echo "=================================================================="
echo "Done. success=$success skipped=$skipped failed=$failed"
echo "Local locations:"
echo "  $LOCAL_CKPT_DIR"
echo "  $LOCAL_EXP_DIR"
echo "=================================================================="
