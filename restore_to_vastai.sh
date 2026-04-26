#!/usr/bin/env bash
#
# Upload checkpoint files from local Mac back to a (new) Vast.ai instance.
# Pairs with backup_from_vastai.sh.
#
# Usage:
#   ./restore_to_vastai.sh <PORT> <USER@HOST> [SUFFIX]
#
# Example:
#   ./restore_to_vastai.sh 41234 root@123.45.67.89 hybrid_fixed_e7
#
# SUFFIX = the same suffix you used in backup_from_vastai.sh.
# If omitted, the script picks the most recent files matching each prefix.
#
# What it uploads (only files that exist locally are uploaded):
#   checkpoints/fresh/baseline_pth_<SUFFIX>.pth   →  /root/spinet-v2/checkpoints/best_model.pth
#   checkpoints/fresh/attention_pth_<SUFFIX>.pth  →  /root/spinet-v2/checkpoints/best_model_attention.pth
#   checkpoints/fresh/hybrid_pth_<SUFFIX>.pth     →  /root/spinet-v2/checkpoints/hybrid/best_model_hybrid.pth
#
# The remote names match what train_*.py / eval_*.py expect by default,
# so no further renaming needed once uploaded.

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <PORT> <USER@HOST> [SUFFIX]"
    echo "Example: $0 41234 root@123.45.67.89 hybrid_fixed_e7"
    exit 1
fi

PORT="$1"
HOST="$2"
SUFFIX="${3:-}"

LOCAL_CKPT_DIR="/Users/kienha/spinet-v2/checkpoints/fresh"
REMOTE_BASE="/root/spinet-v2/checkpoints"

echo "=================================================================="
echo "Restore from local Mac → Vast.ai"
echo "  Host:   $HOST"
echo "  Port:   $PORT"
echo "  Suffix: ${SUFFIX:-<auto: most recent>}"
echo "=================================================================="

# Ensure remote directories exist (mkdir -p is idempotent)
echo "Preparing remote directories..."
ssh -p "$PORT" "$HOST" "mkdir -p $REMOTE_BASE/hybrid"

# Pairs of (local_prefix, remote_path)
PAIRS=(
    "baseline_pth   $REMOTE_BASE/best_model.pth"
    "attention_pth  $REMOTE_BASE/best_model_attention.pth"
    "hybrid_pth     $REMOTE_BASE/hybrid/best_model_hybrid.pth"
)

success=0
skipped=0

for pair in "${PAIRS[@]}"; do
    # shellcheck disable=SC2086
    set -- $pair
    PREFIX="$1"
    REMOTE_PATH="$2"

    if [ -n "$SUFFIX" ]; then
        LOCAL_PATH="$LOCAL_CKPT_DIR/${PREFIX}_${SUFFIX}.pth"
    else
        # Pick most recent matching file
        LOCAL_PATH=$(ls -t "$LOCAL_CKPT_DIR"/${PREFIX}_*.pth 2>/dev/null | head -n1)
    fi

    if [ -z "$LOCAL_PATH" ] || [ ! -f "$LOCAL_PATH" ]; then
        echo "  ⊘ skip: no local file for prefix $PREFIX"
        skipped=$((skipped+1))
        continue
    fi

    SIZE=$(du -h "$LOCAL_PATH" | cut -f1)
    echo "  ⇧ $LOCAL_PATH ($SIZE)"
    echo "    → $HOST:$REMOTE_PATH"
    if scp -P "$PORT" "$LOCAL_PATH" "$HOST:$REMOTE_PATH"; then
        success=$((success+1))
    else
        echo "    ✗ failed"
    fi
done

echo "=================================================================="
echo "Done. uploaded=$success skipped=$skipped"
echo
echo "Next steps on the remote instance:"
echo "  cd /root/spinet-v2"
echo "  git pull   # ensure code is up to date"
echo "  ls -lh checkpoints/{best_model,best_model_attention}.pth checkpoints/hybrid/best_model_hybrid.pth"
echo "=================================================================="
