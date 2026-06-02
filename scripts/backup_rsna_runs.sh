#!/usr/bin/env bash
# =============================================================================
# backup_rsna_runs.sh — tar all finished RSNA multi-seed artifacts into ONE file
#
# Run this ON THE GPU BOX before you stop/destroy the instance. It bundles
# every checkpoint (.pth), metrics (.json/.txt/.csv) and log under the
# multi-seed dirs into a single archive you can scp to local in one shot.
#
# Usage (on box):
#   bash scripts/backup_rsna_runs.sh
#
# Then, FROM LOCAL, download it (fill your instance's port + ip):
#   scp -P <PORT> root@<IP>:~/rsna_seed_backup_*.tar.gz ~/spinet-v2/
#   cd ~/spinet-v2 && tar xzf rsna_seed_backup_*.tar.gz   # restores into checkpoints/ + experiments/
#   bash scripts/check_rsna_runs.sh                       # verify locally
# =============================================================================

set -e
cd "$(dirname "$0")/.."

BASE_DIR="checkpoints/v3_20260503"
LOG_DIR="experiments/v3_rsna_multiseed"
OUT="$HOME/rsna_seed_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

# collect only paths that exist (tar errors on missing args otherwise)
ITEMS=()
for p in \
  "$BASE_DIR/baseline" "$BASE_DIR/cbam" "$BASE_DIR/hybrid" "$BASE_DIR/bmc_only" \
  "$LOG_DIR"; do
  [ -e "$p" ] && ITEMS+=("$p")
done

if [ ${#ITEMS[@]} -eq 0 ]; then
  echo "Nothing to back up (no $BASE_DIR or $LOG_DIR). Did any run finish?"
  exit 1
fi

echo "Archiving: ${ITEMS[*]}"
# only keep the lightweight + checkpoint files (skip nothing here; dirs are small)
tar czf "$OUT" "${ITEMS[@]}"
echo
echo "=================================================================="
echo " Backup created: $OUT"
du -h "$OUT" | awk '{print "  size: "$1}'
echo "=================================================================="
echo " Download it FROM LOCAL with:"
echo "   scp -P <PORT> root@<IP>:$OUT ~/spinet-v2/"
echo "   cd ~/spinet-v2 && tar xzf $(basename "$OUT")"
echo "   bash scripts/check_rsna_runs.sh"
echo "=================================================================="
