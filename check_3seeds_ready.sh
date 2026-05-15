#!/bin/bash
# =============================================================================
# check_3seeds_ready.sh — Pre-flight check for full 3-seed sweep
#
# Verifies that Vast.ai instance has EVERYTHING needed BEFORE starting
# the 6-hour training run.
#
# Usage:
#   bash check_3seeds_ready.sh
# =============================================================================

cd ~/spinet-v2 2>/dev/null || { echo "ERROR: ~/spinet-v2 not found"; exit 1; }

echo "=================================================="
echo "PRE-FLIGHT CHECK: 3-seed RSNA + SPIDER training"
echo "=================================================="

FAIL=0

# 1. Repo + branch + latest commit
echo ""
echo "[1] Repo + latest commit:"
echo "    Branch: $(git branch --show-current)"
LATEST=$(git log --oneline -1)
echo "    HEAD: $LATEST"
if git log --oneline | head -3 | grep -q "3-seed\|3seeds"; then
    echo "    OK 3-seed scripts present"
else
    echo "    WARN: may need git pull"
fi

# 2. RSNA preprocessed data (~16GB) — REQUIRED for RSNA 3-seed
echo ""
echo "[2] RSNA preprocessed data (REQUIRED for RSNA training):"
if [ -d rsna_preprocessed ]; then
    NUM_NPY=$(find rsna_preprocessed -name "*.npy" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh rsna_preprocessed 2>/dev/null | awk '{print $1}')
    if [ "$NUM_NPY" -gt 1000 ]; then
        echo "    OK rsna_preprocessed/ has $NUM_NPY .npy files ($SIZE)"
    else
        echo "    FAIL rsna_preprocessed/ has only $NUM_NPY .npy files (expected >5000)"
        FAIL=$((FAIL+1))
    fi
    if [ -f rsna_preprocessed/train_metadata.csv ]; then
        echo "    OK train_metadata.csv"
    else
        echo "    FAIL train_metadata.csv missing"
        FAIL=$((FAIL+1))
    fi
else
    echo "    FAIL rsna_preprocessed/ MISSING — run:"
    echo "         ./2_download_preprocessed.sh <gdrive_file_id>"
    FAIL=$((FAIL+1))
fi

# 3. SPIDER dataset (REQUIRED for SPIDER stage 2)
echo ""
echo "[3] SPIDER dataset (REQUIRED for SPIDER 3-seed):"
if [ -d spider ]; then
    NUM_MHA=$(find spider -maxdepth 3 -name "*.mha" 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh spider 2>/dev/null | awk '{print $1}')
    if [ "$NUM_MHA" -gt 500 ]; then
        echo "    OK spider/ has $NUM_MHA .mha files ($SIZE)"
    else
        echo "    FAIL spider/ has only $NUM_MHA .mha (expected >700)"
        FAIL=$((FAIL+1))
    fi
else
    echo "    FAIL spider/ MISSING — run ./5_download_spider.sh"
    FAIL=$((FAIL+1))
fi

# 4. Base backbone weights.pt
echo ""
echo "[4] Base backbone weights (~/.spinenet/weights/weights.pt):"
if [ -f ~/.spinenet/weights/weights.pt ]; then
    SIZE=$(ls -lh ~/.spinenet/weights/weights.pt | awk '{print $5}')
    echo "    OK weights.pt ($SIZE)"
else
    echo "    FAIL weights.pt MISSING — run ./3_download_weights.sh"
    FAIL=$((FAIL+1))
fi

# 5. Disk space — RSNA needs ~16GB + temp during training
echo ""
echo "[5] Disk space (CRITICAL for RSNA training):"
DF_OUT=$(df -h ~ | tail -1)
echo "    $DF_OUT"
FREE_GB=$(df -BG ~ | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "$FREE_GB" -lt 5 ]; then
    echo "    FAIL Free disk only ${FREE_GB}GB — training will fail"
    echo "         Need: ~5GB free for new seed checkpoints + logs"
    FAIL=$((FAIL+1))
elif [ "$FREE_GB" -lt 10 ]; then
    echo "    WARN Free disk only ${FREE_GB}GB — tight but workable"
else
    echo "    OK Free disk: ${FREE_GB}GB"
fi

# 6. GPU + Python + Torch
echo ""
echo "[6] GPU + Python:"
python3 -c "
import torch
print(f'    OK Torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'    OK GPU: {torch.cuda.get_device_name(0)}')
print(f'    OK VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>&1 | sed 's/^/    /'

# 7. RSNA preprocessed dataloader smoke test
echo ""
echo "[7] RSNA preprocessed dataloader smoke test:"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -c "
import sys
try:
    from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
    d = RSNAPreprocessedDataset(split='train')
    print(f'    OK Samples: {len(d)}')
    vol, labels = d[0]
    print(f'    OK Volume: {tuple(vol.shape)}, Label keys: {list(labels.keys())}')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)
" 2>&1

# 8. SPIDER dataloader smoke test (8-label)
echo ""
echo "[8] SPIDER 8-label dataloader smoke test:"
python3 -c "
import sys
try:
    from spider_dataloader import SPIDERDataset
    d = SPIDERDataset(split='training')
    print(f'    OK Samples: {len(d)}')
    vol, labels = d[0]
    print(f'    OK 8-label: {len(labels)} keys')
    assert len(labels) == 8
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)
" 2>&1

# 9. Verify all 3-seed scripts present
echo ""
echo "[9] 3-seed scripts:"
for s in scripts/run_rsna_3seeds_v3.sh scripts/run_spider_8lbl_3seeds.sh scripts/run_all_3seeds_v3.sh; do
    if [ -x "$s" ]; then
        echo "    OK $s"
    else
        echo "    FAIL $s (missing or not executable)"
        FAIL=$((FAIL+1))
    fi
done

# Summary
echo ""
echo "=================================================="
if [ "$FAIL" -eq 0 ]; then
    echo "ALL CHECKS PASSED — ready to run 3-seed sweep"
    echo ""
    echo "Next step:"
    echo "  DRY_RUN=1 bash scripts/run_all_3seeds_v3.sh   # preview"
    echo "  bash scripts/run_all_3seeds_v3.sh 2>&1 | tee experiments/run_all_3seeds.log"
else
    echo "FAILURES: $FAIL — fix before running training"
fi
echo "=================================================="
