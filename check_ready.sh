#!/bin/bash
# =============================================================================
# check_ready.sh — Verify Vast.ai instance is ready for SPIDER 8-label training
#
# Usage:
#   bash check_ready.sh
# =============================================================================

cd ~/spinet-v2 2>/dev/null || { echo "ERROR: ~/spinet-v2 not found. Clone repo first."; exit 1; }

echo "=================================================="
echo "CHECKLIST: Pre-flight for SPIDER 8-label training"
echo "=================================================="

# 1. Repo + branch + commit
echo ""
echo "[1] Repo + branch + commit 8-label:"
echo "    Branch: $(git branch --show-current)"
echo "    Recent commits:"
git log --oneline -3 | sed 's/^/      /'
if git log --oneline | grep -q "f0e9611"; then
    echo "    OK 8-label commit f0e9611 present"
else
    echo "    FAIL: commit f0e9611 missing - run: git pull origin biomedclip-integration"
fi

# 2. RSNA checkpoints
echo ""
echo "[2] RSNA checkpoints (scp from ~/Desktop/v3_check/):"
if [ -f checkpoints/v3_20260503/cbam/best_model_attention.pth ]; then
    SIZE=$(ls -lh checkpoints/v3_20260503/cbam/best_model_attention.pth | awk '{print $5}')
    echo "    OK cbam best_model_attention.pth ($SIZE)"
else
    echo "    FAIL MISSING: checkpoints/v3_20260503/cbam/best_model_attention.pth"
fi

if [ -f checkpoints/v3_20260503/hybrid/best_model_hybrid.pth ]; then
    SIZE=$(ls -lh checkpoints/v3_20260503/hybrid/best_model_hybrid.pth | awk '{print $5}')
    echo "    OK hybrid best_model_hybrid.pth ($SIZE)"
else
    echo "    FAIL MISSING: checkpoints/v3_20260503/hybrid/best_model_hybrid.pth"
fi

# 3. Base backbone
echo ""
echo "[3] Base 3D ResNet34 backbone (~/.spinenet/weights/ckpt1.pt):"
if [ -f ~/.spinenet/weights/ckpt1.pt ]; then
    SIZE=$(ls -lh ~/.spinenet/weights/ckpt1.pt | awk '{print $5}')
    echo "    OK ckpt1.pt ($SIZE)"
else
    echo "    FAIL MISSING - run: ./3_download_weights.sh 1GCmJ0OuNdw9c1E4giLwyA9EK4uwWL6HT"
fi

# 4. SPIDER dataset
echo ""
echo "[4] SPIDER dataset:"
if [ -d spider ]; then
    NUM_MHA=$(find spider -maxdepth 3 -name "*.mha" 2>/dev/null | wc -l | tr -d ' ')
    echo "    OK spider/ has $NUM_MHA .mha files"
    if [ -f spider/overview.csv ]; then
        LINES=$(wc -l < spider/overview.csv | tr -d ' ')
        echo "    OK overview.csv ($LINES lines)"
    else
        echo "    WARN spider/overview.csv missing - check folder structure"
    fi
else
    echo "    FAIL MISSING - run: ./5_download_spider.sh 1XLQpdw9PfEa2J-UrwWJKFCOiHOsGRSDm"
fi

# 5. Disk space
echo ""
echo "[5] Disk space:"
df -h ~ | tail -1 | awk '{print "    Total: "$2"  Used: "$3"  Free: "$4"  Use%: "$5}'

# 6. GPU + Python + Torch
echo ""
echo "[6] GPU + Python environment:"
python3 -c "
import sys
import torch
print(f'    OK Python {sys.version.split()[0]}')
print(f'    OK Torch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'    OK GPU: {torch.cuda.get_device_name(0)}')
print(f'    OK VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>&1 | sed 's/^/    /'

# 7. Dataloader 8-label smoke test
echo ""
echo "[7] Dataloader 8-label smoke test:"
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -c "
import sys
try:
    from spider_dataloader import SpiderDataset
    d = SpiderDataset(split='train')
    print(f'    OK Samples: {len(d)}')
    vol, labels = d[0]
    print(f'    OK Volume shape: {tuple(vol.shape)}')
    keys = list(labels.keys())
    print(f'    OK Label keys ({len(keys)}): {keys}')
    assert len(keys) == 8, f'EXPECTED 8 keys, got {len(keys)}'
    print('    OK 8-LABEL CONFIRMED')
except Exception as e:
    print(f'    FAIL: {e}')
    sys.exit(1)
" 2>&1

echo ""
echo "=================================================="
echo "If all rows show 'OK' -> READY TO TRAIN"
echo "If any 'FAIL' -> fix before running training"
echo "=================================================="
