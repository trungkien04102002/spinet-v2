#!/bin/bash
# Vast.ai: download SPIDER dataset from Google Drive, extract, preprocess, delete zip.
#
# Usage:
#   ./5_download_spider.sh <FILE_ID>
#
# Example:
#   ./5_download_spider.sh 1a2b3c4d5e6f7g8h9i0j
#
# This script:
#   1. Downloads spider.zip from Google Drive (gdown)
#   2. Extracts to spider/
#   3. Verifies expected files exist
#   4. Auto-deletes spider.zip to save disk
#   5. Preprocesses SPIDER → rsna_preprocessed_spider/ (for zero-shot eval)

set -e

echo "======================================================================"
echo "Vast.ai: download + extract SPIDER dataset"
echo "======================================================================"

if [ -z "$1" ]; then
    echo "Error: missing FILE_ID"
    echo
    echo "Usage:"
    echo "  ./5_download_spider.sh <FILE_ID>"
    echo
    echo "How to get FILE_ID:"
    echo "  1. Upload spider.zip to Google Drive (run ./4_zip_spider_local.sh first)"
    echo "  2. Right-click → Get link → 'Anyone with the link'"
    echo "  3. URL: https://drive.google.com/file/d/<FILE_ID>/view"
    echo "         copy the FILE_ID part"
    exit 1
fi

FILE_ID="$1"

if [ -d "spider" ]; then
    echo "Note: spider/ folder already exists. Remove first to avoid mixing? (y/N)"
    read -n 1 ans
    echo
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        rm -rf spider
    fi
fi

# Step 1: ensure gdown installed
echo
echo "[1/4] Ensuring gdown installed..."
pip install -q gdown

# Step 2: download
echo
echo "[2/4] Downloading spider.zip from Google Drive..."
echo "  File ID: $FILE_ID"
echo "  Expected size: ~3 GB"
echo

gdown "https://drive.google.com/uc?id=$FILE_ID" -O spider.zip

if [ ! -f "spider.zip" ]; then
    echo "Download failed."
    echo "Troubleshooting:"
    echo "  - Check FILE_ID is correct"
    echo "  - File must be shared with 'Anyone with the link'"
    echo "  - Manual: gdown https://drive.google.com/uc?id=$FILE_ID"
    exit 1
fi

echo "  Downloaded: $(du -sh spider.zip | awk '{print $1}')"

# Step 3: extract
echo
echo "[3/4] Extracting spider.zip ..."
unzip -q spider.zip

# Verify
echo
echo "[4/4] Verifying contents..."
required=("spider/overview.csv" "spider/radiological_gradings.csv" "spider/images" "spider/masks")
all_ok=1
for path in "${required[@]}"; do
    if [ ! -e "$path" ]; then
        echo "  Missing: $path"
        all_ok=0
    fi
done

if [ "$all_ok" != "1" ]; then
    echo
    echo "Extraction incomplete. Keeping spider.zip for retry."
    exit 1
fi

# Counts
n_img=$(ls spider/images/*.mha 2>/dev/null | wc -l)
n_mask=$(ls spider/masks/*.mha 2>/dev/null | wc -l)
n_grade=$(($(wc -l < spider/radiological_gradings.csv) - 1))
echo "  spider/images/  → $n_img .mha files"
echo "  spider/masks/   → $n_mask .mha files"
echo "  gradings CSV    → $n_grade rows"

# Step 5: cleanup zip
echo
echo "Removing spider.zip to save disk..."
rm -f spider.zip

# Step 6: preprocess for zero-shot eval (crops IVDs → .npy + builds zeroshot CSV)
echo
echo "[5/5] Preprocessing SPIDER for zero-shot eval..."
echo "  This crops 257 patients × ~7 IVDs into (9, 112, 224) .npy volumes."
echo "  Expected runtime: ~10-15 min."
echo
if [ -d "rsna_preprocessed_spider/volumes" ] && [ -f "rsna_preprocessed_spider/spider_zeroshot_test.csv" ]; then
    n_npy=$(ls rsna_preprocessed_spider/volumes/*.npy 2>/dev/null | wc -l)
    echo "  rsna_preprocessed_spider/ already exists ($n_npy .npy files) — skipping."
else
    python3 prepare_spider_zeroshot.py \
        --spider-dir spider \
        --output-dir rsna_preprocessed_spider
fi

echo
echo "======================================================================"
echo "SPIDER ready at:"
echo "  spider/                     (raw, $(du -sh spider 2>/dev/null | awk '{print $1}'))"
echo "  rsna_preprocessed_spider/   (preprocessed, $(du -sh rsna_preprocessed_spider 2>/dev/null | awk '{print $1}'))"
echo "======================================================================"
echo
echo "Next steps:"
echo
echo "  # Zero-shot eval (Hybrid v3 ckpt → 8 SPIDER diseases, no training):"
echo "  bash scripts/run_spider_zeroshot.sh"
echo
echo "  # Or transfer learning (Phase 4):"
echo "  python3 train_spider.py --model baseline \\"
echo "      --rsna-checkpoint checkpoints/best_model.pth \\"
echo "      --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3"
echo
echo "  python3 train_spider.py --model cbam \\"
echo "      --rsna-checkpoint checkpoints/best_model_attention.pth \\"
echo "      --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3"
echo "======================================================================"
