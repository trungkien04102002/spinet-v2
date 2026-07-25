#!/bin/bash
# Run this on the GPU instance to download the T1 per-side crops
# (rsna_preprocessed_t1/, ~17GB) needed by F1 #1/#2/#3.
#
# Much faster than regenerating from raw DICOM (prep_t1_crops.py + Kaggle).
#
# Usage:
#   ./6_download_t1.sh YOUR_FILE_ID
#
# To make the zip (on the Mac, from the repo root):
#   zip -r rsna_preprocessed_t1.zip rsna_preprocessed_t1
#   # then upload rsna_preprocessed_t1.zip to Drive, share "Anyone with link",
#   # copy the ID from https://drive.google.com/file/d/YOUR_FILE_ID/view

set -e

echo "======================================================================"
echo "Download T1 crops (rsna_preprocessed_t1/, ~17GB)"
echo "======================================================================"

# The uploaded rsna_preprocessed_t1.zip (seed-42 T1 crops). Override with an
# argument if you re-upload elsewhere: ./6_download_t1.sh <other_id>
DEFAULT_FILE_ID="19dF-TCY1jlk3nc7X-fjM62Lw4IN6p5ed"
FILE_ID="${1:-$DEFAULT_FILE_ID}"

echo ""
echo "[1/3] Installing dependencies..."
pip install -q gdown

echo ""
echo "[2/3] Downloading T1 crops from Google Drive..."
echo "File ID: $FILE_ID   (~17GB)"
echo ""
gdown "https://drive.google.com/uc?id=$FILE_ID" -O rsna_preprocessed_t1.zip

if [ ! -f "rsna_preprocessed_t1.zip" ]; then
    echo "❌ Download failed!"
    echo "  - Check the file ID"
    echo "  - Make sure the file is shared (Anyone with the link can view)"
    exit 1
fi

echo ""
echo "[3/3] Extracting..."
unzip -q rsna_preprocessed_t1.zip

# Fix a doubly-nested folder (rsna_preprocessed_t1/rsna_preprocessed_t1/).
if [ -d "rsna_preprocessed_t1/rsna_preprocessed_t1" ]; then
    echo "Fixing nested folder structure..."
    mv rsna_preprocessed_t1/rsna_preprocessed_t1/* rsna_preprocessed_t1/
    mv rsna_preprocessed_t1/rsna_preprocessed_t1/.* rsna_preprocessed_t1/ 2>/dev/null || true
    rmdir rsna_preprocessed_t1/rsna_preprocessed_t1
fi

if [ -f "rsna_preprocessed_t1/t1_metadata.csv" ]; then
    NUM_CROPS=$(ls rsna_preprocessed_t1/volumes | wc -l)
    echo ""
    echo "======================================================================"
    echo "✓ Download Complete!"
    echo "======================================================================"
    echo "Crops (.npy): $NUM_CROPS   (expected ~19689)"
    du -sh rsna_preprocessed_t1
    echo ""
    echo "Next: python3 experiments/f1_improvement/train_t1_foraminal.py --fast-dev"
    echo "======================================================================"
else
    echo "❌ Extraction failed or incomplete (no t1_metadata.csv)!"
    exit 1
fi

read -p "Remove zip file to save space? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm rsna_preprocessed_t1.zip
    echo "✓ Removed zip file"
fi
