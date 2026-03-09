#!/bin/bash
# Run this on GPU instance to download preprocessed .npy files
# Much faster than downloading + converting DICOM every time!
#
# Usage:
#   ./2_download_preprocessed.sh YOUR_FILE_ID
#
# Example:
#   ./2_download_preprocessed.sh 1a2b3c4d5e6f7g8h9i0j

set -e

echo "======================================================================"
echo "Step 2: Download Preprocessed Data (GPU Instance)"
echo "======================================================================"

# Check if file ID provided
if [ -z "$1" ]; then
    echo "❌ Error: No file ID provided"
    echo ""
    echo "Usage:"
    echo "  ./2_download_preprocessed.sh YOUR_FILE_ID"
    echo ""
    echo "Example:"
    echo "  ./2_download_preprocessed.sh 1a2b3c4d5e6f7g8h9i0j"
    echo ""
    echo "To get file ID:"
    echo "  1. Upload rsna_preprocessed.zip to Google Drive"
    echo "  2. Right-click → Get link"
    echo "  3. Copy the ID from: https://drive.google.com/file/d/YOUR_FILE_ID/view"
    exit 1
fi

FILE_ID=$1

# Install gdown if needed
echo ""
echo "[1/3] Installing dependencies..."
pip install -q gdown

# Download preprocessed zip
echo ""
echo "[2/3] Downloading preprocessed data from Google Drive..."
echo "File ID: $FILE_ID"
echo "Size: ~8GB (much smaller than original 30GB!)"
echo ""

gdown "https://drive.google.com/uc?id=$FILE_ID" -O rsna_preprocessed.zip

# Check if download succeeded
if [ ! -f "rsna_preprocessed.zip" ]; then
    echo "❌ Download failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  - Check file ID is correct"
    echo "  - Make sure file is shared (Anyone with the link can view)"
    echo "  - Try manual download: gdown https://drive.google.com/uc?id=$FILE_ID"
    exit 1
fi

# Extract
echo ""
echo "[3/3] Extracting..."
unzip -q rsna_preprocessed.zip -d rsna_preprocessed

# Verify
if [ -f "rsna_preprocessed/train_metadata.csv" ]; then
    NUM_SAMPLES=$(wc -l < rsna_preprocessed/train_metadata.csv)
    echo ""
    echo "======================================================================"
    echo "✓ Download Complete!"
    echo "======================================================================"
    echo "Samples: $NUM_SAMPLES"
    du -sh rsna_preprocessed
    echo ""
    echo "Next steps:"
    echo "  1. Start training:"
    echo "     python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3"
    echo ""
    echo "  2. Monitor GPU:"
    echo "     nvidia-smi -l 1"
    echo ""
    echo "Data ready at: rsna_preprocessed/"
    echo "======================================================================"
else
    echo "❌ Extraction failed or incomplete!"
    exit 1
fi

# Optional: Remove zip to save space
read -p "Remove zip file to save space? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm rsna_preprocessed.zip
    echo "✓ Removed zip file"
fi
