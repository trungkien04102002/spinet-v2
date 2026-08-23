#!/bin/bash
# Download pretrained SpineNet backbone weights from Google Drive
# Usage: ./3_download_weights.sh YOUR_WEIGHTS_FILE_ID

set -e

echo "======================================================================"
echo "Step 3: Download Pretrained Weights"
echo "======================================================================"

# Check if file ID provided
if [ -z "$1" ]; then
    echo "❌ Error: No file ID provided"
    echo ""
    echo "Usage:"
    echo "  ./3_download_weights.sh YOUR_FILE_ID"
    echo ""
    echo "To get file ID:"
    echo "  1. Upload your weights .pt file to Google Drive"
    echo "  2. Right-click → Get link"
    echo "  3. Copy the ID from: https://drive.google.com/file/d/YOUR_FILE_ID/view"
    echo ""
    echo "If you don't have pretrained weights, you can:"
    echo "  - Train from scratch: python3 train_rsna_baseline.py --unfreeze-backbone"
    echo "  - Or skip this step and train with random backbone"
    exit 1
fi

FILE_ID=$1

# Install gdown if needed
echo ""
echo "[1/3] Installing gdown..."
pip install -q gdown

# Create weights directory
echo ""
echo "[2/3] Creating weights directory..."
mkdir -p ~/.spinenet/weights

# Download weights
echo ""
echo "[3/3] Downloading weights from Google Drive..."
echo "File ID: $FILE_ID"
echo ""

gdown "https://drive.google.com/uc?id=$FILE_ID" -O ~/.spinenet/weights/weights.pt

# Verify
if [ -f ~/.spinenet/weights/weights.pt ]; then
    echo ""
    echo "======================================================================"
    echo "✓ Weights Downloaded!"
    echo "======================================================================"
    ls -lh ~/.spinenet/weights/
    echo ""
    echo "Now you can train with pretrained backbone:"
    echo "  python3 train_rsna_baseline.py --epochs 30 --batch-size 32 --lr 1e-3"
    echo ""
    echo "The backbone will automatically load from ~/.spinenet/weights/"
    echo "======================================================================"
else
    echo "❌ Download failed!"
    exit 1
fi
