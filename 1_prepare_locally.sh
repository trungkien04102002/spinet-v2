#!/bin/bash
# Run this ONCE on your local machine to create preprocessed .npy files
# Then you can upload the zip to cloud and reuse on all GPU instances

set -e

echo "======================================================================"
echo "Step 1: Prepare RSNA Data Locally (One-Time Setup)"
echo "======================================================================"

# Check if original data exists
if [ -d "rsna-2024-lumbar-spine-degenerative-classification" ]; then
    echo "✓ Found original RSNA dataset"
else
    echo "❌ Original dataset not found!"
    echo ""
    echo "Please download it first:"
    echo "  1. Download from: https://drive.google.com/file/d/19M2C--zpbretFJtYQ6S16FgGt9in3_Qq/view"
    echo "  2. Extract to: rsna-2024-lumbar-spine-degenerative-classification/"
    echo ""
    echo "OR run with download:"
    echo "  python3 prepare_rsna_data.py"
    exit 1
fi

# Preprocess DICOM to .npy
echo ""
echo "Converting DICOM to .npy files..."
echo "This will take 20-40 minutes (one-time only)"
echo ""

python3 prepare_rsna_data.py --skip-download

# Check if preprocessing succeeded
if [ ! -d "rsna_preprocessed" ]; then
    echo "❌ Preprocessing failed!"
    exit 1
fi

# Create zip for upload
echo ""
echo "======================================================================"
echo "Creating zip file for upload..."
echo "======================================================================"

# Remove old zip if exists
rm -f rsna_preprocessed.zip

# Create zip (with progress)
cd rsna_preprocessed
zip -r -q ../rsna_preprocessed.zip . &
PID=$!

# Show progress
echo -n "Zipping"
while kill -0 $PID 2>/dev/null; do
    echo -n "."
    sleep 1
done
echo " Done!"
cd ..

# Check size
echo ""
echo "======================================================================"
echo "✓ Preprocessing Complete!"
echo "======================================================================"
du -sh rsna_preprocessed.zip
echo ""
echo "Next steps:"
echo "  1. Upload rsna_preprocessed.zip to Google Drive or cloud storage"
echo "  2. Get shareable link (e.g., https://drive.google.com/file/d/YOUR_FILE_ID/view)"
echo "  3. On GPU instance, use: ./2_download_preprocessed.sh YOUR_FILE_ID"
echo ""
echo "Files created:"
echo "  - rsna_preprocessed.zip (~8GB) - Upload this to cloud"
echo "======================================================================"
