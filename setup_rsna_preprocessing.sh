#!/bin/bash
# Setup script for RSNA preprocessing on GPU instance
# Run this on your rented GPU instance to prepare the dataset

set -e  # Exit on error

echo "======================================================================"
echo "RSNA 2024 Dataset Preprocessing Setup"
echo "======================================================================"

# Step 1: Install dependencies
echo ""
echo "[1/4] Installing dependencies..."
pip install gdown tqdm pandas pydicom numpy torch pillow scikit-learn

# Step 2: Download and extract dataset
echo ""
echo "[2/4] Downloading RSNA dataset from Google Drive..."
echo "This will download ~8-10GB and may take 10-30 minutes depending on connection."
python3 prepare_rsna_data.py

# Step 3: Verify preprocessing
echo ""
echo "[3/4] Verifying preprocessed data..."
python3 -c "
import pandas as pd
from pathlib import Path

metadata_path = Path('rsna_preprocessed/train_metadata.csv')
if metadata_path.exists():
    df = pd.read_csv(metadata_path)
    print(f'✓ Found {len(df)} preprocessed samples')
    print(f'✓ Unique patients: {df[\"study_id\"].nunique()}')
    print(f'✓ Distribution:')
    print(f'  - Spinal canal: {df[\"spinal_canal\"].value_counts().to_dict()}')
    print(f'  - Left foraminal: {df[\"left_foraminal\"].value_counts().to_dict()}')
    print(f'  - Right foraminal: {df[\"right_foraminal\"].value_counts().to_dict()}')
else:
    print('❌ Metadata not found!')
"

# Step 4: Calculate size
echo ""
echo "[4/4] Checking disk usage..."
du -sh rsna_preprocessed/

echo ""
echo "======================================================================"
echo "✓ Preprocessing Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Start training: python3 train_rsna_baseline.py --epochs 30 --batch-size 32"
echo "  2. Monitor progress: tail -f checkpoints/training.log"
echo ""
echo "Files created:"
echo "  - rsna_preprocessed/volumes/       # Patient-level .npy files"
echo "  - rsna_preprocessed/train_metadata.csv  # Metadata with labels"
echo "======================================================================"
