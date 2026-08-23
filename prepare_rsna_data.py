#!/usr/bin/env python3
"""
Prepare RSNA 2024 Dataset for Training

This script:
1. Downloads RSNA dataset from Google Drive
2. Extracts the zip file
3. Preprocesses IVD volumes (DICOM → .npy)
4. Creates metadata for fast training

Usage:
    python3 prepare_rsna_data.py
    python3 prepare_rsna_data.py --skip-download  # If data already downloaded
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# Import RSNA dataloader
from rsna_dataloader import RSNASpineDataset


def download_from_gdrive(file_id: str, output_path: str):
    """Download file from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Error: gdown not installed. Installing...")
        os.system(f"{sys.executable} -m pip install gdown")
        import gdown

    # Construct download URL
    url = f'https://drive.google.com/uc?id={file_id}'

    print(f"Downloading from Google Drive...")
    print(f"File ID: {file_id}")
    print(f"Output: {output_path}")

    gdown.download(url, output_path, quiet=False)
    print(f"✓ Download complete: {output_path}")


def extract_zip(zip_path: str, extract_to: str):
    """Extract zip file."""
    print(f"\nExtracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total file count for progress bar
        file_list = zip_ref.namelist()

        # Extract with progress bar
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, extract_to)

    print(f"✓ Extracted to: {extract_to}")


def preprocess_volumes(data_dir: str, output_dir: str, split: str = 'train'):
    """
    Preprocess RSNA data: DICOM → .npy volumes.

    Args:
        data_dir: Path to RSNA dataset (rsna-2024-lumbar-spine-degenerative-classification)
        output_dir: Path to save preprocessed .npy files
        split: 'train' or 'test'
    """
    print(f"\n{'='*70}")
    print(f"Preprocessing {split} data")
    print(f"{'='*70}")

    # Create output directory
    volumes_dir = Path(output_dir) / 'volumes'
    volumes_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataset = RSNASpineDataset(
        data_dir=data_dir,
        split=split,
        levels=None,  # All 5 levels
        cache_images=False  # Don't cache in RAM during preprocessing
    )

    print(f"✓ Found {len(dataset)} samples")

    if len(dataset) == 0:
        print("❌ No samples found! Check your dataset.")
        return

    # Preprocess all samples
    print(f"\nPreprocessing volumes...")

    metadata = []
    failed_samples = []

    for idx in tqdm(range(len(dataset)), desc="Processing"):
        try:
            # Get volume and labels
            volume, labels = dataset[idx]

            # Get sample info
            sample = dataset.samples[idx]

            # Create patient folder (study_id)
            patient_dir = volumes_dir / str(sample['study_id'])
            patient_dir.mkdir(exist_ok=True)

            # Create filename: {series_id}_{level}.npy
            filename = f"{sample['series_id']}_{sample['level']}.npy"
            filepath = patient_dir / filename

            # Save volume as .npy
            np.save(filepath, volume.numpy())

            # Store metadata (with relative path from volumes/)
            relative_path = f"{sample['study_id']}/{filename}"
            metadata.append({
                'filepath': relative_path,
                'study_id': sample['study_id'],
                'series_id': sample['series_id'],
                'level': sample['level'],
                'condition': sample['condition'],
                'spinal_canal': labels['spinal_canal'],
                'left_foraminal': labels['left_foraminal'],
                'right_foraminal': labels['right_foraminal']
            })

        except Exception as e:
            failed_samples.append({'idx': idx, 'error': str(e)})
            print(f"\n⚠ Warning: Failed to process sample {idx}: {e}")

    # Save metadata
    print(f"\nSaving metadata...")
    metadata_df = pd.DataFrame(metadata)
    metadata_path = Path(output_dir) / f'{split}_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Metadata saved: {metadata_path}")

    # Save failed samples log
    if failed_samples:
        failed_path = Path(output_dir) / f'{split}_failed.json'
        with open(failed_path, 'w') as f:
            json.dump(failed_samples, f, indent=2)
        print(f"⚠ {len(failed_samples)} samples failed (see {failed_path})")

    # Summary
    print(f"\n{'='*70}")
    print(f"Preprocessing Complete!")
    print(f"{'='*70}")
    print(f"✓ Successful: {len(metadata)} samples")
    print(f"✗ Failed: {len(failed_samples)} samples")
    print(f"✓ Volumes saved to: {volumes_dir}")
    print(f"✓ Metadata: {metadata_path}")

    # Calculate size
    total_size_mb = sum(f.stat().st_size for f in volumes_dir.glob('*.npy')) / (1024 * 1024)
    print(f"✓ Total size: {total_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Prepare RSNA dataset for training')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip download step (if data already exists)')
    parser.add_argument('--data-dir', type=str, default='rsna-2024-lumbar-spine-degenerative-classification',
                        help='Path to extracted RSNA dataset')
    parser.add_argument('--output-dir', type=str, default='rsna_preprocessed',
                        help='Output directory for preprocessed .npy files')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                        help='Dataset split to preprocess')
    args = parser.parse_args()

    print("="*70)
    print("RSNA 2024 Dataset Preparation")
    print("="*70)

    # Step 1: Download (optional)
    zip_filename = 'rsna-2024-dataset.zip'

    if not args.skip_download:
        print("\n[Step 1/3] Downloading dataset from Google Drive...")

        # Extract file ID from URL
        file_id = '19M2C--zpbretFJtYQ6S16FgGt9in3_Qq'

        # Check if zip already exists
        if os.path.exists(zip_filename):
            response = input(f"{zip_filename} already exists. Re-download? (y/N): ")
            if response.lower() != 'y':
                print("Skipping download...")
            else:
                download_from_gdrive(file_id, zip_filename)
        else:
            download_from_gdrive(file_id, zip_filename)

        # Step 2: Extract
        print("\n[Step 2/3] Extracting dataset...")

        if os.path.exists(args.data_dir):
            response = input(f"{args.data_dir} already exists. Re-extract? (y/N): ")
            if response.lower() != 'y':
                print("Skipping extraction...")
            else:
                extract_zip(zip_filename, '.')
        else:
            extract_zip(zip_filename, '.')
    else:
        print("\n[Step 1-2/3] Skipped (--skip-download)")

    # Step 3: Preprocess
    print("\n[Step 3/3] Preprocessing volumes...")

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory not found: {args.data_dir}")
        print(f"   Make sure the dataset is extracted correctly.")
        return

    # Preprocess
    preprocess_volumes(args.data_dir, args.output_dir, args.split)

    # Final instructions
    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print(f"1. Upload preprocessed data to Google Drive:")
    print(f"   zip -r {args.output_dir}.zip {args.output_dir}/")
    print(f"   Upload {args.output_dir}.zip to Google Drive")
    print(f"")
    print(f"2. On GPU instance, download and extract:")
    print(f"   gdown <your_file_id>")
    print(f"   unzip {args.output_dir}.zip")
    print(f"")
    print(f"3. Start training:")
    print(f"   python3 train_rsna.py --data-dir {args.output_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
