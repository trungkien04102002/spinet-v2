#!/usr/bin/env python3
"""
Quick test script for SpineNet
Can be used to verify installation without running Jupyter

Usage:
    python test_spinenet.py                  # Run with original format (11 tasks)
    python test_spinenet.py --format=rsna    # Run with RSNA format (3 tasks)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import os
import argparse
import torch
import torch.nn.functional as F
import spinenet
from spinenet import SpineNet, download_example_scan
from spinenet.io import load_dicoms_from_folder
from spinenet.models.grading_baseline import GradingModelBaseline

# RSNA severity labels
SEVERITY_LABELS = ['Normal/Mild', 'Moderate', 'Severe']

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test SpineNet with different formats')
    parser.add_argument('--format', type=str, default='original',
                        choices=['original', 'rsna'],
                        help='Output format: original (11 tasks) or rsna (3 tasks)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use pretrained SpineNet backbone (for RSNA format with transfer learning)')
    args = parser.parse_args()

    print("=" * 60)
    print(f"SpineNet Quick Test (format={args.format})")
    print("=" * 60)

    if args.format == 'rsna':
        if args.use_pretrained:
            print("\n✓ RSNA mode with TRANSFER LEARNING")
            print("  - 3 tasks: Spinal Canal, Left Foraminal, Right Foraminal")
            print("  - 3 classes per task: Normal/Mild, Moderate, Severe")
            print("  - Backbone: PRETRAINED from SpineNet (transfer learning)")
            print("  - Heads: RANDOM weights (not trained yet)")
            print("  - Predictions will be random, but architecture is ready for training")
        else:
            print("\n⚠ RSNA mode: Using custom baseline model (not SpineNet pretrained)")
            print("  - 3 tasks: Spinal Canal, Left Foraminal, Right Foraminal")
            print("  - 3 classes per task: Normal/Mild, Moderate, Severe")
            print("  - Model has RANDOM weights (for architecture testing only)")
            print("  - Predictions will be meaningless until trained on RSNA data")
    else:
        print("\n✓ Original mode: Using SpineNet pretrained model")
        print("  - 11 tasks: Pfirrmann, Stenosis, Herniation, etc.")

    # Create directories
    os.makedirs('example_scans', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Download example scan
    print("\n[1/5] Downloading example scan...")
    scan_name = 't2_lumbar_scan_2'
    download_example_scan(scan_name, file_path='example_scans')
    print(f"✓ Downloaded {scan_name}")

    # Download weights (for original format, or for RSNA with pretrained)
    if args.format == 'original' or (args.format == 'rsna' and args.use_pretrained):
        print("\n[2/5] Downloading model weights...")
        spinenet.download_weights(verbose=True, force=False)
        print("✓ Weights downloaded")
    else:
        print("\n[2/5] Skipping weight download (RSNA mode uses untrained model)")

    # Check device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\n[3/5] Initializing model on device: {device}")
    if device == 'cpu':
        print("   ⚠ Warning: Running on CPU. This will be slower than GPU.")

    # Initialize SpineNet (for vertebrae detection)
    spnt = SpineNet(device=device, verbose=True)

    # Load scan
    print("\n[4/5] Loading DICOM scan...")
    overwrite_dict = {
        'SliceThickness': [2],
        'ImageOrientationPatient': [0, 1, 0, 0, 0, -1]
    }
    scan = load_dicoms_from_folder(
        f'example_scans/{scan_name}',
        require_extensions=False,
        metadata_overwrites=overwrite_dict
    )
    print(f"✓ Loaded scan with shape: {scan.volume.shape}")
    print(f"  - Pixel spacing: {scan.pixel_spacing} mm")
    print(f"  - Slice thickness: {scan.slice_thickness} mm")

    # Detect vertebrae
    print("\n[5/5] Detecting vertebrae...")
    vert_dicts = spnt.detect_vb(scan.volume, scan.pixel_spacing)
    detected_labels = [v["predicted_label"] for v in vert_dicts]
    print(f"✓ Detected {len(vert_dicts)} vertebrae: {detected_labels}")

    # Grade IVDs based on format
    print("\nGrading intervertebral discs...")
    ivd_dicts = spnt.get_ivds_from_vert_dicts(vert_dicts, scan.volume)

    if args.format == 'original':
        # Use original SpineNet grading model
        ivd_grades = spnt.grade_ivds(ivd_dicts)

        # Display results
        print("\n" + "=" * 60)
        print("GRADING RESULTS (Original Format)")
        print("=" * 60)
        print(ivd_grades)

        # Save results
        output_file = f'results/{scan_name}_test_results.csv'
        ivd_grades.to_csv(output_file)
        print(f"\n✓ Results saved to: {output_file}")

    else:  # args.format == 'rsna'
        # Use RSNA baseline model
        if args.use_pretrained:
            print("\nUsing RSNA baseline model with PRETRAINED backbone...")
            model = GradingModelBaseline(format='rsna')

            # Load pretrained backbone from SpineNet
            try:
                # Get SpineNet weights path
                weights_dir = os.path.expanduser('~/.spinenet/weights')
                if os.path.exists(weights_dir):
                    print(f"Loading pretrained backbone from: {weights_dir}")
                    model.load_pretrained_backbone(weights_dir, verbose=True)
                    print("\n✓ Transfer learning enabled!")
                    print("  - Backbone: PRETRAINED (knows spine anatomy)")
                    print("  - Heads: RANDOM (need training)")
                else:
                    print(f"⚠ SpineNet weights not found at {weights_dir}")
                    print("  Using random weights for now")
            except Exception as e:
                print(f"⚠ Could not load pretrained weights: {e}")
                print("  Using random weights for now")
        else:
            print("\nUsing RSNA baseline model (RANDOM weights)...")
            model = GradingModelBaseline(format='rsna')

        model.to(device)
        model.eval()

        # Prepare IVD volumes
        ivd_volumes = []
        ivd_labels = []
        for idx, ivd in enumerate(ivd_dicts):
            if 'volume' in ivd and ivd['volume'] is not None:
                vol = ivd['volume']
                if vol.shape == (9, 112, 224):
                    ivd_volumes.append(vol)
                    # Get IVD label (try different possible keys)
                    label = (ivd.get('predicted_label') or
                            ivd.get('label') or
                            ivd.get('ivd_label') or
                            ivd.get('ivd_name') or
                            f'IVD_{idx+1}')  # Fallback: IVD_1, IVD_2, etc.
                    ivd_labels.append(label)

        print(f"✓ Prepared {len(ivd_volumes)} IVD volumes")

        if len(ivd_volumes) > 0:
            # Batch predict
            batch = torch.stack([torch.from_numpy(vol).float() for vol in ivd_volumes])
            batch = batch.unsqueeze(1).to(device)  # [N, 1, 9, 112, 224]

            with torch.no_grad():
                outputs = model(batch)

            # Display results
            print("\n" + "=" * 60)
            if args.use_pretrained:
                print("GRADING RESULTS (RSNA Format - PRETRAINED BACKBONE)")
                print("=" * 60)
                print("⚠ Note: Backbone is pretrained, but heads need training")
                print("  Predictions are still random until heads are trained")
            else:
                print("GRADING RESULTS (RSNA Format - RANDOM WEIGHTS)")
                print("=" * 60)
                print("⚠ WARNING: These predictions are MEANINGLESS (untrained model)")
            print("=" * 60)

            for i, ivd_label in enumerate(ivd_labels):
                print(f"\nIVD: {ivd_label}")
                print("-" * 60)

                for task_key, task_name in [
                    ('spinal_canal', 'Spinal Canal Stenosis'),
                    ('left_foraminal', 'Left Neural Foraminal Narrowing'),
                    ('right_foraminal', 'Right Neural Foraminal Narrowing')
                ]:
                    logits = outputs[task_key][i]
                    probs = F.softmax(logits, dim=0)
                    pred_class = torch.argmax(probs).item()
                    pred_label = SEVERITY_LABELS[pred_class]

                    print(f"  {task_name}: {pred_label}")
                    print(f"    Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")

            print("\n" + "=" * 60)
            print("Note: Train model on RSNA 2024 data for real predictions!")
            print("=" * 60)
        else:
            print("❌ No valid IVD volumes found")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
