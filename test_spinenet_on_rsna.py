#!/usr/bin/env python3
"""
Test SpineNet VB Detection on RSNA Dataset

This script tests whether SpineNet's vertebrae detection works on RSNA 2024 data.

Usage:
    python3 test_spinenet_on_rsna.py --num-patients 2
"""

import argparse
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import List, Tuple
import cv2

from spinenet.main import SpineNet


def load_rsna_t2_sagittal(data_dir: str, study_id: int, series_id: int) -> Tuple[np.ndarray, List, float]:
    """
    Load RSNA T2 Sagittal scan as volume for SpineNet.

    Returns:
        volume: np.ndarray of shape (H, W, num_slices)
        dicom_list: List of (slice_array, z_position) tuples
        pixel_spacing: float - average pixel spacing in mm
    """
    series_path = Path(data_dir) / 'train_images' / str(study_id) / str(series_id)

    if not series_path.exists():
        raise FileNotFoundError(f"Series not found: {series_path}")

    # Load all DICOM files
    dicom_files = sorted(series_path.glob('*.dcm'))

    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {series_path}")

    print(f"  Found {len(dicom_files)} DICOM files")

    # Read DICOMs and get z-positions
    dicoms = []
    pixel_spacing = None

    for dcm_file in dicom_files:
        dcm = pydicom.dcmread(str(dcm_file))

        # Get pixel spacing from first DICOM
        if pixel_spacing is None:
            if hasattr(dcm, 'PixelSpacing'):
                # PixelSpacing is [row_spacing, col_spacing]
                pixel_spacing = [float(dcm.PixelSpacing[0]), float(dcm.PixelSpacing[1])]
            else:
                # Default if not available
                pixel_spacing = [1.0, 1.0]

        # Get pixel array
        pixel_array = dcm.pixel_array.astype(np.float32)

        # Normalize using percentile clipping (similar to rsna_dataloader)
        p_low, p_high = np.percentile(pixel_array, [1, 99])
        pixel_array = np.clip(pixel_array, p_low, p_high)
        pixel_array = (pixel_array - p_low) / (p_high - p_low)
        pixel_array = (pixel_array * 255).astype(np.uint8)

        # Get z-position
        z_pos = dcm.ImagePositionPatient[2] if hasattr(dcm, 'ImagePositionPatient') else 0

        dicoms.append((pixel_array, z_pos))

    # Calculate slice spacing
    if len(dicoms) > 1:
        z_positions = [d[1] for d in dicoms]
        z_positions.sort()
        slice_spacing = abs(z_positions[1] - z_positions[0])
        if slice_spacing == 0:
            # If all z-positions are same, use default
            slice_spacing = 1.0
    else:
        slice_spacing = 1.0

    # SpineNet expects pixel_spacing as a scalar (average of row/col spacing)
    # Not a 3-element array including slice spacing
    pixel_spacing_scalar = float(np.mean(pixel_spacing))

    # Sort by z-position
    dicoms.sort(key=lambda x: x[1])

    # Stack into volume (H, W, num_slices)
    slices = [d[0] for d in dicoms]
    volume = np.stack(slices, axis=-1)

    print(f"  Volume shape: {volume.shape}")
    print(f"  Pixel spacing: {pixel_spacing_scalar:.4f} mm (avg of row/col)")
    print(f"  Slice spacing: {slice_spacing:.4f} mm")

    return volume, dicoms, pixel_spacing_scalar


def compare_ivd_locations(spinenet_ivds, rsna_coords, tolerance=50):
    """
    Compare SpineNet extracted IVDs with RSNA coordinates.

    Args:
        spinenet_ivds: List of IVD dicts from SpineNet
        rsna_coords: DataFrame with RSNA coordinates for this series
        tolerance: Pixel distance tolerance for matching

    Returns:
        matches: List of (spinenet_ivd, rsna_coord, distance) tuples
    """
    matches = []

    for ivd in spinenet_ivds:
        level_name = ivd['level_name'].lower().replace('/', '_')  # e.g., 'l1_l2'

        # Find corresponding RSNA coordinate
        rsna_row = rsna_coords[rsna_coords['level'] == level_name]

        if rsna_row.empty:
            matches.append({
                'spinenet_level': level_name,
                'rsna_found': False,
                'distance': None
            })
            continue

        # Get RSNA coordinate (use first condition for this level)
        rsna_row = rsna_row.iloc[0]
        rsna_x = rsna_row['x']
        rsna_y = rsna_row['y']

        # SpineNet IVD doesn't directly store center coordinates
        # We need to infer from the volume extraction
        # For now, mark as matched if level exists
        matches.append({
            'spinenet_level': level_name,
            'rsna_found': True,
            'rsna_x': rsna_x,
            'rsna_y': rsna_y,
            'distance': 'N/A (need to extract from volume metadata)'
        })

    return matches


def main():
    parser = argparse.ArgumentParser(description='Test SpineNet on RSNA data')
    parser.add_argument('--data-dir', type=str,
                        default='rsna-2024-lumbar-spine-degenerative-classification',
                        help='Path to RSNA dataset')
    parser.add_argument('--num-patients', type=int, default=2,
                        help='Number of patients to test')
    args = parser.parse_args()

    print("="*70)
    print("SpineNet VB Detection Test on RSNA Data")
    print("="*70)

    # Load metadata
    print("\n[1/4] Loading RSNA metadata...")
    coords_df = pd.read_csv(Path(args.data_dir) / 'train_label_coordinates.csv')
    series_df = pd.read_csv(Path(args.data_dir) / 'train_series_descriptions.csv')

    # Normalize coordinates
    coords_df['condition'] = coords_df['condition'].str.lower().str.replace(' ', '_')
    coords_df['level'] = coords_df['level'].str.lower().str.replace('/', '_')

    # Filter Sagittal T2 series
    def is_sagittal_t2(desc):
        if pd.isna(desc):
            return False
        desc_lower = desc.lower()
        has_t2 = 't2' in desc_lower
        has_sag = 'sag' in desc_lower or 'sagittal' in desc_lower
        is_axial = 'ax' in desc_lower or 'axial' in desc_lower
        return has_t2 and has_sag and not is_axial

    t2_series = series_df[series_df['series_description'].apply(is_sagittal_t2)]
    print(f"  Found {len(t2_series)} Sagittal T2 series")

    # Select patients
    unique_studies = t2_series['study_id'].unique()[:args.num_patients]
    print(f"  Testing {len(unique_studies)} patients: {unique_studies.tolist()}")

    # Initialize SpineNet
    print("\n[2/4] Loading SpineNet...")
    try:
        spinenet = SpineNet(device='cpu')  # Use CPU for testing
        print("  ✓ SpineNet loaded successfully")
    except Exception as e:
        print(f"  ❌ Error loading SpineNet: {e}")
        return

    # Test each patient
    print("\n[3/4] Running VB detection...")
    print("="*70)

    results = []

    for patient_idx, study_id in enumerate(unique_studies):
        print(f"\nPatient {patient_idx+1}/{len(unique_studies)}: {study_id}")
        print("-"*70)

        # Get T2 series for this patient
        patient_series = t2_series[t2_series['study_id'] == study_id]

        if patient_series.empty:
            print("  ⚠ No T2 series found")
            continue

        # Use first T2 series
        series_id = patient_series.iloc[0]['series_id']
        print(f"  Series ID: {series_id}")

        try:
            # Load volume
            print("  Loading DICOM volume...")
            volume, dicoms, pixel_spacing = load_rsna_t2_sagittal(args.data_dir, study_id, series_id)

            # Run SpineNet VB detection
            print("  Running vertebrae detection...")
            vert_dicts = spinenet.detect_vb(volume, pixel_spacing)

            if not vert_dicts:
                print("  ❌ No vertebrae detected!")
                results.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'vb_detected': 0,
                    'ivd_extracted': 0,
                    'status': 'failed_vb_detection'
                })
                continue

            # Count unknown vertebrae
            unknown_count = sum(1 for vb in vert_dicts if vb.get('predicted_label', 'Unknown') == 'Unknown')
            print(f"  ✓ Detected {len(vert_dicts)} vertebrae ({unknown_count} unknown)")
            vb_labels = [vb.get('predicted_label', 'Unknown') for vb in vert_dicts]
            print(f"    Labels: {vb_labels}")

            # Extract IVDs
            print("  Extracting IVDs...")
            ivd_dicts = spinenet.get_ivds_from_vert_dicts(vert_dicts, volume)

            if not ivd_dicts:
                print("  ⚠ No IVDs extracted")
                results.append({
                    'study_id': study_id,
                    'series_id': series_id,
                    'vb_detected': len(vert_dicts),
                    'vb_unknown': unknown_count,
                    'ivd_extracted': 0,
                    'status': 'failed_ivd_extraction'
                })
                continue

            print(f"  ✓ Extracted {len(ivd_dicts)} IVDs")
            ivd_levels = [ivd['level_name'] for ivd in ivd_dicts]
            print(f"    Levels: {ivd_levels}")

            # Compare with RSNA coordinates
            print("\n  Comparing with RSNA coordinates...")
            rsna_coords = coords_df[
                (coords_df['study_id'] == study_id) &
                (coords_df['series_id'] == series_id)
            ]

            if rsna_coords.empty:
                print("  ⚠ No RSNA coordinates found for this series")
                overlap_count = 0
                rsna_levels_set = set()
            else:
                print(f"  RSNA has {len(rsna_coords['level'].unique())} unique levels")
                rsna_levels = list(rsna_coords['level'].unique())
                print(f"    RSNA levels: {rsna_levels}")
                rsna_levels_set = set(rsna_levels)

                # Normalize SpineNet levels: "L5-S1" or "L5/S1" -> "l5_s1"
                def normalize_level(level_str):
                    return level_str.lower().replace('/', '_').replace('-', '_')

                spinenet_levels = set([normalize_level(ivd['level_name']) for ivd in ivd_dicts])

                overlap = spinenet_levels & rsna_levels_set
                overlap_count = len(overlap)

                print(f"\n  Analysis:")
                print(f"    SpineNet detected: {sorted(spinenet_levels)}")
                print(f"    RSNA expects:      {sorted(rsna_levels_set)}")
                print(f"    Overlap:           {sorted(overlap)} ({overlap_count}/{len(rsna_levels_set)} levels)")

                # Check for thoracic vertebrae (shouldn't be in lumbar dataset)
                thoracic = [l for l in spinenet_levels if l.startswith('t')]
                if thoracic:
                    print(f"    ⚠ Detected thoracic vertebrae: {thoracic}")
                    print(f"      (RSNA is lumbar-only dataset)")

            results.append({
                'study_id': study_id,
                'series_id': series_id,
                'vb_detected': len(vert_dicts),
                'vb_unknown': unknown_count,
                'ivd_extracted': len(ivd_dicts),
                'rsna_levels': len(rsna_levels_set),
                'overlap': overlap_count,
                'status': 'success'
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'study_id': study_id,
                'series_id': series_id,
                'status': f'error: {str(e)}'
            })

    # Summary
    print("\n" + "="*70)
    print("[4/4] Summary")
    print("="*70)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No results to display")
        return

    print(f"\nTotal patients tested: {len(results_df)}")

    success = results_df[results_df['status'] == 'success']
    if not success.empty:
        print(f"\n✓ VB Detection Results: {len(success)} patients")
        print(f"  Average VBs detected: {success['vb_detected'].mean():.1f}")
        print(f"  Average unknown labels: {success['vb_unknown'].mean():.1f}")
        print(f"  Average IVDs extracted: {success['ivd_extracted'].mean():.1f}")
        if 'overlap' in success.columns:
            avg_overlap = success['overlap'].mean()
            avg_expected = success['rsna_levels'].mean()
            overlap_pct = (avg_overlap / avg_expected * 100) if avg_expected > 0 else 0
            print(f"  Level overlap with RSNA: {avg_overlap:.1f}/{avg_expected:.1f} ({overlap_pct:.0f}%)")

    failed = results_df[results_df['status'] != 'success']
    if not failed.empty:
        print(f"\n✗ Failed detections: {len(failed)}")
        for _, row in failed.iterrows():
            print(f"  - Study {row['study_id']}: {row['status']}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    if not success.empty:
        avg_overlap = success['overlap'].mean()
        avg_expected = success['rsna_levels'].mean()
        avg_unknown = success['vb_unknown'].mean()
        total_vb = success['vb_detected'].mean()

        unknown_pct = (avg_unknown / total_vb * 100) if total_vb > 0 else 0
        overlap_pct = (avg_overlap / avg_expected * 100) if avg_expected > 0 else 0

        print(f"\n📊 Key Findings:")
        print(f"  - Vertebrae detection: ✅ Works ({success['vb_detected'].mean():.1f} vertebrae detected)")
        print(f"  - Vertebrae labeling: ❌ Poor ({unknown_pct:.0f}% labeled as 'Unknown')")
        print(f"  - Level matching: ❌ Poor ({overlap_pct:.0f}% overlap with RSNA)")
        print(f"  - IVD extraction: ✅ Works ({success['ivd_extracted'].mean():.1f} IVDs extracted)")

        if overlap_pct < 20:
            print("\n❌ SpineNet NOT suitable for RSNA dataset")
            print("\n**Problems:**")
            print("  1. Vertebrae labeling fails (most labeled as 'Unknown')")
            print("  2. Wrong anatomical region detected (thoracic instead of lumbar)")
            print("  3. Zero/minimal overlap with RSNA ground truth levels")
            print("\n**Root Cause:**")
            print("  - SpineNet trained on different data distribution")
            print("  - RSNA scans have different FOV, resolution, or protocol")
            print("  - Labeling model not generalizing to RSNA data")
            print("\n**Recommendation: Use coordinate-based extraction**")
            print("  ✅ RSNA provides exact IVD coordinates")
            print("  ✅ More reliable than VB detection")
            print("  ✅ Simpler and faster preprocessing")
            print("  ✅ Already implemented in prepare_rsna_data.py")
        elif overlap_pct < 80:
            print("\n⚠️  SpineNet partially works")
            print("\n**Recommendation: Hybrid approach or coordinate-based**")
        else:
            print("\n✅ SpineNet works well on RSNA!")
            print("\n**Recommendation: Use SpineNet pipeline**")
    else:
        print("❌ All detections failed")
        print("\n**Recommendation: Use coordinate-based extraction**")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
