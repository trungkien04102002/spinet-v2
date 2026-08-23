#!/usr/bin/env python3
"""
Quick visualization of preprocessed .npy IVD volumes.

Usage:
    python3 visualize_npy.py                      # Show 3x3 grid of 9 slices
    python3 visualize_npy.py --patient 4003253    # Specific patient
    python3 visualize_npy.py --show-stats         # Statistics only
    python3 visualize_npy.py --check-crop         # Verify extraction is correct
    python3 visualize_npy.py --3d                 # 3D orthogonal views
    python3 visualize_npy.py --3d-interactive     # Interactive 3D (scroll to change slices)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import pydicom
import matplotlib.patches as patches


def visualize_volume(volume, title="IVD Volume"):
    """Display all 9 slices of IVD volume."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(9):
        ax = axes[i // 3, i % 3]
        ax.imshow(volume[i], cmap='gray')
        ax.set_title(f'Slice {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_volume_3d(volume, title="IVD Volume 3D"):
    """
    Display volume in 3D using orthogonal slice views.

    Shows:
    - Sagittal view (along depth axis)
    - Axial view (along height axis)
    - Coronal view (along width axis)
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Volume shape: (9, 112, 224) = (depth, height, width)
    depth, height, width = volume.shape

    # 1. Sagittal view (original slices - along depth)
    ax1 = fig.add_subplot(131)
    center_slice_depth = depth // 2
    ax1.imshow(volume[center_slice_depth], cmap='gray', aspect='auto')
    ax1.set_title(f'Sagittal (Depth slice {center_slice_depth+1}/{depth})\n[Height × Width]')
    ax1.set_xlabel('Width (224 pixels)')
    ax1.set_ylabel('Height (112 pixels)')

    # 2. Axial view (along height - horizontal slice through spine)
    ax2 = fig.add_subplot(132)
    center_slice_height = height // 2
    axial_slice = volume[:, center_slice_height, :]
    ax2.imshow(axial_slice, cmap='gray', aspect='auto')
    ax2.set_title(f'Axial (Height slice {center_slice_height+1}/{height})\n[Depth × Width]')
    ax2.set_xlabel('Width (224 pixels)')
    ax2.set_ylabel('Depth (9 slices)')

    # 3. Coronal view (along width - front to back)
    ax3 = fig.add_subplot(133)
    center_slice_width = width // 2
    coronal_slice = volume[:, :, center_slice_width]
    ax3.imshow(coronal_slice, cmap='gray', aspect='auto')
    ax3.set_title(f'Coronal (Width slice {center_slice_width+1}/{width})\n[Depth × Height]')
    ax3.set_xlabel('Height (112 pixels)')
    ax3.set_ylabel('Depth (9 slices)')

    plt.tight_layout()
    plt.show()

    print(f"\n  3D Volume Info:")
    print(f"    Shape: {volume.shape} (depth, height, width)")
    print(f"    Depth: {depth} slices (sagittal MRI slices)")
    print(f"    Height: {height} pixels (superior-inferior)")
    print(f"    Width: {width} pixels (left-right)")


def visualize_volume_3d_interactive(volume, title="IVD Volume 3D Interactive"):
    """
    Interactive 3D visualization with scrollable slices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title + '\n(Click and drag to change slices)', fontsize=16)

    depth, height, width = volume.shape

    # Initial slices
    slices = [depth // 2, height // 2, width // 2]

    def update_views():
        # Sagittal
        axes[0].clear()
        axes[0].imshow(volume[slices[0]], cmap='gray', aspect='auto')
        axes[0].set_title(f'Sagittal (Slice {slices[0]+1}/{depth})')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')

        # Axial
        axes[1].clear()
        axes[1].imshow(volume[:, slices[1], :], cmap='gray', aspect='auto')
        axes[1].set_title(f'Axial (Slice {slices[1]+1}/{height})')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Depth')

        # Coronal
        axes[2].clear()
        axes[2].imshow(volume[:, :, slices[2]], cmap='gray', aspect='auto')
        axes[2].set_title(f'Coronal (Slice {slices[2]+1}/{width})')
        axes[2].set_xlabel('Height')
        axes[2].set_ylabel('Depth')

        fig.canvas.draw()

    def on_scroll(event):
        # Determine scroll direction
        direction = 1 if event.button == 'up' else -1

        if event.inaxes == axes[0]:
            slices[0] = max(0, min(depth - 1, slices[0] + direction))
        elif event.inaxes == axes[1]:
            slices[1] = max(0, min(height - 1, slices[1] + direction))
        elif event.inaxes == axes[2]:
            slices[2] = max(0, min(width - 1, slices[2] + direction))
        update_views()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    update_views()
    plt.show()


def visualize_crop_verification(volume, row, data_dir):
    """
    Verify crop extraction by showing:
    1. Original DICOM with crop location marked
    2. Center slice from extracted volume
    """
    # Load coordinate info
    coords_df = pd.read_csv(Path(data_dir) / 'train_label_coordinates.csv')

    # Normalize condition and level names
    coords_df['condition'] = coords_df['condition'].str.lower().str.replace(' ', '_')
    coords_df['level'] = coords_df['level'].str.lower().str.replace('/', '_')

    # Find matching coordinate
    coord = coords_df[
        (coords_df['study_id'] == row['study_id']) &
        (coords_df['series_id'] == row['series_id']) &
        (coords_df['level'] == row['level']) &
        (coords_df['condition'] == row['condition'])
    ]

    if coord.empty:
        print(f"⚠️  No coordinate found for this sample")
        return

    coord = coord.iloc[0]
    x, y, instance_num = coord['x'], coord['y'], coord['instance_number']

    # Load original DICOM
    series_path = Path(data_dir) / 'train_images' / str(row['study_id']) / str(row['series_id'])
    dicom_files = sorted(series_path.glob('*.dcm'))

    if not dicom_files:
        print(f"⚠️  No DICOM files found")
        return

    # Load all DICOMs and sort by z-position
    dicoms = []
    for dcm_file in dicom_files:
        dcm = pydicom.dcmread(str(dcm_file))
        pixel_array = dcm.pixel_array.astype(np.float32)
        z_pos = dcm.ImagePositionPatient[2] if hasattr(dcm, 'ImagePositionPatient') else 0
        dicoms.append((pixel_array, z_pos, dcm.InstanceNumber if hasattr(dcm, 'InstanceNumber') else 0))

    dicoms.sort(key=lambda x: x[1])  # Sort by z-position

    # Find the center slice (should be at instance_number)
    center_idx = None
    for i, (_, _, inst_num) in enumerate(dicoms):
        if inst_num == instance_num:
            center_idx = i
            break

    if center_idx is None:
        print(f"⚠️  Instance number {instance_num} not found")
        center_idx = len(dicoms) // 2  # Fallback to middle

    center_slice = dicoms[center_idx][0]

    # Normalize center slice same way as preprocessing
    p_low, p_high = np.percentile(center_slice, [1, 99])
    center_slice = np.clip(center_slice, p_low, p_high)
    center_slice = (center_slice - p_low) / (p_high - p_low)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Crop Verification - Patient {row["study_id"]} - {row["level"]}', fontsize=16)

    # 1. Original DICOM with crop location
    ax = axes[0]
    ax.imshow(center_slice, cmap='gray')
    ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3)

    # Draw crop region (240x120 before resize to 224x112)
    crop_w, crop_h = 240, 120
    rect = patches.Rectangle(
        (x - crop_w//2, y - crop_h//2),
        crop_w, crop_h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(f'Original DICOM (slice {center_idx})\nRed + : IVD coordinate ({x:.0f}, {y:.0f})\nRed box: Crop region')
    ax.axis('on')

    # 2. Zoomed crop region from DICOM
    ax = axes[1]
    x_min = max(0, int(x - crop_w//2))
    x_max = min(center_slice.shape[1], int(x + crop_w//2))
    y_min = max(0, int(y - crop_h//2))
    y_max = min(center_slice.shape[0], int(y + crop_h//2))

    cropped = center_slice[y_min:y_max, x_min:x_max]
    ax.imshow(cropped, cmap='gray')
    ax.set_title(f'Cropped from DICOM\n{cropped.shape}')
    ax.axis('off')

    # 3. Center slice from .npy
    ax = axes[2]
    center_npy = volume[4]  # Middle slice (0-indexed, so 4 is center of 9 slices)
    ax.imshow(center_npy, cmap='gray')
    ax.set_title(f'Extracted .npy (center slice)\n{center_npy.shape}')
    ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Print comparison
    print(f"\n  Crop Verification:")
    print(f"    Original DICOM shape: {center_slice.shape}")
    print(f"    Coordinate: ({x:.0f}, {y:.0f})")
    print(f"    Crop region: {crop_w}x{crop_h} around coordinate")
    print(f"    Extracted .npy: {volume.shape}")
    print(f"    ✅ Shapes match: crop -> resize({crop_w}x{crop_h} -> 224x112) -> stack(9 slices)")


def check_volume_stats(volume, filepath):
    """Print statistics about the volume."""
    print(f"\nVolume: {filepath}")
    print(f"  Shape: {volume.shape}")
    print(f"  Dtype: {volume.dtype}")
    print(f"  Min: {volume.min():.4f}")
    print(f"  Max: {volume.max():.4f}")
    print(f"  Mean: {volume.mean():.4f}")
    print(f"  Std: {volume.std():.4f}")

    # Check for potential issues
    issues = []

    if volume.shape != (9, 112, 224):
        issues.append(f"❌ Wrong shape! Expected (9, 112, 224), got {volume.shape}")
    else:
        issues.append("✅ Correct shape (9, 112, 224)")

    if volume.min() < 0:
        issues.append("⚠️  Negative values found")
    else:
        issues.append("✅ No negative values")

    if volume.max() > 1.5:
        issues.append("⚠️  Values > 1.5 (expected normalized to [0, 1])")
    else:
        issues.append("✅ Values in expected range")

    if np.isnan(volume).any():
        issues.append("❌ NaN values found!")
    else:
        issues.append("✅ No NaN values")

    if np.isinf(volume).any():
        issues.append("❌ Inf values found!")
    else:
        issues.append("✅ No Inf values")

    # Check if all slices are similar (might indicate a problem)
    slice_means = [volume[i].mean() for i in range(9)]
    if max(slice_means) - min(slice_means) < 0.01:
        issues.append("⚠️  All slices very similar (might be an issue)")
    else:
        issues.append("✅ Slices have variation")

    print("\n  Quality Checks:")
    for issue in issues:
        print(f"    {issue}")

    return len([i for i in issues if i.startswith('❌')]) == 0


def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessed .npy volumes')
    parser.add_argument('--data-dir', type=str, default='rsna_preprocessed',
                        help='Path to preprocessed data directory')
    parser.add_argument('--rsna-dir', type=str,
                        default='rsna-2024-lumbar-spine-degenerative-classification',
                        help='Path to original RSNA dataset (for crop verification)')
    parser.add_argument('--patient', type=int, default=None,
                        help='Specific patient ID to visualize')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of random samples to check')
    parser.add_argument('--show-stats', action='store_true',
                        help='Show statistics without visualization')
    parser.add_argument('--check-crop', action='store_true',
                        help='Verify crop extraction against original DICOM')
    parser.add_argument('--3d', action='store_true', dest='view_3d',
                        help='Show 3D orthogonal views (sagittal, axial, coronal)')
    parser.add_argument('--3d-interactive', action='store_true', dest='view_3d_interactive',
                        help='Show interactive 3D views with scrollable slices')
    args = parser.parse_args()

    print("="*70)
    print("NPY Volume Checker")
    print("="*70)

    # Check if preprocessed data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n❌ Error: {data_dir} not found")
        print("\nRun preprocessing first:")
        print("  python3 prepare_rsna_data.py --skip-download")
        return

    # Load metadata
    metadata_path = data_dir / 'train_metadata.csv'
    if not metadata_path.exists():
        print(f"\n❌ Error: {metadata_path} not found")
        return

    metadata = pd.read_csv(metadata_path)
    print(f"\n✓ Found {len(metadata)} preprocessed volumes")

    # Select samples to check
    if args.patient is not None:
        samples = metadata[metadata['study_id'] == args.patient]
        if samples.empty:
            print(f"\n❌ No samples found for patient {args.patient}")
            return
        print(f"\n✓ Found {len(samples)} samples for patient {args.patient}")
    else:
        # Random samples
        samples = metadata.sample(min(args.num_samples, len(metadata)))
        print(f"\n✓ Randomly selected {len(samples)} samples")

    # Check each sample
    volumes_dir = data_dir / 'volumes'
    all_valid = True

    for idx, (_, row) in enumerate(samples.iterrows()):
        print(f"\n{'='*70}")
        print(f"Sample {idx+1}/{len(samples)}")
        print(f"{'='*70}")
        print(f"Study ID: {row['study_id']}")
        print(f"Series ID: {row['series_id']}")
        print(f"Level: {row['level']}")
        print(f"Condition: {row['condition']}")
        print(f"Labels: Spinal={row['spinal_canal']}, Left={row['left_foraminal']}, Right={row['right_foraminal']}")

        # Load volume
        volume_path = volumes_dir / row['filepath']

        if not volume_path.exists():
            print(f"\n❌ File not found: {volume_path}")
            all_valid = False
            continue

        try:
            volume = np.load(volume_path)

            # Check stats
            is_valid = check_volume_stats(volume, row['filepath'])
            all_valid = all_valid and is_valid

            # Visualize based on mode
            if args.check_crop:
                # Show crop verification
                visualize_crop_verification(volume, row, args.rsna_dir)
            elif args.view_3d_interactive:
                # Show interactive 3D visualization
                title = f"Patient {row['study_id']} - {row['level']} - {row['condition']}"
                visualize_volume_3d_interactive(volume, title)
            elif args.view_3d:
                # Show 3D orthogonal views
                title = f"Patient {row['study_id']} - {row['level']} - {row['condition']}"
                visualize_volume_3d(volume, title)
            elif not args.show_stats:
                # Show normal visualization
                title = f"Patient {row['study_id']} - {row['level']} - {row['condition']}"
                visualize_volume(volume, title)

        except Exception as e:
            print(f"\n❌ Error loading {volume_path}: {e}")
            all_valid = False

    # Final summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    if all_valid:
        print("\n✅ All volumes look good!")
        if not args.check_crop and not args.view_3d and not args.view_3d_interactive:
            print("\n💡 Other visualization options:")
            print("  python3 visualize_npy.py --check-crop        # Verify crop extraction")
            print("  python3 visualize_npy.py --3d                # 3D orthogonal views")
            print("  python3 visualize_npy.py --3d-interactive    # Interactive 3D")
        print("\nNext steps:")
        print("  1. Run training: python3 train_rsna.py")
        print("  2. Or test model: python3 test_rsna_preprocessed.py --num-patients 5")
    else:
        print("\n⚠️  Some volumes have issues!")
        print("\nRecommendation:")
        print("  - Check preprocessing parameters")
        print("  - Re-run: python3 prepare_rsna_data.py --skip-download")

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
