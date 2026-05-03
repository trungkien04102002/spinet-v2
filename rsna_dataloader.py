"""
RSNA 2024 Lumbar Spine Dataset Loader

Extracts IVD volumes from Sagittal T2 MRI scans for grading model input.
Follows SpineNetV2 preprocessing specifications.

Usage:
    from rsna_dataloader import RSNASpineDataset

    dataset = RSNASpineDataset(
        data_dir='rsna-2024-lumbar-spine-degenerative-classification',
        split='train'
    )

    # Get one sample
    ivd_volume, labels = dataset[0]
    # ivd_volume: torch.Tensor of shape (9, 112, 224)
    # labels: dict with keys 'spinal_canal', 'left_foraminal', 'right_foraminal'
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2


class RSNASpineDataset(Dataset):
    """
    PyTorch Dataset for RSNA 2024 Lumbar Spine Degenerative Classification.

    Extracts IVD volumes from Sagittal T2 MRI using provided coordinates.
    """

    # Label mapping: RSNA string → integer
    SEVERITY_MAPPING = {
        'Normal/Mild': 0,
        'Moderate': 1,
        'Severe': 2
    }

    # IVD levels
    IVD_LEVELS = ['l1_l2', 'l2_l3', 'l3_l4', 'l4_l5', 'l5_s1']

    # Conditions
    CONDITIONS = [
        'spinal_canal_stenosis',
        'left_neural_foraminal_narrowing',
        'right_neural_foraminal_narrowing'
    ]

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        levels: Optional[List[str]] = None,
        transform=None,
        cache_images: bool = False
    ):
        """
        Args:
            data_dir: Path to RSNA dataset directory
            split: 'train' or 'test'
            levels: List of IVD levels to use (default: all 5 levels)
            transform: Optional transform to apply
            cache_images: If True, cache DICOM images in memory (faster but uses more RAM)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.levels = levels or self.IVD_LEVELS
        self.transform = transform
        self.cache_images = cache_images
        self._image_cache = {}

        # Load CSV files
        print(f"Loading {split} dataset from {data_dir}...")
        self._load_metadata()

        # Build sample list
        self.samples = self._build_sample_list()
        print(f"✓ Loaded {len(self.samples)} IVD samples")

    def _load_metadata(self):
        """Load all metadata CSV files."""
        # Labels
        self.labels_df = pd.read_csv(self.data_dir / f'{self.split}.csv')

        # Coordinates (x, y, instance_number for each IVD)
        self.coords_df = pd.read_csv(self.data_dir / f'{self.split}_label_coordinates.csv')

        # Normalize condition and level names in coords_df
        self.coords_df['condition'] = self.coords_df['condition'].apply(self._normalize_condition)
        self.coords_df['level'] = self.coords_df['level'].apply(self._normalize_level)

        # Series descriptions (to identify Sagittal T2)
        self.series_df = pd.read_csv(self.data_dir / f'{self.split}_series_descriptions.csv')

        # Filter only Sagittal T2 series
        self.sagittal_t2_series = self._filter_sagittal_t2()
        print(f"  - Found {len(self.sagittal_t2_series)} Sagittal T2 series")

    @staticmethod
    def _normalize_condition(condition: str) -> str:
        """Normalize condition name from CSV format to code format.

        Examples:
            "Spinal Canal Stenosis" -> "spinal_canal_stenosis"
            "Left Neural Foraminal Narrowing" -> "left_neural_foraminal_narrowing"
        """
        return condition.lower().replace(' ', '_')

    @staticmethod
    def _normalize_level(level: str) -> str:
        """Normalize level name from CSV format to code format.

        Examples:
            "L1/L2" -> "l1_l2"
            "L5/S1" -> "l5_s1"
        """
        return level.lower().replace('/', '_')

    def _filter_sagittal_t2(self) -> pd.DataFrame:
        """Filter series to only Sagittal T2."""
        # Common Sagittal T2 keywords
        t2_keywords = ['t2', 'sag', 'sagittal']

        def is_sagittal_t2(desc: str) -> bool:
            if pd.isna(desc):
                return False
            desc_lower = desc.lower()
            # Must contain 't2' and 'sag'/'sagittal'
            has_t2 = 't2' in desc_lower
            has_sag = 'sag' in desc_lower or 'sagittal' in desc_lower
            # Exclude axial, coronal
            is_axial = 'ax' in desc_lower or 'axial' in desc_lower
            is_coronal = 'cor' in desc_lower or 'coronal' in desc_lower

            return has_t2 and has_sag and not is_axial and not is_coronal

        mask = self.series_df['series_description'].apply(is_sagittal_t2)
        return self.series_df[mask]

    def _build_sample_list(self) -> List[Dict]:
        """Build list of all valid IVD samples."""
        samples = []

        for _, coord_row in self.coords_df.iterrows():
            study_id = coord_row['study_id']
            series_id = coord_row['series_id']
            level = coord_row['level']
            condition = coord_row['condition']

            # Skip if not in requested levels
            if level not in self.levels:
                continue

            # Skip if not Sagittal T2
            is_t2 = ((self.sagittal_t2_series['study_id'] == study_id) &
                     (self.sagittal_t2_series['series_id'] == series_id)).any()
            if not is_t2:
                continue

            # Skip if condition not in our 3 tasks
            if condition not in self.CONDITIONS:
                continue

            # Get label
            label = self._get_label(study_id, condition, level)
            if label is None:  # Skip if label missing
                continue

            # Create sample entry
            sample = {
                'study_id': study_id,
                'series_id': series_id,
                'level': level,
                'condition': condition,
                'x': coord_row['x'],
                'y': coord_row['y'],
                'instance_number': coord_row['instance_number'],
                'label': label
            }
            samples.append(sample)

        return samples

    def _get_label(self, study_id: int, condition: str, level: str) -> Optional[int]:
        """Get label for a specific study/condition/level."""
        # Find row in labels_df
        row = self.labels_df[self.labels_df['study_id'] == study_id]
        if row.empty:
            return None

        # Column name: e.g., 'spinal_canal_stenosis_l1_l2'
        col_name = f"{condition}_{level}"
        if col_name not in row.columns:
            return None

        label_str = row[col_name].values[0]
        if pd.isna(label_str):
            return None

        return self.SEVERITY_MAPPING.get(label_str, None)

    def _load_dicom_series(self, study_id: int, series_id: int) -> List[Tuple[pydicom.Dataset, float]]:
        """
        Load all DICOM files for a series and sort by ImagePositionPatient[2].

        Returns:
            List of (dicom_dataset, z_position) sorted by z_position
        """
        # Check cache
        cache_key = (study_id, series_id)
        if self.cache_images and cache_key in self._image_cache:
            return self._image_cache[cache_key]

        # Find DICOM files
        series_dir = self.data_dir / f'{self.split}_images' / str(study_id) / str(series_id)
        if not series_dir.exists():
            raise FileNotFoundError(f"Series directory not found: {series_dir}")

        # Load all DICOMs
        dicoms = []
        for dcm_file in series_dir.glob('*.dcm'):
            try:
                ds = pydicom.dcmread(dcm_file)
                # Get z-position (ImagePositionPatient[2])
                if hasattr(ds, 'ImagePositionPatient'):
                    z_pos = float(ds.ImagePositionPatient[2])
                else:
                    # Fallback: use instance number
                    z_pos = float(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0.0

                dicoms.append((ds, z_pos))
            except Exception as e:
                print(f"Warning: Failed to load {dcm_file}: {e}")
                continue

        # Sort by z-position (critical for correct slice order!)
        dicoms.sort(key=lambda x: x[1])

        # Cache if enabled
        if self.cache_images:
            self._image_cache[cache_key] = dicoms

        return dicoms

    def _extract_ivd_volume(
        self,
        dicoms: List[Tuple[pydicom.Dataset, float]],
        center_instance: int,
        x: float,
        y: float
    ) -> np.ndarray:
        """
        Extract 9-slice IVD volume centered at (x, y, center_instance).

        Args:
            dicoms: List of (dicom, z_position) sorted by z
            center_instance: Instance number of center slice
            x, y: Center coordinates in pixels

        Returns:
            numpy array of shape (9, 112, 224)
        """
        # Find center slice index
        instance_numbers = [ds.InstanceNumber for ds, _ in dicoms]
        try:
            center_idx = instance_numbers.index(center_instance)
        except ValueError:
            # Center instance not found, use closest
            center_idx = len(dicoms) // 2

        # Extract 9 slices: [center-4, ..., center, ..., center+4]
        slices = []
        for offset in range(-4, 5):  # -4, -3, -2, -1, 0, 1, 2, 3, 4
            idx = center_idx + offset

            # Handle boundary cases (pad with edge slices)
            if idx < 0:
                idx = 0
            elif idx >= len(dicoms):
                idx = len(dicoms) - 1

            ds, _ = dicoms[idx]

            # Get pixel array
            img = ds.pixel_array.astype(np.float32)

            # Crop around (x, y) with 2:1 aspect ratio
            # Use 240x120 crop before resize (allows for variation in IVD size)
            crop_w = 240
            crop_h = 120

            x_int, y_int = int(x), int(y)
            x1 = max(0, x_int - crop_w // 2)
            y1 = max(0, y_int - crop_h // 2)
            x2 = min(img.shape[1], x1 + crop_w)
            y2 = min(img.shape[0], y1 + crop_h)

            cropped = img[y1:y2, x1:x2]

            # Resize to 112 x 224 (height x width)
            resized = cv2.resize(cropped, (224, 112), interpolation=cv2.INTER_LINEAR)

            slices.append(resized)

        # Stack to (9, 112, 224)
        volume = np.stack(slices, axis=0)

        # Normalize (simple percentile clipping)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-8)  # Scale to [0, 1]

        return volume

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get one IVD sample.

        Returns:
            volume: torch.Tensor of shape (9, 112, 224)
            labels: dict with keys from CONDITIONS, values in {0, 1, 2}
        """
        sample = self.samples[idx]

        # Load DICOM series
        dicoms = self._load_dicom_series(sample['study_id'], sample['series_id'])

        # Extract IVD volume
        volume = self._extract_ivd_volume(
            dicoms,
            sample['instance_number'],
            sample['x'],
            sample['y']
        )

        # Convert to torch tensor
        volume = torch.from_numpy(volume).float()

        # Get all labels for this study/level (built before transform so the
        # augmentation pipeline can swap left_/right_ pairs on horizontal flip).
        labels = {}
        for condition in self.CONDITIONS:
            label = self._get_label(sample['study_id'], condition, sample['level'])
            # Map condition name to model output key
            if 'spinal_canal' in condition:
                key = 'spinal_canal'
            elif 'left' in condition:
                key = 'left_foraminal'
            elif 'right' in condition:
                key = 'right_foraminal'
            else:
                continue

            labels[key] = label if label is not None else -1  # -1 for missing labels

        if self.transform is not None:
            volume, labels = self.transform(volume, labels)

        return volume, labels


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("="*70)
    print("RSNA 2024 Dataset Loader - Quick Test")
    print("="*70)

    # Check if dataset exists
    data_dir = 'rsna-2024-lumbar-spine-degenerative-classification'
    if not os.path.exists(data_dir):
        print(f"\n❌ Dataset not found at: {data_dir}")
        print("\nPlease:")
        print("  1. Download RSNA 2024 dataset from Kaggle")
        print("  2. Extract to this directory")
        print("  3. Run this script again")
        sys.exit(1)

    # Create dataset
    try:
        dataset = RSNASpineDataset(
            data_dir=data_dir,
            split='train',
            levels=['l3_l4', 'l4_l5'],  # Test with 2 levels only
            cache_images=False
        )

        print(f"\n✓ Dataset created successfully!")
        print(f"  - Total samples: {len(dataset)}")

        # Test loading one sample
        if len(dataset) > 0:
            print(f"\nTesting sample loading...")
            volume, labels = dataset[0]

            print(f"\n✓ Sample loaded!")
            print(f"  - Volume shape: {volume.shape}")
            print(f"  - Volume dtype: {volume.dtype}")
            print(f"  - Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
            print(f"  - Labels: {labels}")

            # Check format
            assert volume.shape == (9, 112, 224), f"Wrong shape: {volume.shape}"
            assert volume.dtype == torch.float32, f"Wrong dtype: {volume.dtype}"

            print(f"\n✓ Format verification passed!")
            print(f"\nDataset ready for training! 🚀")
        else:
            print("\n⚠ No samples found. Check your data filtering.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
