"""
SPIDER Dataset Loader for SpineNet V2 Transfer Learning

Extracts IVD regions from .mha volumes using segmentation masks.
Matches SpineNet V2 grading model format with 3 conditions per sample.

SpineNet V2 Original (RSNA):
- 3 conditions: spinal_canal, left_foraminal, right_foraminal
- Each: 3 classes (0: Normal/Mild, 1: Moderate, 2: Severe)

SPIDER Transfer Learning:
- 3 conditions: pfirrmann, spondylolisthesis, disc_herniation
- Pfirrmann: 5 classes (0-4, originally 1-5)
- Spondylolisthesis: 2 classes (0: No, 1: Yes)
- Disc herniation: 2 classes (0: No, 1: Yes)

Usage:
    from spider_dataloader import SPIDERDataset

    dataset = SPIDERDataset(
        data_dir='spider',
        split='training',
        modality='t2'
    )
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import SimpleITK as sitk
from typing import Dict, Tuple, List
import cv2


class SPIDERDataset(Dataset):
    """
    PyTorch Dataset for SPIDER with 3 conditions matching SpineNet V2 format.

    Returns same structure as RSNA dataset:
    - volume: (9, 112, 224)
    - labels: dict with 3 conditions
    """

    def __init__(
        self,
        data_dir: str = 'spider',
        split: str = 'training',  # 'training' or 'validation'
        modality: str = 't2',  # 't1' or 't2'
        num_slices: int = 9,
        height: int = 112,
        width: int = 224,
        transform=None
    ):
        """
        Args:
            data_dir: Path to SPIDER dataset directory
            split: 'training' or 'validation'
            modality: 't1' or 't2'
            num_slices: Number of slices per IVD (default: 9)
            height: Target height (default: 112)
            width: Target width (default: 224)
            transform: Optional transform
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.modality = modality
        self.num_slices = num_slices
        self.height = height
        self.width = width
        self.transform = transform

        # Load metadata
        self.overview = pd.read_csv(self.data_dir / 'overview.csv')
        self.gradings = pd.read_csv(self.data_dir / 'radiological_gradings.csv')

        # Filter by split
        self.overview = self.overview[self.overview['subset'] == split].reset_index(drop=True)

        # Build sample list
        self.samples = self._build_sample_list()

        print(f"✓ Loaded SPIDER {split} dataset:")
        print(f"  - Modality: {modality}")
        print(f"  - Samples: {len(self.samples)} IVDs")
        print(f"  - Patients: {len(self.overview)}")
        print(f"  - Format: 3 conditions (pfirrmann, spondylolisthesis, disc_herniation)")
        print(f"  - Output shape: ({num_slices}, {height}, {width})")

    def _build_sample_list(self) -> List[Tuple[int, int]]:
        """Build list of (patient_id, ivd_level) samples."""
        samples = []

        for _, row in self.overview.iterrows():
            filename = row['new_file_name']
            patient_id = int(filename.split('_')[0])
            num_discs = row['num_discs']

            # Get gradings for this patient
            patient_gradings = self.gradings[self.gradings['Patient'] == patient_id]

            # Add each IVD level that has grading
            for ivd_level in range(1, num_discs + 1):
                ivd_grading = patient_gradings[patient_gradings['IVD label'] == ivd_level]
                if len(ivd_grading) > 0:
                    samples.append((patient_id, ivd_level))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get one IVD sample.

        Returns:
            volume: torch.Tensor of shape (9, 112, 224)
            labels: dict with keys 'pfirrmann', 'spondylolisthesis', 'disc_herniation'
        """
        patient_id, ivd_level = self.samples[idx]

        # Load volume and mask
        volume = self._load_volume(patient_id)
        mask = self._load_mask(patient_id)

        # Extract IVD region using mask (label = 200 + ivd_level)
        ivd_volume = self._extract_ivd(volume, mask, ivd_level)

        # Preprocess to (9, 112, 224)
        ivd_volume = self._preprocess_ivd(ivd_volume)

        # Convert to tensor
        ivd_volume = torch.from_numpy(ivd_volume).float()

        # Apply transform if provided
        if self.transform is not None:
            ivd_volume = self.transform(ivd_volume)

        # Get labels (3 conditions)
        labels = self._get_labels(patient_id, ivd_level)

        return ivd_volume, labels

    def _load_volume(self, patient_id: int) -> np.ndarray:
        """Load MRI volume from .mha file."""
        filename = f"{patient_id}_{self.modality}.mha"
        filepath = self.data_dir / 'images' / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Volume not found: {filepath}")

        image = sitk.ReadImage(str(filepath))
        volume = sitk.GetArrayFromImage(image)  # (slices, height, width)
        return volume

    def _load_mask(self, patient_id: int) -> np.ndarray:
        """Load segmentation mask (.mha file with IVD labels 201-207)."""
        # Try current modality first, then fallback
        for mod in [self.modality, 't1', 't2']:
            filename = f"{patient_id}_{mod}.mha"
            filepath = self.data_dir / 'masks' / filename

            if filepath.exists():
                image = sitk.ReadImage(str(filepath))
                mask = sitk.GetArrayFromImage(image)
                return mask

        raise FileNotFoundError(f"Mask not found for patient {patient_id}")

    def _extract_ivd(self, volume: np.ndarray, mask: np.ndarray, ivd_level: int) -> np.ndarray:
        """
        Extract IVD region using segmentation mask.

        SPIDER masks: IVD label = 200 + ivd_level
        - Label 201: IVD level 1 (L1/L2)
        - Label 202: IVD level 2 (L2/L3)
        - ...
        - Label 207: IVD level 7 (L5/S1)

        Args:
            volume: Full spine volume (slices, H, W)
            mask: Segmentation mask (slices, H, W)
            ivd_level: IVD level (1-7)

        Returns:
            IVD volume cropped around disc
        """
        ivd_label = 200 + ivd_level

        # Find bounding box
        coords = np.where(mask == ivd_label)

        if len(coords[0]) == 0:
            # Fallback: use center region if mask not found
            center_slice = volume.shape[0] // 2
            center_h = volume.shape[1] // 2
            center_w = volume.shape[2] // 2

            slice_start = max(0, center_slice - self.num_slices // 2)
            slice_end = min(volume.shape[0], slice_start + self.num_slices)
            h_start = max(0, center_h - self.height // 2)
            h_end = min(volume.shape[1], h_start + self.height)
            w_start = max(0, center_w - self.width // 2)
            w_end = min(volume.shape[2], w_start + self.width)

            return volume[slice_start:slice_end, h_start:h_end, w_start:w_end]
        else:
            # Get bounding box from mask
            slice_min, slice_max = coords[0].min(), coords[0].max()
            h_min, h_max = coords[1].min(), coords[1].max()
            w_min, w_max = coords[2].min(), coords[2].max()

            # Add margin
            margin_slice, margin_h, margin_w = 2, 10, 10

            slice_min = max(0, slice_min - margin_slice)
            slice_max = min(volume.shape[0], slice_max + margin_slice)
            h_min = max(0, h_min - margin_h)
            h_max = min(volume.shape[1], h_max + margin_h)
            w_min = max(0, w_min - margin_w)
            w_max = min(volume.shape[2], w_max + margin_w)

            return volume[slice_min:slice_max, h_min:h_max, w_min:w_max]

    def _preprocess_ivd(self, ivd_volume: np.ndarray) -> np.ndarray:
        """
        Preprocess to (9, 112, 224) format.

        Steps:
        1. Resample to 9 slices
        2. Resize to (112, 224)
        3. Normalize to [0, 1]
        """
        num_slices_orig = ivd_volume.shape[0]

        # 1. Resample to 9 slices
        if num_slices_orig != self.num_slices:
            slice_indices = np.linspace(0, num_slices_orig - 1, self.num_slices)
            resampled = np.zeros((self.num_slices, ivd_volume.shape[1], ivd_volume.shape[2]))

            for i, idx in enumerate(slice_indices):
                idx_floor = int(np.floor(idx))
                idx_ceil = int(np.ceil(idx))

                if idx_floor == idx_ceil:
                    resampled[i] = ivd_volume[idx_floor]
                else:
                    weight = idx - idx_floor
                    resampled[i] = (1 - weight) * ivd_volume[idx_floor] + weight * ivd_volume[idx_ceil]

            ivd_volume = resampled

        # 2. Resize to (112, 224)
        resized = np.zeros((self.num_slices, self.height, self.width))
        for i in range(self.num_slices):
            resized[i] = cv2.resize(ivd_volume[i], (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # 3. Normalize to [0, 1]
        p1, p99 = np.percentile(resized, [1, 99])
        resized = np.clip(resized, p1, p99)
        resized = (resized - p1) / (p99 - p1 + 1e-8)

        return resized

    def _get_labels(self, patient_id: int, ivd_level: int) -> Dict[str, int]:
        """
        Get 3 condition labels matching SpineNet V2 format.

        Returns:
            {
                'pfirrmann': 0-4 (5 classes),
                'spondylolisthesis': 0-1 (binary),
                'disc_herniation': 0-1 (binary)
            }
        """
        grading = self.gradings[
            (self.gradings['Patient'] == patient_id) &
            (self.gradings['IVD label'] == ivd_level)
        ]

        if len(grading) == 0:
            raise ValueError(f"No grading found for patient {patient_id} IVD {ivd_level}")

        grading = grading.iloc[0]

        # Return 3 conditions (same structure as RSNA dataset)
        labels = {
            'pfirrmann': int(grading['Pfirrman grade']) - 1,  # Convert 1-5 → 0-4
            'spondylolisthesis': int(grading['Spondylolisthesis']),  # Binary 0-1
            'disc_herniation': int(grading['Disc herniation']),  # Binary 0-1
        }

        return labels


# Test script
if __name__ == "__main__":
    print("="*70)
    print("SPIDER Dataset - SpineNet V2 Transfer Learning")
    print("="*70)

    data_dir = 'spider'
    if not Path(data_dir).exists():
        print(f"\n❌ SPIDER data not found: {data_dir}")
        exit(1)

    # Create dataset
    print(f"\nLoading SPIDER dataset...")
    dataset = SPIDERDataset(data_dir=data_dir, split='training', modality='t2')

    # Test sample loading
    print("\n" + "="*70)
    print("Testing Sample Loading")
    print("="*70)
    volume, labels = dataset[0]

    print(f"\n✓ Sample loaded!")
    print(f"  - Volume shape: {volume.shape}")
    print(f"  - Volume dtype: {volume.dtype}")
    print(f"  - Volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"\n  - Labels (3 conditions):")
    print(f"    • Pfirrmann: {labels['pfirrmann']} (0-4 scale)")
    print(f"      → Original: {labels['pfirrmann'] + 1} (1-5 scale)")
    print(f"    • Spondylolisthesis: {labels['spondylolisthesis']} (0: No, 1: Yes)")
    print(f"    • Disc herniation: {labels['disc_herniation']} (0: No, 1: Yes)")

    # Test DataLoader batching
    print("\n" + "="*70)
    print("Testing DataLoader (Batching)")
    print("="*70)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_volumes, batch_labels = next(iter(loader))

    print(f"\n✓ Batch loaded!")
    print(f"  - Batch shape: {batch_volumes.shape}")
    print(f"  - Pfirrmann: {batch_labels['pfirrmann']}")
    print(f"  - Spondylolisthesis: {batch_labels['spondylolisthesis']}")
    print(f"  - Disc herniation: {batch_labels['disc_herniation']}")

    # Test with SpineNet model (will need modification for 5 classes)
    print("\n" + "="*70)
    print("Testing with SpineNet Grading Model")
    print("="*70)
    from spinenet.models.grading_baseline import GradingModelBaseline

    model = GradingModelBaseline(format='rsna')
    model.eval()

    batch_input = batch_volumes.unsqueeze(1)  # [4, 1, 9, 112, 224]

    with torch.no_grad():
        outputs = model(batch_input)

    print(f"\n✓ Model inference successful!")
    print(f"  - Output keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        print(f"  - {key}: {value.shape}")

    print("\n" + "="*70)
    print("Format Comparison")
    print("="*70)
    print("\nRSNA (original):")
    print("  - Conditions: spinal_canal, left_foraminal, right_foraminal")
    print("  - Classes: 3 each (0: Normal/Mild, 1: Moderate, 2: Severe)")
    print("\nSPIDER (transfer learning):")
    print("  - Conditions: pfirrmann, spondylolisthesis, disc_herniation")
    print("  - Classes: 5, 2, 2 respectively")
    print("\nBoth return same structure:")
    print("  - volume: (9, 112, 224)")
    print("  - labels: dict with 3 conditions")

    print("\n" + "="*70)
    print("✓ DATASET READY FOR TRANSFER LEARNING!")
    print("="*70)
    print(f"\n  - Training samples: {len(dataset)}")
    print(f"  - Compatible with SpineNet V2 architecture")
    print(f"  - Need to modify heads for different class counts")
    print()
