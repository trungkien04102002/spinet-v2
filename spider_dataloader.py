"""
SPIDER Dataset Loader for SpineNet V2 Transfer Learning

Extracts IVD regions from .mha volumes using segmentation masks.
Matches SpineNet V2 grading model format with 3 conditions per sample.

SpineNet V2 Original (RSNA):
- 3 conditions: spinal_canal, left_foraminal, right_foraminal
- Each: 3 classes (0: Normal/Mild, 1: Moderate, 2: Severe)

SPIDER Transfer Learning (Phase 4 — 8 conditions, all SPIDER labels):
- pfirrmann:         5 classes (0-4, originally 1-5)  — disc degeneration
- modic:             4 classes (0-3)                  — vertebra inflammation
- disc_narrowing:    2 classes (0: No, 1: Yes)        — disc structure
- spondylolisthesis: 2 classes (0: No, 1: Yes)        — spinal alignment
- up_endplate:       2 classes (0: No, 1: Yes)        — upper endplate damage
- low_endplate:      2 classes (0: No, 1: Yes)        — lower endplate damage
- disc_herniation:   2 classes (0: No, 1: Yes)        — disc herniation
- disc_bulging:      2 classes (0: No, 1: Yes)        — disc bulging

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

        # Filter by split. 'all' means use everything (training + validation),
        # which is what we want for zero-shot eval (no SPIDER training).
        if split != 'all':
            self.overview = self.overview[self.overview['subset'] == split].reset_index(drop=True)
        else:
            self.overview = self.overview.reset_index(drop=True)

        # Drop the 3 non-lumbar series (2 TSPINE + 1 MRI RUGPOLI).
        # LSPINE / empty / "MRI LWK" / "MRI LWK _DECENTR" are all lumbar
        # (LWK = "Lendenwervelkolom", Dutch for lumbar spine).
        bp = self.overview['BodyPartExamined'].fillna('').str.strip().str.upper()
        non_lumbar = bp.isin(['TSPINE', 'MRI RUGPOLI ZEVE'])
        if non_lumbar.any():
            self.overview = self.overview[~non_lumbar].reset_index(drop=True)

        # Filter by modality: only keep series whose filename ends with the chosen
        # modality (e.g. '*_t2' but not '*_t2_SPACE'). T2_SPACE is a different
        # acquisition type (3D isotropic) that this loader's resampling assumes away.
        names = self.overview['new_file_name'].astype(str)
        suffix = '_' + modality
        keep = names.str.endswith(suffix) & ~names.str.endswith('_SPACE')
        self.overview = self.overview[keep].reset_index(drop=True)

        # Build sample list
        self.samples = self._build_sample_list()

        print(f"✓ Loaded SPIDER {split} dataset:")
        print(f"  - Modality: {modality}")
        print(f"  - Samples: {len(self.samples)} IVDs")
        print(f"  - Patients: {len(self.overview)}")
        print(f"  - Format: 8 conditions (pfirrmann, modic, disc_narrowing, spondylolisthesis, up_endplate, low_endplate, disc_herniation, disc_bulging)")
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

    def get_labels(self, idx: int) -> Dict[str, int]:
        patient_id, ivd_level = self.samples[idx]
        return self._get_labels(patient_id, ivd_level)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get one IVD sample.

        Returns:
            volume: torch.Tensor of shape (9, 112, 224)
            labels: dict with 8 keys — see _get_labels() for full list
        """
        patient_id, ivd_level = self.samples[idx]

        # Load volume and mask. Spacing in mm/voxel for axes (1=SI, 2=PA) is needed
        # to crop in physical units that match the RSNA training distribution.
        volume, si_mm, pa_mm = self._load_volume(patient_id)
        mask = self._load_mask(patient_id)

        # Extract IVD region using mask (label = 200 + ivd_level)
        ivd_volume = self._extract_ivd(volume, mask, ivd_level, si_mm, pa_mm)

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

    # Canonical orientation for the grading model.
    #
    # SPIDER .mha files come in two patterns: some have axis 0 as sagittal slices
    # (e.g. 107_t2 with shape (17, 512, 512)), others have axis 2 as sagittal slices
    # (e.g. 1_t2 with shape (578, 448, 50)). Without standardization, the same
    # crop/resample logic interprets entirely different anatomical axes per case.
    #
    # 'PIR' reorients so that after sitk.GetArrayFromImage the axes are:
    #   axis 0 = R direction = lateral (sagittal slice axis, smallest count, largest spacing)
    #   axis 1 = I direction = superior->inferior (image vertical, head at index 0)
    #   axis 2 = P direction = anterior->posterior (image horizontal width)
    # This matches the (slices, H=112, W=224) convention the grading model expects.
    _CANONICAL_ORIENT = 'PIR'

    def _load_volume(self, patient_id: int) -> Tuple[np.ndarray, float, float]:
        """Load MRI volume from .mha, reorient, and return (volume, si_mm, pa_mm).

        After PIR reorientation, sitk GetSpacing returns (i_spc, j_spc, k_spc)
        for physical axes (P, I, R). Numpy array axes map as:
          axis 1 (numpy) = j (sitk) = I direction → spacing index 1 (si_mm)
          axis 2 (numpy) = i (sitk) = P direction → spacing index 0 (pa_mm)
        """
        filename = f"{patient_id}_{self.modality}.mha"
        filepath = self.data_dir / 'images' / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Volume not found: {filepath}")

        image = sitk.ReadImage(str(filepath))
        image = sitk.DICOMOrient(image, self._CANONICAL_ORIENT)
        volume = sitk.GetArrayFromImage(image)
        spacing = image.GetSpacing()
        si_mm = float(spacing[1])
        pa_mm = float(spacing[0])
        return volume, si_mm, pa_mm

    def _load_mask(self, patient_id: int) -> np.ndarray:
        """Load segmentation mask (.mha with IVD labels 201-207), reoriented to canonical."""
        # Try current modality first, then fallback. T1/T2 from the same patient
        # share coordinate system in SPIDER (verified across all 48 t1-only-mask cases).
        for mod in [self.modality, 't1', 't2']:
            filename = f"{patient_id}_{mod}.mha"
            filepath = self.data_dir / 'masks' / filename

            if filepath.exists():
                image = sitk.ReadImage(str(filepath))
                image = sitk.DICOMOrient(image, self._CANONICAL_ORIENT)
                mask = sitk.GetArrayFromImage(image)
                return mask

        raise FileNotFoundError(f"Mask not found for patient {patient_id}")

    # Physical crop extent in millimetres around the disc center.
    # Chosen to match RSNA training crop (240 px PA × 120 px SI at ~0.5 mm/px ≈
    # 120 mm × 60 mm), which captures the disc plus 1-2 vertebra of context above
    # and below — the distribution the grading backbone was trained on.
    _CROP_SI_MM = 60.0
    _CROP_PA_MM = 120.0

    def _extract_ivd(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        ivd_level: int,
        si_mm: float,
        pa_mm: float,
    ) -> np.ndarray:
        """
        Extract IVD region centered on the disc with RSNA-matched physical extent.

        SPIDER masks: IVD label = 200 + ivd_level (label 201..207, lowest disc up).

        Args:
            volume: Full spine volume in canonical (LR_slices, SI, PA) ordering.
            mask:   Segmentation mask in same ordering and shape.
            ivd_level: IVD level (1-7).
            si_mm:  mm per voxel along axis 1 (SI direction).
            pa_mm:  mm per voxel along axis 2 (PA direction).

        Returns:
            Cropped volume of shape ~(num_slices, ~_CROP_SI_MM/si_mm, ~_CROP_PA_MM/pa_mm).
            The in-plane physical aspect is exactly _CROP_PA_MM:_CROP_SI_MM = 2:1, so the
            downstream resize to (112, 224) preserves anatomical aspect (modulo edge
            clipping near the volume boundary, which mirrors RSNA behavior).
        """
        ivd_label = 200 + ivd_level
        coords = np.where(mask == ivd_label)

        if len(coords[0]) == 0:
            # Fallback: center of volume if mask label not found (rare).
            slice_center = volume.shape[0] // 2
            si_center = volume.shape[1] // 2
            pa_center = volume.shape[2] // 2
        else:
            slice_center = (int(coords[0].min()) + int(coords[0].max())) // 2
            si_center = (int(coords[1].min()) + int(coords[1].max())) // 2
            pa_center = (int(coords[2].min()) + int(coords[2].max())) // 2

        # Sagittal slice axis: take num_slices consecutive slices centered on the
        # disc's lateral midline (matches RSNA's [-4, +4] around center IVD).
        half_slices = self.num_slices // 2
        s_start = max(0, slice_center - half_slices)
        s_end = min(volume.shape[0], s_start + self.num_slices)
        # If clipped at the right boundary, shift the start back to keep num_slices.
        if s_end - s_start < self.num_slices:
            s_start = max(0, s_end - self.num_slices)

        # In-plane: crop physical extent in mm, converted to voxels via spacing.
        half_si_v = max(1, int(round((self._CROP_SI_MM / 2) / si_mm)))
        half_pa_v = max(1, int(round((self._CROP_PA_MM / 2) / pa_mm)))

        si_start = max(0, si_center - half_si_v)
        si_end = min(volume.shape[1], si_center + half_si_v)
        pa_start = max(0, pa_center - half_pa_v)
        pa_end = min(volume.shape[2], pa_center + half_pa_v)

        return volume[s_start:s_end, si_start:si_end, pa_start:pa_end]

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
        Get 8 SPIDER condition labels for transfer learning.

        Returns:
            {
                'pfirrmann':         0-4 (5 classes, originally 1-5),
                'modic':             0-3 (4 classes),
                'disc_narrowing':    0 or 1 (binary),
                'spondylolisthesis': 0 or 1 (binary),
                'up_endplate':       0 or 1 (binary),
                'low_endplate':      0 or 1 (binary),
                'disc_herniation':   0 or 1 (binary),
                'disc_bulging':      0 or 1 (binary),
            }
        """
        grading = self.gradings[
            (self.gradings['Patient'] == patient_id) &
            (self.gradings['IVD label'] == ivd_level)
        ]

        if len(grading) == 0:
            raise ValueError(f"No grading found for patient {patient_id} IVD {ivd_level}")

        grading = grading.iloc[0]

        return {
            'pfirrmann':         int(grading['Pfirrman grade']) - 1,
            'modic':             int(grading['Modic']),
            'disc_narrowing':    int(grading['Disc narrowing']),
            'spondylolisthesis': int(grading['Spondylolisthesis']),
            'up_endplate':       int(grading['UP endplate']),
            'low_endplate':      int(grading['LOW endplate']),
            'disc_herniation':   int(grading['Disc herniation']),
            'disc_bulging':      int(grading['Disc bulging']),
        }


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
    print(f"\n  - Labels (4 conditions):")
    print(f"    • Pfirrmann:         {labels['pfirrmann']} (0-4 scale, original 1-5)")
    print(f"    • Modic:             {labels['modic']} (0-3, 4 classes)")
    print(f"    • Disc narrowing:    {labels['disc_narrowing']} (0: No, 1: Yes)")
    print(f"    • Spondylolisthesis: {labels['spondylolisthesis']} (0: No, 1: Yes)")

    # Test DataLoader batching
    print("\n" + "="*70)
    print("Testing DataLoader (Batching)")
    print("="*70)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch_volumes, batch_labels = next(iter(loader))

    print(f"\n✓ Batch loaded!")
    print(f"  - Batch shape: {batch_volumes.shape}")
    print(f"  - Pfirrmann:         {batch_labels['pfirrmann']}")
    print(f"  - Modic:             {batch_labels['modic']}")
    print(f"  - Disc narrowing:    {batch_labels['disc_narrowing']}")
    print(f"  - Spondylolisthesis: {batch_labels['spondylolisthesis']}")

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
    print("\nSPIDER (transfer learning, 8 labels):")
    print("  - Conditions: pfirrmann, modic, disc_narrowing, spondylolisthesis,")
    print("                up_endplate, low_endplate, disc_herniation, disc_bulging")
    print("  - Classes: 5, 4, 2, 2, 2, 2, 2, 2 respectively")
    print("\nBoth return same structure:")
    print("  - volume: (9, 112, 224)")
    print("  - labels: dict (3 keys for RSNA / 8 keys for SPIDER)")

    print("\n" + "="*70)
    print("✓ DATASET READY FOR TRANSFER LEARNING!")
    print("="*70)
    print(f"\n  - Training samples: {len(dataset)}")
    print(f"  - Compatible with SpineNet V2 architecture")
    print(f"  - Need to modify heads for different class counts")
    print()
