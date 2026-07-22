"""
Multi-view (T2 + T1) RSNA DataLoader for the leader/supporter fusion model.

Extends the single-sequence `RSNAPreprocessedDataset`
(`rsna_preprocessed_dataloader.py`) to yield a *pair* of volumes per IVD:

    t2_vol, t1_vol, labels = dataset[i]

- T2 is the existing, already-preprocessed crop under `rsna_preprocessed/`
  (Sagittal T2, used today for all 3 conditions — see
  `docs/LVTN_phase3/MULTIVIEW_RESEARCH.md` root finding).
- T1 is the Sagittal T1 crop that a *sibling* preprocessing task produces
  under a directory with the SAME layout (`<t1_dir>/train_metadata.csv` +
  `<t1_dir>/volumes/<study_id>/<series_id>_<level>.npy`). That directory may
  not exist yet on this machine — pass `allow_missing_t1=True` (or the CLI
  flag `--allow-missing-t1`) to fall back to an all-zeros T1 volume so the
  rest of the pipeline (model/training loop) can be smoke-tested today.

Matching key: T1 and T2 crops are matched on (study_id, level) — NOT
series_id, since the T1 and T2 sagittal series are different DICOM series
for the same study/level. If more than one T1 row exists for a
(study_id, level) pair (e.g. multiple T1 series), the first is used.

NOTE on augmentation: to keep T1/T2 crops spatially in sync (e.g. a
random left-right flip must be applied identically to both volumes AND
swap the left/right labels the same way), this dataset does NOT wire in
`spinenet.augmentation.get_training_augmentation` yet. That is left for a
follow-up once T1 crops exist for real — see `docs/LVTN_phase3/
MULTIVIEW_RESEARCH.md` staged plan (#2/#3). `transform` is accepted here
only as a future extension point and is currently unused when set to the
default `None`; passing a transform raises `NotImplementedError` until the
sync logic is written.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


VOLUME_SHAPE = (9, 112, 224)


class RSNAMultiViewDataset(Dataset):
    """Paired T2 + T1 IVD dataset for multi-view fusion training.

    Args:
        data_dir: Path to the existing T2 preprocessed directory
            (default 'rsna_preprocessed'). Must contain
            `{split}_metadata.csv` + `volumes/`.
        t1_dir: Path to the T1 preprocessed directory, same layout as
            `data_dir`. Produced by a sibling task; may not exist yet.
        split: 'train' or 'test' (selects `{split}_metadata.csv` in
            `data_dir`; if `t1_dir` also has a matching metadata file for
            the same split it is used, otherwise T1 falls back per-sample).
        allow_missing_t1: If True, missing T1 metadata/files/dir are
            tolerated and replaced with an all-zeros volume of the same
            shape as T2. If False (default), a missing T1 source raises
            FileNotFoundError — use this once T1 crops are real, to catch
            silent gaps.
        transform: Reserved for future T2/T1-synced augmentation. Must be
            None for now.
    """

    def __init__(
        self,
        data_dir: str = "rsna_preprocessed",
        t1_dir: str = "rsna_preprocessed_t1",
        split: str = "train",
        allow_missing_t1: bool = False,
        transform=None,
    ):
        if transform is not None:
            raise NotImplementedError(
                "Synced T2/T1 augmentation is not implemented yet — pass "
                "transform=None. See module docstring."
            )
        self.transform = transform

        self.data_dir = Path(data_dir)
        self.t1_dir = Path(t1_dir)
        self.split = split
        self.allow_missing_t1 = allow_missing_t1

        # --- T2 metadata (must exist; this is the canonical sample index) ---
        metadata_path = self.data_dir / f"{split}_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"T2 metadata not found: {metadata_path}\n"
                f"Run prepare_rsna_data.py first to preprocess the T2 dataset."
            )
        self.metadata = pd.read_csv(metadata_path)
        self.volumes_dir = self.data_dir / "volumes"

        # --- T1 metadata (may not exist yet — sibling preprocessing task) ---
        self.t1_volumes_dir = self.t1_dir / "volumes"
        t1_metadata_path = self.t1_dir / f"{split}_metadata.csv"
        self._t1_lookup: Optional[Dict[Tuple, str]] = None

        if t1_metadata_path.exists():
            t1_metadata = pd.read_csv(t1_metadata_path)
            # Build a (study_id, level) -> filepath lookup. Keep first match
            # if duplicates exist (e.g. multiple T1 series per level).
            self._t1_lookup = {}
            for _, row in t1_metadata.iterrows():
                key = (int(row["study_id"]), str(row["level"]))
                if key not in self._t1_lookup:
                    self._t1_lookup[key] = row["filepath"]
            print(
                f"[RSNAMultiViewDataset] Loaded T1 metadata: "
                f"{len(t1_metadata)} rows from {t1_metadata_path}"
            )
        elif allow_missing_t1:
            warnings.warn(
                f"[RSNAMultiViewDataset] T1 metadata not found at "
                f"{t1_metadata_path} — allow_missing_t1=True, so ALL T1 "
                f"volumes will be dummy zeros of shape {VOLUME_SHAPE}. "
                f"This is a smoke-test / dev mode only; do not use for a "
                f"real training run.",
                stacklevel=2,
            )
        else:
            raise FileNotFoundError(
                f"T1 metadata not found: {t1_metadata_path}\n"
                f"Pass allow_missing_t1=True (--allow-missing-t1) for a "
                f"dummy-zeros dev/smoke-test run, or produce T1 crops first."
            )

        print(
            f"[RSNAMultiViewDataset] Loaded {len(self.metadata)} T2 samples "
            f"from {data_dir} (split={split}); T1 dir={t1_dir} "
            f"(present={self._t1_lookup is not None})"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def get_labels(self, idx: int) -> Dict[str, int]:
        """Fast label-only accessor (used by compute_class_weights)."""
        row = self.metadata.iloc[idx]
        return {
            "spinal_canal": int(row["spinal_canal"]),
            "left_foraminal": int(row["left_foraminal"]),
            "right_foraminal": int(row["right_foraminal"]),
        }

    def _load_t2(self, row) -> torch.Tensor:
        volume_path = self.volumes_dir / row["filepath"]
        volume = np.load(volume_path)
        return torch.from_numpy(volume).float()

    def _load_t1(self, row) -> torch.Tensor:
        if self._t1_lookup is not None:
            key = (int(row["study_id"]), str(row["level"]))
            filepath = self._t1_lookup.get(key)
            if filepath is not None:
                t1_path = self.t1_volumes_dir / filepath
                if t1_path.exists():
                    volume = np.load(t1_path)
                    return torch.from_numpy(volume).float()
                elif not self.allow_missing_t1:
                    raise FileNotFoundError(
                        f"T1 metadata references missing file: {t1_path}"
                    )
            elif not self.allow_missing_t1:
                raise FileNotFoundError(
                    f"No T1 crop found for study_id={row['study_id']} "
                    f"level={row['level']} and allow_missing_t1=False"
                )
        elif not self.allow_missing_t1:
            # Should not happen — constructor already raised in this case —
            # but guard defensively.
            raise FileNotFoundError(
                "T1 metadata/dir not available and allow_missing_t1=False"
            )

        # Dummy zeros fallback (allow_missing_t1=True path).
        return torch.zeros(VOLUME_SHAPE, dtype=torch.float32)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Returns:
            t2_vol: torch.Tensor (9, 112, 224) — Sagittal T2 crop
            t1_vol: torch.Tensor (9, 112, 224) — Sagittal T1 crop (or zeros)
            labels: dict with 'spinal_canal', 'left_foraminal',
                'right_foraminal'
        """
        row = self.metadata.iloc[idx]

        t2_vol = self._load_t2(row)
        t1_vol = self._load_t1(row)

        labels = {
            "spinal_canal": int(row["spinal_canal"]),
            "left_foraminal": int(row["left_foraminal"]),
            "right_foraminal": int(row["right_foraminal"]),
        }

        return t2_vol, t1_vol, labels


def make_patient_split(metadata: pd.DataFrame, val_split: float, seed: int):
    """Seed-42-identical patient-level split — mirrors train_rsna_baseline.py
    §[2/6] EXACTLY (same sklearn call, same random_state) so multi-view runs
    are comparable to the single-view harness. Returns (train_indices,
    val_indices) as lists positional into `metadata`.
    """
    from sklearn.model_selection import train_test_split

    unique_patients = metadata["study_id"].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_split, random_state=seed
    )
    train_indices = metadata[metadata["study_id"].isin(train_patients)].index.tolist()
    val_indices = metadata[metadata["study_id"].isin(val_patients)].index.tolist()
    return train_indices, val_indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-view dataset self-test")
    parser.add_argument("--data-dir", type=str, default="rsna_preprocessed")
    parser.add_argument("--t1-dir", type=str, default="rsna_preprocessed_t1")
    parser.add_argument("--allow-missing-t1", action="store_true")
    args = parser.parse_args()

    dataset = RSNAMultiViewDataset(
        data_dir=args.data_dir,
        t1_dir=args.t1_dir,
        split="train",
        allow_missing_t1=args.allow_missing_t1,
    )
    print(f"Dataset size: {len(dataset)}")
    t2, t1, labels = dataset[0]
    print(f"T2 shape: {t2.shape}, T1 shape: {t1.shape}")
    print(f"Labels: {labels}")

    train_idx, val_idx = make_patient_split(dataset.metadata, val_split=0.2, seed=42)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")
