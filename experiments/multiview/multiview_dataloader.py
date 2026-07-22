"""
Multi-view (T2 + per-side T1) RSNA DataLoader for the leader/supporter
fusion model.

Extends the single-sequence `RSNAPreprocessedDataset`
(`rsna_preprocessed_dataloader.py`) to yield FOUR things per IVD:

    t2_vol, t1_left_vol, t1_right_vol, labels = dataset[i]

- T2 is the existing, already-preprocessed MIDLINE crop under
  `rsna_preprocessed/` (Sagittal T2), used for spinal_canal AND as the
  T2-supporter input for both foraminal heads — see
  `docs/LVTN_phase3/MULTIVIEW_RESEARCH.md` root finding.
- T1 is produced by the sibling task `experiments/f1_improvement/
  prep_t1_crops.py`. IMPORTANT (reconciled 2026-07-22): unlike T2, T1
  foraminal crops are **per-side**, because left/right neural foraminal
  narrowing is annotated on DIFFERENT slices of the Sagittal T1 series
  (the foramina are lateral structures, not midline like the canal). So
  there is NO single T1 crop per (study_id, level) — there are TWO:
      rsna_preprocessed_t1/volumes/<study_id>/<series_id>_<level>_left.npy
      rsna_preprocessed_t1/volumes/<study_id>/<series_id>_<level>_right.npy
  and ONE metadata file `rsna_preprocessed_t1/t1_metadata.csv` (NOT
  `{split}_metadata.csv`) with one row per (study, level, side); each row's
  OTHER side's label (and spinal_canal) is -1 (see prep_t1_crops.py's
  `build_metadata_row`).

That T1 directory may not exist / may be a partial subset (e.g. a
`--limit`-truncated validation run) — pass `allow_missing_t1=True` (or the
CLI flag `--allow-missing-t1`) to fall back to an all-zeros T1 volume,
per-side independently, wherever a match isn't found.

Matching key: T1 rows are matched on (study_id, level, side) — NOT
series_id, since the T1 series id differs from the T2 series id for the
same study/level.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


VOLUME_SHAPE = (9, 112, 224)
SIDES = ("left", "right")


class RSNAMultiViewDataset(Dataset):
    """Paired T2 + per-side-T1 IVD dataset for multi-view fusion training.

    Args:
        data_dir: Path to the existing T2 preprocessed directory
            (default 'rsna_preprocessed'). Must contain
            `{split}_metadata.csv` + `volumes/`.
        t1_dir: Path to the T1 preprocessed directory produced by
            `experiments/f1_improvement/prep_t1_crops.py`. Same parent
            layout as `data_dir` but metadata lives in a fixed
            `t1_metadata.csv` (not `{split}_metadata.csv` — the prep
            script only ever writes a 'train' split with labels).
        split: 'train' or 'test' — selects `{split}_metadata.csv` in
            `data_dir` (the T2/canonical sample index).
        t1_metadata_filename: name of the T1 metadata CSV inside `t1_dir`
            (default 't1_metadata.csv', matching prep_t1_crops.py).
        allow_missing_t1: If True, a missing T1 dir/metadata file, or a
            missing (study, level, side) row/file, is tolerated and
            replaced with an all-zeros volume of the same shape as T2. If
            False (default), a missing T1 source raises
            FileNotFoundError — use this once T1 crops are complete, to
            catch silent gaps.
        transform: Reserved for future T2/T1-synced augmentation. Must be
            None for now (see note below).
    """

    def __init__(
        self,
        data_dir: str = "rsna_preprocessed",
        t1_dir: str = "rsna_preprocessed_t1",
        split: str = "train",
        t1_metadata_filename: str = "t1_metadata.csv",
        allow_missing_t1: bool = False,
        transform=None,
    ):
        if transform is not None:
            raise NotImplementedError(
                "Synced T2/T1 augmentation is not implemented yet — a "
                "random flip must swap left<->right T1 crops AND labels "
                "identically. Pass transform=None. See module docstring."
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

        # --- T1 metadata (per-side; may not exist yet or be a partial
        # --- subset — sibling task experiments/f1_improvement/prep_t1_crops.py) ---
        self.t1_volumes_dir = self.t1_dir / "volumes"
        t1_metadata_path = self.t1_dir / t1_metadata_filename
        # One lookup per side: (study_id, level) -> filepath. None until
        # populated below; stays None (all sides) if T1 metadata is absent.
        self._t1_lookup: Dict[str, Optional[Dict[Tuple, str]]] = {
            "left": None,
            "right": None,
        }

        if t1_metadata_path.exists():
            t1_metadata = pd.read_csv(t1_metadata_path)
            side_of = _infer_side_column(t1_metadata)

            for side in SIDES:
                lookup: Dict[Tuple, str] = {}
                side_rows = t1_metadata[side_of == side]
                for _, row in side_rows.iterrows():
                    key = (int(row["study_id"]), str(row["level"]))
                    if key not in lookup:  # keep first match if duplicates
                        lookup[key] = row["filepath"]
                self._t1_lookup[side] = lookup

            n_left = len(self._t1_lookup["left"])
            n_right = len(self._t1_lookup["right"])
            print(
                f"[RSNAMultiViewDataset] Loaded T1 metadata from "
                f"{t1_metadata_path}: {len(t1_metadata)} rows -> "
                f"{n_left} left / {n_right} right (study, level) crops"
            )
        elif allow_missing_t1:
            warnings.warn(
                f"[RSNAMultiViewDataset] T1 metadata not found at "
                f"{t1_metadata_path} — allow_missing_t1=True, so ALL "
                f"T1 (left+right) volumes will be dummy zeros of shape "
                f"{VOLUME_SHAPE}. This is a smoke-test / dev mode only; "
                f"do not use for a real training run.",
                stacklevel=2,
            )
        else:
            raise FileNotFoundError(
                f"T1 metadata not found: {t1_metadata_path}\n"
                f"Pass allow_missing_t1=True (--allow-missing-t1) for a "
                f"dummy-zeros dev/smoke-test run, or run "
                f"experiments/f1_improvement/prep_t1_crops.py first."
            )

        print(
            f"[RSNAMultiViewDataset] Loaded {len(self.metadata)} T2 samples "
            f"from {data_dir} (split={split}); T1 dir={t1_dir} "
            f"(present={self._t1_lookup['left'] is not None})"
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def get_labels(self, idx: int) -> Dict[str, int]:
        """Fast label-only accessor (used by compute_class_weights). Labels
        always come from the T2 (canonical) metadata — the T1 metadata's
        per-side rows carry the SAME labels (with -1 for the other side /
        spinal_canal), so there is nothing extra to merge here."""
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

    def _load_t1_side(self, row, side: str) -> torch.Tensor:
        lookup = self._t1_lookup[side]
        if lookup is not None:
            key = (int(row["study_id"]), str(row["level"]))
            filepath = lookup.get(key)
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
                    f"No T1-{side} crop found for study_id={row['study_id']} "
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Returns:
            t2_vol:       torch.Tensor (9, 112, 224) — Sagittal T2 midline crop
            t1_left_vol:  torch.Tensor (9, 112, 224) — Sagittal T1 LEFT crop (or zeros)
            t1_right_vol: torch.Tensor (9, 112, 224) — Sagittal T1 RIGHT crop (or zeros)
            labels: dict with 'spinal_canal', 'left_foraminal',
                'right_foraminal'
        """
        row = self.metadata.iloc[idx]

        t2_vol = self._load_t2(row)
        t1_left_vol = self._load_t1_side(row, "left")
        t1_right_vol = self._load_t1_side(row, "right")

        labels = {
            "spinal_canal": int(row["spinal_canal"]),
            "left_foraminal": int(row["left_foraminal"]),
            "right_foraminal": int(row["right_foraminal"]),
        }

        return t2_vol, t1_left_vol, t1_right_vol, labels


def _infer_side_column(t1_metadata: pd.DataFrame) -> pd.Series:
    """Return a Series of 'left'/'right' (or '' if unknown) for each row of
    the T1 metadata, robust to either representation prep_t1_crops.py uses:
      1. a 'condition' column ('left_neural_foraminal_narrowing' / 'right_...')
      2. a '<series_id>_<level>_<side>.npy' filepath suffix
    """
    if "condition" in t1_metadata.columns:
        cond = t1_metadata["condition"].astype(str)
        side = pd.Series(
            np.where(
                cond.str.startswith("left_"),
                "left",
                np.where(cond.str.startswith("right_"), "right", ""),
            ),
            index=t1_metadata.index,
        )
        if (side != "").all():
            return side

    # Fallback: parse the filepath suffix, e.g. ".../123_l4_l5_left.npy".
    stem = t1_metadata["filepath"].astype(str).str.replace(".npy", "", regex=False)
    side = stem.str.rsplit("_", n=1).str[-1]
    return side.where(side.isin(list(SIDES)), "")


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
    t2, t1_left, t1_right, labels = dataset[0]
    print(f"T2 shape: {t2.shape}, T1-left shape: {t1_left.shape}, T1-right shape: {t1_right.shape}")
    print(f"Labels: {labels}")

    train_idx, val_idx = make_patient_split(dataset.metadata, val_split=0.2, seed=42)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")
