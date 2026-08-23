#!/usr/bin/env python3
"""
Extract Sagittal T1 per-IVD foraminal crops (#1 T1-foraminal experiment).

Why this exists
----------------
The current RSNA pipeline (``rsna_dataloader.py`` / ``prepare_rsna_data.py``)
anchors every sample on the Spinal Canal Stenosis coordinate, which is only
annotated on **Sagittal T2/STIR**. Left/right foraminal labels are then
attached to that same T2 midline crop -- but
``train_label_coordinates.csv`` actually gives left/right neural foraminal
narrowing its own coordinates on a *different* series: **Sagittal T1**
(confirmed empirically: ~9.8k Left/Right Neural Foraminal Narrowing rows are
all on "Sagittal T1" series, vs. Spinal Canal Stenosis being ~9.7k on
"Sagittal T2/STIR"). Left and right foraminal coordinates also live on
different slices (different ``instance_number``) within that same T1 series,
since the neural foramina are lateral structures, not midline like the
spinal canal.

This script extracts a (9, 112, 224) crop **per side** (left / right),
centered on that side's own T1 coordinate + slice, mirroring
``RSNASpineDataset._extract_ivd_volume`` but reading from the Sagittal T1
series instead of Sagittal T2.

Output layout (mirrors ``rsna_preprocessed/``):
    rsna_preprocessed_t1/
        volumes/<study_id>/<series_id>_<level>_<side>.npy   # side in {left, right}
        t1_metadata.csv   # filepath,study_id,series_id,level,condition,
                           # spinal_canal,left_foraminal,right_foraminal
        t1_failed.json     # (only if any sample failed to extract)

``spinal_canal`` is always -1 in this metadata: this experiment is about
foraminal-on-T1 only, spinal canal grading is untouched (still T2, still the
existing pipeline). For a given row, whichever side it was NOT centered on
is also -1 (e.g. a "left" row has left_foraminal=<label>, right_foraminal=-1).

This lets ``rsna_preprocessed_dataloader.RSNAPreprocessedDataset`` load this
directory directly via ``split='t1'`` (it looks for ``{split}_metadata.csv``
-> ``t1_metadata.csv``), no changes needed to that file.

Usage:
    # Small subset, for validating the pipeline (CPU, a few minutes):
    python3 experiments/f1_improvement/prep_t1_crops.py --limit 20

    # Full prep (all studies, CPU, DO NOT run without confirming -- takes a
    # while, comparable to the original T2 preprocessing):
    python3 experiments/f1_improvement/prep_t1_crops.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# Repo root must be on sys.path for `import rsna_dataloader` when this script
# is run directly from experiments/f1_improvement/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsna_dataloader import RSNASpineDataset  # noqa: E402  (reuse constants/helpers only)


IVD_LEVELS = RSNASpineDataset.IVD_LEVELS
SEVERITY_MAPPING = RSNASpineDataset.SEVERITY_MAPPING

# Only these two conditions are anchored on Sagittal T1 coordinates.
FORAMINAL_CONDITIONS = {
    "left_neural_foraminal_narrowing": "left",
    "right_neural_foraminal_narrowing": "right",
}


def is_sagittal_t1(desc) -> bool:
    """Analogous to RSNASpineDataset._filter_sagittal_t2, but for Sagittal T1.

    Must contain 't1' and 'sag'/'sagittal', and must NOT be axial/coronal.
    """
    if pd.isna(desc):
        return False
    desc_lower = str(desc).lower()
    has_t1 = "t1" in desc_lower
    has_sag = "sag" in desc_lower or "sagittal" in desc_lower
    is_axial = "ax" in desc_lower or "axial" in desc_lower
    is_coronal = "cor" in desc_lower or "coronal" in desc_lower
    return has_t1 and has_sag and not is_axial and not is_coronal


def get_label(labels_df: pd.DataFrame, study_id: int, condition: str, level: str) -> Optional[int]:
    """Same lookup convention as RSNASpineDataset._get_label."""
    row = labels_df[labels_df["study_id"] == study_id]
    if row.empty:
        return None
    col_name = f"{condition}_{level}"
    if col_name not in row.columns:
        return None
    label_str = row[col_name].values[0]
    if pd.isna(label_str):
        return None
    return SEVERITY_MAPPING.get(label_str, None)


def build_samples(
    data_dir: str, split: str, limit: Optional[int] = None
) -> Tuple[List[Dict], List[int], set]:
    """Build the list of per-side foraminal T1 samples.

    Returns:
        samples: list of dicts (one per study/level/side with a valid label
            on a resolvable Sagittal T1 series)
        considered_study_ids: the (possibly --limit-truncated) universe of
            study_ids we looked at
        studies_with_t1: subset of considered_study_ids that produced >=1
            sample
    """
    data_dir = Path(data_dir)

    labels_df = pd.read_csv(data_dir / f"{split}.csv")
    coords_df = pd.read_csv(data_dir / f"{split}_label_coordinates.csv")
    coords_df["condition"] = coords_df["condition"].apply(RSNASpineDataset._normalize_condition)
    coords_df["level"] = coords_df["level"].apply(RSNASpineDataset._normalize_level)
    series_df = pd.read_csv(data_dir / f"{split}_series_descriptions.csv")

    t1_mask = series_df["series_description"].apply(is_sagittal_t1)
    t1_series_df = series_df[t1_mask]
    print(
        f"  - Found {len(t1_series_df)} Sagittal T1 series across "
        f"{t1_series_df['study_id'].nunique()} studies (all studies: "
        f"{series_df['study_id'].nunique()})"
    )
    t1_series_pairs = set(zip(t1_series_df["study_id"], t1_series_df["series_id"]))

    considered_study_ids = sorted(labels_df["study_id"].unique().tolist())
    if limit is not None:
        considered_study_ids = considered_study_ids[:limit]
    study_id_set = set(considered_study_ids)

    coords_df = coords_df[coords_df["study_id"].isin(study_id_set)]

    samples: List[Dict] = []
    studies_with_t1: set = set()

    for _, row in coords_df.iterrows():
        condition = row["condition"]
        if condition not in FORAMINAL_CONDITIONS:
            continue
        level = row["level"]
        if level not in IVD_LEVELS:
            continue

        study_id = int(row["study_id"])
        series_id = int(row["series_id"])
        if (study_id, series_id) not in t1_series_pairs:
            continue

        label = get_label(labels_df, study_id, condition, level)
        if label is None:  # missing label -> skip, same convention as T2 pipeline
            continue

        samples.append(
            {
                "study_id": study_id,
                "series_id": series_id,
                "level": level,
                "condition": condition,
                "side": FORAMINAL_CONDITIONS[condition],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "instance_number": row["instance_number"],
                "label": label,
            }
        )
        studies_with_t1.add(study_id)

    return samples, considered_study_ids, studies_with_t1


def load_dicom_series(data_dir: str, split: str, study_id: int, series_id: int):
    """Mirrors RSNASpineDataset._load_dicom_series (standalone, no caching class needed)."""
    series_dir = Path(data_dir) / f"{split}_images" / str(study_id) / str(series_id)
    if not series_dir.exists():
        raise FileNotFoundError(f"Series directory not found: {series_dir}")

    dicoms = []
    for dcm_file in series_dir.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(dcm_file)
            if hasattr(ds, "ImagePositionPatient"):
                z_pos = float(ds.ImagePositionPatient[2])
            else:
                z_pos = float(ds.InstanceNumber) if hasattr(ds, "InstanceNumber") else 0.0
            dicoms.append((ds, z_pos))
        except Exception as e:
            print(f"Warning: Failed to load {dcm_file}: {e}")
            continue

    dicoms.sort(key=lambda x: x[1])
    return dicoms


def extract_ivd_volume(dicoms, center_instance, x: float, y: float) -> np.ndarray:
    """Identical crop/resize/normalize logic to RSNASpineDataset._extract_ivd_volume."""
    instance_numbers = [ds.InstanceNumber for ds, _ in dicoms]
    try:
        center_idx = instance_numbers.index(center_instance)
    except ValueError:
        center_idx = len(dicoms) // 2

    slices = []
    for offset in range(-4, 5):
        idx = center_idx + offset
        if idx < 0:
            idx = 0
        elif idx >= len(dicoms):
            idx = len(dicoms) - 1

        ds, _ = dicoms[idx]
        img = ds.pixel_array.astype(np.float32)

        crop_w = 240
        crop_h = 120
        x_int, y_int = int(x), int(y)
        x1 = max(0, x_int - crop_w // 2)
        y1 = max(0, y_int - crop_h // 2)
        x2 = min(img.shape[1], x1 + crop_w)
        y2 = min(img.shape[0], y1 + crop_h)

        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224, 112), interpolation=cv2.INTER_LINEAR)
        slices.append(resized)

    volume = np.stack(slices, axis=0)
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-8)
    return volume.astype(np.float32)


def build_metadata_row(rel_path: str, sample: Dict) -> Dict:
    return {
        "filepath": rel_path,
        "study_id": sample["study_id"],
        "series_id": sample["series_id"],
        "level": sample["level"],
        "condition": sample["condition"],
        "spinal_canal": -1,
        "left_foraminal": sample["label"] if sample["side"] == "left" else -1,
        "right_foraminal": sample["label"] if sample["side"] == "right" else -1,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract Sagittal T1 per-IVD foraminal crops for the T1-foraminal experiment"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="rsna-2024-lumbar-spine-degenerative-classification",
        help="Path to raw RSNA dataset (with train_images/, train.csv, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rsna_preprocessed_t1",
        help="Output directory for T1 crops + metadata (default: rsna_preprocessed_t1)",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train"], help="Dataset split (only 'train' has labels)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to the first N studies (sorted by study_id) -- for quick validation runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extraction (and DICOM I/O) for .npy files that already exist on disk",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RSNA Sagittal T1 Foraminal Crop Extraction")
    print("=" * 70)
    print(f"  Data dir:   {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Limit:      {args.limit if args.limit is not None else 'ALL studies'}")
    print(f"  Skip existing: {args.skip_existing}")

    print("\n[1/3] Scanning coordinates + series descriptions...")
    samples, considered_study_ids, studies_with_t1 = build_samples(args.data_dir, args.split, args.limit)

    n_considered = len(considered_study_ids)
    n_with_t1 = len(studies_with_t1)
    missing_studies = sorted(set(considered_study_ids) - studies_with_t1)

    print(f"  - Studies considered:                     {n_considered}")
    print(f"  - Studies with >=1 usable T1 foraminal sample: {n_with_t1} ({100.0 * n_with_t1 / max(n_considered, 1):.1f}%)")
    print(f"  - Studies with NO usable T1 foraminal data:    {len(missing_studies)}")
    if missing_studies:
        preview = missing_studies[:20]
        suffix = " ..." if len(missing_studies) > 20 else ""
        print(f"    -> {preview}{suffix}")
    print(f"  - Total (study, level, side) samples to extract: {len(samples)}")

    if not samples:
        print("\n No samples found -- nothing to extract. Check --data-dir / --limit.")
        return

    volumes_dir = Path(args.output_dir) / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    print("\n[2/3] Extracting volumes...")
    metadata = []
    failed = []
    dicom_cache: Dict[Tuple[int, int], List] = {}

    for sample in tqdm(samples, desc="Extracting T1 crops"):
        study_id = sample["study_id"]
        series_id = sample["series_id"]
        level = sample["level"]
        side = sample["side"]

        filename = f"{series_id}_{level}_{side}.npy"
        patient_dir = volumes_dir / str(study_id)
        filepath = patient_dir / filename
        rel_path = f"{study_id}/{filename}"

        if args.skip_existing and filepath.exists():
            metadata.append(build_metadata_row(rel_path, sample))
            continue

        try:
            cache_key = (study_id, series_id)
            if cache_key not in dicom_cache:
                dicom_cache.clear()  # only ever keep the current series in memory
                dicom_cache[cache_key] = load_dicom_series(args.data_dir, args.split, study_id, series_id)
            dicoms = dicom_cache[cache_key]

            volume = extract_ivd_volume(dicoms, sample["instance_number"], sample["x"], sample["y"])

            patient_dir.mkdir(parents=True, exist_ok=True)
            np.save(filepath, volume)

            metadata.append(build_metadata_row(rel_path, sample))
        except Exception as e:
            failed.append(
                {
                    "study_id": study_id,
                    "series_id": series_id,
                    "level": level,
                    "side": side,
                    "error": str(e),
                }
            )
            print(f"\n Warning: failed to extract study={study_id} series={series_id} level={level} side={side}: {e}")

    print("\n[3/3] Saving metadata...")
    metadata_df = pd.DataFrame(metadata)
    metadata_path = Path(args.output_dir) / "t1_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"  Metadata saved: {metadata_path} ({len(metadata_df)} rows)")

    if failed:
        failed_path = Path(args.output_dir) / "t1_failed.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"  {len(failed)} samples failed (see {failed_path})")

    print("\n" + "=" * 70)
    print("Extraction complete")
    print("=" * 70)
    print(f"  Successful: {len(metadata)} samples")
    print(f"  Failed:     {len(failed)} samples")
    print(f"  Volumes:    {volumes_dir}")

    if volumes_dir.exists():
        npy_files = list(volumes_dir.rglob("*.npy"))
        total_mb = sum(p.stat().st_size for p in npy_files) / (1024 * 1024)
        print(f"  {len(npy_files)} .npy files, {total_mb:.1f} MB total")


if __name__ == "__main__":
    main()
