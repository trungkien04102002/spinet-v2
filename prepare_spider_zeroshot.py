"""
Prepare SPIDER dataset for zero-shot evaluation.

Reads:
    spider/radiological_gradings.csv  (disease labels per IVD)
    spider/overview.csv               (imaging metadata)
    spider/images/                    (.mha files)
    spider/masks/                     (.mha segmentation masks with labels 201-207)

Outputs:
    rsna_preprocessed_spider/spider_zeroshot_test.csv
        per-IVV row with disease labels and reference to .npy file
    rsna_preprocessed_spider/volumes/<patient>_<ivd_label>.npy
        cropped IVV volumes resampled to (9, 112, 224) float32 [0, 1]

Reuses spider_dataloader.py logic for IVV cropping.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Reuse existing spider dataloader for crop logic
from spider_dataloader import SPIDERDataset


# 8 disease labels from radiological_gradings.csv with text prompts.
# Used for zero-shot evaluation: encode text via BiomedCLIP at eval time.
SPIDER_DISEASES = {
    "Modic": {
        "type": "multiclass", "num_classes": 4,
        "prompts": {
            0: "no modic changes",
            1: "modic type 1 endplate inflammation",
            2: "modic type 2 fatty endplate degeneration",
            3: "modic type 3 sclerotic endplate changes",
        },
    },
    "UP_endplate": {
        "type": "binary",
        "prompts": {
            0: "normal upper endplate",
            1: "upper endplate degeneration",
        },
    },
    "LOW_endplate": {
        "type": "binary",
        "prompts": {
            0: "normal lower endplate",
            1: "lower endplate degeneration",
        },
    },
    "Spondylolisthesis": {
        "type": "binary",
        "prompts": {
            0: "no spondylolisthesis",
            1: "spondylolisthesis with vertebral slippage",
        },
    },
    "Disc_herniation": {
        "type": "binary",
        "prompts": {
            0: "no disc herniation",
            1: "lumbar disc herniation",
        },
    },
    "Disc_narrowing": {
        "type": "binary",
        "prompts": {
            0: "normal disc height",
            1: "disc space narrowing",
        },
    },
    "Disc_bulging": {
        "type": "binary",
        "prompts": {
            0: "normal disc contour",
            1: "lumbar disc bulging",
        },
    },
    "Pfirrman_grade": {
        "type": "multiclass", "num_classes": 5,
        "prompts": {
            1: "pfirrmann grade 1 normal disc",
            2: "pfirrmann grade 2 mild disc degeneration",
            3: "pfirrmann grade 3 moderate disc degeneration",
            4: "pfirrmann grade 4 severe disc degeneration",
            5: "pfirrmann grade 5 end-stage disc degeneration",
        },
    },
}

# Label tier difficulty for paper analysis
DISEASE_TIER = {
    "Disc_narrowing": "easy",       # Similar to RSNA narrowing
    "Disc_bulging": "medium",
    "Disc_herniation": "medium",
    "Spondylolisthesis": "medium",
    "Modic": "hard",
    "UP_endplate": "hard",
    "LOW_endplate": "hard",
    "Pfirrman_grade": "hard",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spider-dir", type=str, default="spider")
    p.add_argument("--output-dir", type=str, default="rsna_preprocessed_spider")
    p.add_argument("--split", type=str, default="all", choices=["all", "training", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    spider_dir = Path(args.spider_dir)
    out_dir = Path(args.output_dir)
    volumes_dir = out_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)

    # Load disease labels
    grading_csv = spider_dir / "radiological_gradings.csv"
    overview_csv = spider_dir / "overview.csv"
    if not grading_csv.exists():
        raise FileNotFoundError(f"Missing: {grading_csv}")

    gradings = pd.read_csv(grading_csv)
    overview = pd.read_csv(overview_csv) if overview_csv.exists() else None
    print(f"Loaded {len(gradings)} grading rows")

    # Filter by split if requested
    if args.split != "all" and overview is not None:
        # overview has 'subset' column with 'training' / 'validation' / 'test'
        valid_patients = set()
        if "subset" in overview.columns:
            for _, row in overview.iterrows():
                if row.get("subset", "").strip() == args.split:
                    # new_file_name is e.g. "1_t1" -> patient id 1
                    pid = str(row["new_file_name"]).split("_")[0]
                    try:
                        valid_patients.add(int(pid))
                    except ValueError:
                        continue
        if valid_patients:
            gradings = gradings[gradings["Patient"].isin(valid_patients)]
            print(f"Filtered to {len(gradings)} rows for split='{args.split}'")

    # Build SPIDER dataset (reuses existing loader for IVV cropping)
    print("Initializing SPIDER dataset...")
    spider_ds = SPIDERDataset(data_dir=str(spider_dir), split="all", transform=None)
    print(f"SPIDER samples: {len(spider_ds)}")

    # Index spider_ds samples by (patient_id, ivd_label) for lookup.
    # SPIDERDataset.samples is a list of (patient_id, ivd_level) tuples.
    sample_index = {tuple(spider_ds.samples[idx]): idx for idx in range(len(spider_ds))}

    # Convert each gradings row to a test sample
    rows = []
    miss = 0
    for _, row in tqdm(gradings.iterrows(), total=len(gradings), desc="Preparing"):
        pid = int(row["Patient"])
        ivd = int(row["IVD label"])
        key = (pid, ivd)
        if key not in sample_index:
            miss += 1
            continue

        # Load + cache volume as .npy
        npy_path = volumes_dir / f"{pid}_{ivd}.npy"
        if not npy_path.exists():
            volume, _ = spider_ds[sample_index[key]]
            np.save(npy_path, volume.numpy().astype(np.float32))

        rec = {
            "patient_id": pid,
            "ivd_label": ivd,
            "npy_path": str(npy_path.relative_to(out_dir)),
            "Modic": int(row["Modic"]),
            "UP_endplate": int(row["UP endplate"]),
            "LOW_endplate": int(row["LOW endplate"]),
            "Spondylolisthesis": int(row["Spondylolisthesis"]),
            "Disc_herniation": int(row["Disc herniation"]),
            "Disc_narrowing": int(row["Disc narrowing"]),
            "Disc_bulging": int(row["Disc bulging"]),
            "Pfirrman_grade": int(row["Pfirrman grade"]),
        }
        rows.append(rec)

    print(f"Matched: {len(rows)}; Missing: {miss}")

    out_csv = out_dir / "spider_zeroshot_test.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Summary stats per disease
    df = pd.DataFrame(rows)
    print("\nDisease prevalence:")
    for disease in SPIDER_DISEASES:
        col = df[disease]
        if SPIDER_DISEASES[disease]["type"] == "binary":
            pos = (col == 1).sum()
            print(f"  {disease}: {pos}/{len(df)} positive ({100*pos/len(df):.1f}%)")
        else:
            print(f"  {disease}: distribution {col.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
