"""Export the canonical RSNA patient-level split (per seed) so external SOTA
baselines train/eval on EXACTLY the same partition as our Hybrid model.

Replicates the split in train_rsna_hybrid.py verbatim:
    unique_patients = metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=0.2, random_state=seed)   # shuffle=True, no stratify

Evaluation protocol = report on the 20% validation split (there is no separate
held-out test set in this project). Seeds {42, 123, 456}.

Outputs, per seed S, under experiments/sota_comparison/splits/:
    seed{S}_train_studies.txt   # one study_id per line
    seed{S}_val_studies.txt
    seed{S}_manifest.csv        # full per-IVD rows + a 'split' column (train/val)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEEDS = [42, 123, 456]
VAL_SPLIT = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="rsna_preprocessed/train_metadata.csv")
    ap.add_argument("--out-dir", default="experiments/sota_comparison/splits")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    # .unique() preserves first-seen order -> deterministic input to train_test_split.
    unique_patients = meta["study_id"].unique()
    print(f"Loaded {len(meta)} IVD rows across {len(unique_patients)} studies")

    for seed in args.seeds:
        train_patients, val_patients = train_test_split(
            unique_patients, test_size=args.val_split, random_state=seed
        )
        train_set = set(train_patients)
        val_set = set(val_patients)

        m = meta.copy()
        m["split"] = m["study_id"].apply(
            lambda s: "train" if s in train_set else "val"
        )

        (out_dir / f"seed{seed}_train_studies.txt").write_text(
            "\n".join(str(s) for s in train_patients) + "\n"
        )
        (out_dir / f"seed{seed}_val_studies.txt").write_text(
            "\n".join(str(s) for s in val_patients) + "\n"
        )
        m.to_csv(out_dir / f"seed{seed}_manifest.csv", index=False)

        n_tr = (m["split"] == "train").sum()
        n_va = (m["split"] == "val").sum()
        print(
            f"seed {seed}: "
            f"{len(train_patients)} train studies / {len(val_patients)} val studies | "
            f"{n_tr} train rows / {n_va} val rows"
        )


if __name__ == "__main__":
    main()
