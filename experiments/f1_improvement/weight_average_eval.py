#!/usr/bin/env python3
"""
THROWAWAY SPIKE (2026-08-22) -- does weight averaging beat the best single epoch?

Run #1 (T1-foraminal) plateaued into a noisy band: epochs 13-20 oscillate inside
Severe F1 0.273-0.310 with a swing of +/-0.02 to 0.03, and `--select-by severe_f1`
saved the top of that band (0.3095) rather than the plateau (~0.291). IMWA
(arXiv:2404.16331, Pattern Recognition) reports that averaging model weights helps
class-imbalanced learning specifically, so this script tests it on the checkpoints we
already have. No training, no GPU-hours beyond a few inference passes.

It deliberately imports the dataset, split and `evaluate()` from train_t1_foraminal so
the evaluation protocol is byte-identical to the run being compared against. Do not
reimplement the metric here.

Usage (on the box, from the repo root, venv activated):
    python3 experiments/f1_improvement/weight_average_eval.py \
        --ckpt-dir checkpoints/t1_foraminal --epochs 5 10 15 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset  # noqa: E402
from experiments.f1_improvement.train_t1_foraminal import (  # noqa: E402
    CONDITIONS,
    build_model,
    evaluate,
)


def average_state_dicts(state_dicts):
    """Mean of the float tensors; integer tensors taken from the last checkpoint.

    `num_batches_tracked` is int64 and averaging it is meaningless. The BatchNorm
    running_mean / running_var ARE averaged: freeze_backbone() only clears
    requires_grad, it does not put BN into eval mode, so those buffers drifted across
    epochs and averaging them is the same kind of statistic-smoothing as averaging the
    weights.
    """
    out = {}
    for key in state_dicts[0]:
        vals = [sd[key] for sd in state_dicts]
        if vals[0].dtype.is_floating_point:
            out[key] = torch.stack([v.float() for v in vals], dim=0).mean(dim=0).to(vals[0].dtype)
        else:
            out[key] = vals[-1].clone()
    return out


def severe_f1_of(per_class_metrics):
    """Mean Severe F1 over the conditions that have valid labels -- same reduction
    train_t1_foraminal.py uses when it logs avg_severe_f1."""
    vals = [
        per_class_metrics[c]["f1"][2]
        for c in per_class_metrics
        if per_class_metrics.get(c) is not None and len(per_class_metrics[c]["f1"]) >= 3
    ]
    return float(np.mean(vals)) if vals else float("nan")


def report(tag, per_class_metrics, accs, val_loss):
    line = f"{tag:<28} sevF1 {severe_f1_of(per_class_metrics):.4f}  loss {val_loss:.4f}"
    for c in CONDITIONS:
        m = per_class_metrics.get(c)
        if m is None:
            line += f"  | {c}: n/a"
            continue
        p, r, f1 = m["precision"][2], m["recall"][2], m["f1"][2]
        line += f"  | {c[:5]}: F1 {f1:.3f} P {p:.3f} R {r:.3f}"
    line += f"  | acc {np.mean([accs[c] for c in CONDITIONS]):.3f}"
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser(description="Weight-averaging spike for run #1")
    ap.add_argument("--ckpt-dir", type=str, default="checkpoints/t1_foraminal")
    ap.add_argument("--prefix", type=str, default="t1_foraminal_cbam")
    ap.add_argument("--epochs", type=int, nargs="+", default=[5, 10, 15, 20],
                    help="Which periodic checkpoints to consider")
    ap.add_argument("--data-dir", type=str, default="rsna_preprocessed_t1")
    ap.add_argument("--model", type=str, default="cbam", choices=["cbam", "baseline"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    paths = {e: ckpt_dir / f"checkpoint_{args.prefix}_epoch_{e}.pth" for e in args.epochs}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n  " + "\n  ".join(missing))

    # --- identical split to the training run -------------------------------------
    dataset = RSNAPreprocessedDataset(data_dir=args.data_dir, split="t1")
    unique_patients = dataset.metadata["study_id"].unique()
    _, val_patients = train_test_split(
        unique_patients, test_size=args.val_split, random_state=args.seed
    )
    val_idx = dataset.metadata[dataset.metadata["study_id"].isin(val_patients)].index.tolist()
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"Val: {len(val_idx)} samples / {len(val_patients)} patients "
          f"(seed {args.seed}, split {args.val_split})\n", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)

    loaded = {}
    for e, p in paths.items():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        loaded[e] = ck["model_state_dict"]

    print("--- individual checkpoints (baseline to beat) ---", flush=True)
    for e in args.epochs:
        model.load_state_dict(loaded[e])
        loss, accs, _, _, pcm, _, _ = evaluate(model, val_loader, device)
        report(f"epoch {e}", pcm, accs, loss)

    # Suffix windows: averaging the tail of training, widening one checkpoint at a
    # time. IMWA reports early-epoch averaging helps most, but epoch 5 here is a much
    # weaker model, so let the data decide how far back to reach.
    print("\n--- weight averages ---", flush=True)
    order = sorted(args.epochs)
    for k in range(2, len(order) + 1):
        window = order[-k:]
        model.load_state_dict(average_state_dicts([loaded[e] for e in window]))
        loss, accs, _, _, pcm, _, _ = evaluate(model, val_loader, device)
        report("avg " + "+".join(str(e) for e in window), pcm, accs, loss)


if __name__ == "__main__":
    main()
