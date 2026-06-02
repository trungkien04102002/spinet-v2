#!/usr/bin/env python3
"""Aggregate RSNA ablation metrics across seeds into a mean +/- std table.

Reads the ``*_best_metrics.json`` files written by the training scripts for the
4 configurations x N seeds and prints (a) a human-readable table and (b) LaTeX
rows ready to paste into the paper's Table 1 (tab:main).

The metric formulas are verified to reproduce the seed-42 paper numbers:
  Mean Accuracy   = mean over conditions of val_accuracies
  Mean F1 macro   = mean over conditions of mean_class(per_class f1)
  Mean Recall     = mean over conditions of mean_class(per_class recall)
  Mean Precision  = mean over conditions of mean_class(per_class precision)
  Mean AUC/AUPRC  = auc_auprc_overall.macro_auc / macro_auprc
  Severe F1       = avg_severe_f1
  Severe AUC/AUPRC= auc_auprc_overall.severe_auc / severe_auprc

Usage:
  python3 scripts/aggregate_rsna_seeds.py                 # seeds 42 123 456
  python3 scripts/aggregate_rsna_seeds.py --seeds 42 123 456 --base-dir checkpoints/v3_20260503
"""
import argparse
import json
from pathlib import Path

import numpy as np

CONDS = ["spinal_canal", "left_foraminal", "right_foraminal"]

# config -> (sub-directory, function(seed) -> file prefix)
def _tag(seed):
    return "" if seed == 42 else f"_seed{seed}"

CONFIGS = {
    "SpineNetV2": ("baseline", lambda s: f"baseline{_tag(s)}"),
    "CBAM-only":  ("cbam",     lambda s: f"attention{_tag(s)}"),
    "BMC-only":   ("bmc_only", lambda s: (
        "hybrid_biomedclip_only" if s == 42 else f"hybrid_seed{s}_biomedclip_only")),
    "Hybrid":     ("hybrid",   lambda s: f"hybrid{_tag(s)}"),
}

METRIC_ORDER = [
    ("Mean Accuracy",        "pct"),
    ("Mean F1 macro",        "f3"),
    ("Mean Recall macro",    "f3"),
    ("Mean Precision macro", "f3"),
    ("Mean AUC macro",       "f3"),
    ("Mean AUPRC macro",     "f3"),
    ("Severe F1",            "f3"),
    ("Severe AUC",           "f3"),
    ("Severe AUPRC",         "f3"),
]


def metrics_from_json(d):
    acc = np.mean([d["val_accuracies"][c] for c in CONDS]) * 100.0
    pcm = d["per_class_metrics"]
    f1 = np.mean([np.mean(pcm[c]["f1"]) for c in CONDS])
    rec = np.mean([np.mean(pcm[c]["recall"]) for c in CONDS])
    prec = np.mean([np.mean(pcm[c]["precision"]) for c in CONDS])
    o = d["auc_auprc_overall"]
    return {
        "Mean Accuracy": acc,
        "Mean F1 macro": f1,
        "Mean Recall macro": rec,
        "Mean Precision macro": prec,
        "Mean AUC macro": o["macro_auc"],
        "Mean AUPRC macro": o["macro_auprc"],
        "Severe F1": d["avg_severe_f1"],
        "Severe AUC": o["severe_auc"],
        "Severe AUPRC": o["severe_auprc"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="checkpoints/v3_20260503")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = ap.parse_args()

    base = Path(args.base_dir)
    # collect: config -> metric -> list over seeds
    collected = {cfg: {m: [] for m, _ in METRIC_ORDER} for cfg in CONFIGS}
    missing = []
    for cfg, (subdir, prefix_fn) in CONFIGS.items():
        for s in args.seeds:
            fp = base / subdir / f"{prefix_fn(s)}_best_metrics.json"
            if not fp.exists():
                missing.append(str(fp))
                continue
            vals = metrics_from_json(json.load(open(fp)))
            for m in collected[cfg]:
                collected[cfg][m].append(vals[m])

    if missing:
        print("WARNING: missing metric files (skipped):")
        for m in missing:
            print("  -", m)
        print()

    def agg(xs):
        a = np.asarray(xs, float)
        if a.size == 0:
            return None
        sd = a.std(ddof=1) if a.size > 1 else 0.0
        return a.mean(), sd, a.size

    # ---- human-readable table ----
    cfgs = list(CONFIGS.keys())
    print(f"RSNA ablation over seeds {args.seeds}  (mean +/- std, n per cell shown)")
    print("=" * 92)
    print(f"{'Metric':<22}" + "".join(f"{c:>17}" for c in cfgs))
    print("-" * 92)
    for m, fmt in METRIC_ORDER:
        cells = []
        for c in cfgs:
            r = agg(collected[c][m])
            if r is None:
                cells.append(f"{'--':>17}"); continue
            mean, sd, n = r
            if fmt == "pct":
                cells.append(f"{mean:6.2f}+/-{sd:4.2f}(n{n})".rjust(17))
            else:
                cells.append(f"{mean:.3f}+/-{sd:.3f}(n{n})".rjust(17))
        print(f"{m:<22}" + "".join(cells))
    print("=" * 92)

    # ---- LaTeX rows (tab:main format) ----
    print("\nLaTeX rows (paste into tab:main; bold the winner manually):\n")
    for m, fmt in METRIC_ORDER:
        parts = []
        for c in cfgs:
            r = agg(collected[c][m])
            if r is None:
                parts.append("--"); continue
            mean, sd, n = r
            if fmt == "pct":
                parts.append(f"{mean:.1f} $\\pm$ {sd:.1f}")
            else:
                parts.append(f"{mean:.3f} $\\pm$ {sd:.3f}")
        print(f"{m:<20} & " + " & ".join(parts) + r" \\")


if __name__ == "__main__":
    main()
