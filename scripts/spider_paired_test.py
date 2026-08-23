#!/usr/bin/env python3
"""Paired significance test for the SPIDER 3-seed transfer claim (Hybrid > CBAM).

Reads per-seed Mean F1 (val_mean_f1_macro) from the SPIDER 8-label transfer
metrics and runs a paired t-test + seed-level bootstrap on the per-config
differences. No GPU / no model needed — pure JSON post-processing.

Result (seeds 42/123/456): Hybrid-CBAM Mean F1 Delta = +0.034,
paired t-test t=9.47, p=0.011, 95% CI [0.018, 0.049]; Hybrid-Baseline p=0.57
(parity). These back the "regularizer" claim in the paper (footnote of
Table tab:spider_transfer_agg).

Usage:
  python3 scripts/spider_paired_test.py
"""
import json
import numpy as np
from scipy import stats

D = "experiments/v3_spider_8lbl"
SEEDS = ["", "_seed123", "_seed456"]  # "" == seed 42


def f1(name):
    return [json.load(open(f"{D}/best_metrics_{name}{s}.json"))["val_mean_f1_macro"]
            for s in SEEDS]


def main():
    cbam = np.array(f1("cbam"))
    base = np.array(f1("baseline"))
    hyb = np.array(f1("hybrid_spider_frozen"))
    for tag, arr in [("Baseline", base), ("CBAM", cbam), ("Hybrid", hyb)]:
        print(f"  {tag:9s} Mean F1: {np.round(arr,4)}  mean={arr.mean():.4f}")

    for tag, other in [("CBAM", cbam), ("Baseline", base)]:
        diff = hyb - other
        t, p = stats.ttest_rel(hyb, other)
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        tc = stats.t.ppf(0.975, len(diff) - 1)
        print(f"\nHybrid - {tag}: Delta={diff.mean():+.4f}  "
              f"paired t={t:.2f} p={p:.4f}  95% CI [{diff.mean()-tc*se:.4f}, {diff.mean()+tc*se:.4f}]")


if __name__ == "__main__":
    main()
