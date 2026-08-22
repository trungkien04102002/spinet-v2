#!/usr/bin/env python3
"""
Does temperature scaling fix the Hybrid's weighted log loss?

A paired test showed the Hybrid ranks Severe cases at least as well as CBAM
(severe AUC +0.010) yet scores clearly WORSE on the RSNA competition metric
(sample-weighted log loss +0.026, p<0.001). That combination -- good ranking, bad
proper score -- is the signature of miscalibration rather than a weaker model, and
the Hybrid's recipe is full of things that decalibrate: focal loss, Kendall
uncertainty weighting, and a learned cosine logit scale.

Temperature scaling (Guo et al., ICML 2017) divides the logits by a single scalar T,
which cannot change the argmax or any ranking-based metric (AUPRC, AUC stay fixed by
construction) but can move a proper score a great deal. The 3rd place RSNA solution
applied exactly this, T=0.91, to its spinal logits.

Fitting T on the same rows used to report the score would flatter it. This script
cross-fits by PATIENT: fit T on half the patients, score the other half, swap, and
average. Severe cases cluster within patient, so the split is over patients, not rows.

Usage:
    python3 experiments/f1_improvement/temperature_calibration.py \
        --npz experiments/f1_improvement/logits/hybrid_seed42_val.npz --label Hybrid
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_metrics import (COMPETITION_WEIGHTS, CONDITIONS,  # noqa: E402
                            sample_weighted_log_loss)


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def fit_temperature(logits, labels):
    """Minimise the competition's weighted log loss over a single scalar T."""
    def obj(logT):
        return sample_weighted_log_loss(softmax(logits / np.exp(logT)), labels)
    # Search log T so T stays positive; bounds cover T in ~[0.22, 4.5].
    res = minimize_scalar(obj, bounds=(-1.5, 1.5), method="bounded")
    return float(np.exp(res.x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    label = args.label or Path(args.npz).stem
    study_ids = data["study_id"]
    patients = np.unique(study_ids)

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(patients)
    halves = [set(shuffled[: len(shuffled) // 2]), set(shuffled[len(shuffled) // 2:])]

    print("=" * 74)
    print(f"Temperature scaling, cross-fitted over patient halves -- {label}")
    print("=" * 74)
    print(f"{'condition':<17}{'T (fold1/fold2)':>18}{'wLogLoss before':>18}{'after':>10}{'delta':>10}")
    print("-" * 74)

    before_all, after_all = [], []
    for cond in CONDITIONS:
        if f"logits_{cond}" not in data.files:
            continue
        logits = data[f"logits_{cond}"]
        labels = data[f"labels_{cond}"]
        valid = labels >= 0

        before = sample_weighted_log_loss(softmax(logits[valid]), labels[valid])

        scored, temps = [], []
        for fit_half, eval_half in ((0, 1), (1, 0)):
            fit_rows = valid & np.isin(study_ids, list(halves[fit_half]))
            eval_rows = valid & np.isin(study_ids, list(halves[eval_half]))
            if fit_rows.sum() == 0 or eval_rows.sum() == 0:
                continue
            T = fit_temperature(logits[fit_rows], labels[fit_rows])
            temps.append(T)
            # Weight each held-out half by its size so the average is over samples.
            scored.append((sample_weighted_log_loss(softmax(logits[eval_rows] / T),
                                                    labels[eval_rows]),
                           int(eval_rows.sum())))
        if not scored:
            continue
        after = sum(v * n for v, n in scored) / sum(n for _, n in scored)
        before_all.append(before)
        after_all.append(after)
        tstr = "/".join(f"{t:.3f}" for t in temps)
        print(f"{cond:<17}{tstr:>18}{before:>18.4f}{after:>10.4f}{after - before:>+10.4f}")

    if before_all:
        mb, ma = float(np.mean(before_all)), float(np.mean(after_all))
        print("-" * 74)
        print(f"{'MEAN':<17}{'':>18}{mb:>18.4f}{ma:>10.4f}{ma - mb:>+10.4f}")
        print("\nLower is better. Ranking metrics (AUPRC, AUC) are unchanged by")
        print("construction -- temperature scaling cannot reorder samples.")


if __name__ == "__main__":
    main()
