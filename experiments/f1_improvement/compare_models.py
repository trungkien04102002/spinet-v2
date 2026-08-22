#!/usr/bin/env python3
"""
Paired comparison of two models evaluated on the SAME validation split.

Comparing two independent confidence intervals is the wrong test here: both models saw
the same patients, so most of the sampling error is shared and cancels. Resampling
patients once per draw and computing the DIFFERENCE on that same resample is far more
sensitive -- for a fixed val set the paired interval is several times narrower than
either model's own interval.

Bootstrap resamples patients (not rows): Severe findings cluster within a patient
across adjacent levels and bilaterally, so a row-level bootstrap gives intervals that
are too narrow.

Reports, for every metric, the mean paired difference (b - a), a patient-clustered 95%
interval, and a two-sided bootstrap p-value for "no difference".

Usage:
    python3 experiments/f1_improvement/compare_models.py \
        --a experiments/f1_improvement/logits/cbam_seed42_val.npz  --a-label CBAM \
        --b experiments/f1_improvement/logits/hybrid_seed42_val.npz --b-label Hybrid
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robust_metrics import CONDITIONS, PRIMARY, SECONDARY, evaluate  # noqa: E402

# Metrics where a LOWER value is better; the printed verdict flips for these.
LOWER_IS_BETTER = {"weighted_logloss"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    da = np.load(args.a, allow_pickle=True)
    db = np.load(args.b, allow_pickle=True)

    # The paired test is only valid if both dumps describe the same rows in the same
    # order. Verify rather than assume -- a silent misalignment would invent a result.
    if not np.array_equal(da["study_id"], db["study_id"]):
        raise SystemExit("Refusing to compare: the two dumps do not share the same "
                         "rows in the same order (study_id arrays differ).")
    for cond in args.conditions:
        key = f"labels_{cond}"
        if key in da.files and key in db.files and not np.array_equal(da[key], db[key]):
            raise SystemExit(f"Refusing to compare: {key} differs between dumps.")

    conds = [c for c in args.conditions if f"probs_{c}" in da.files and f"probs_{c}" in db.files]
    _, mean_a = evaluate(da, conds)
    _, mean_b = evaluate(db, conds)

    rng = np.random.default_rng(args.seed)
    study_ids = da["study_id"]
    patients = np.unique(study_ids)
    rows_by_patient = {p: np.flatnonzero(study_ids == p) for p in patients}

    diffs = {}
    for _ in range(args.n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in picked])
        try:
            _, ma = evaluate(da, conds, row_mask=idx)
            _, mb = evaluate(db, conds, row_mask=idx)
        except ValueError:
            continue
        for k in ma:
            if k in mb:
                diffs.setdefault(k, []).append(mb[k] - ma[k])
    diffs = {k: np.array(v) for k, v in diffs.items()}

    print("=" * 88)
    print(f"PAIRED:  B = {args.b_label}   minus   A = {args.a_label}")
    print(f"{len(patients)} patients, {len(study_ids)} rows, {args.n_boot} patient-clustered draws")
    print("=" * 88)
    hdr = (f"{'metric':<20}{args.a_label[:11]:>12}{args.b_label[:11]:>12}"
           f"{'diff':>10}{'95% CI of diff':>22}{'p':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))

    for key in PRIMARY + SECONDARY:
        if key not in diffs:
            continue
        d = diffs[key]
        lo, hi = np.percentile(d, [2.5, 97.5])
        # Two-sided bootstrap p: how often the difference crosses zero.
        p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
        p = min(1.0, float(p))
        better = args.b_label if (np.mean(d) > 0) != (key in LOWER_IS_BETTER) else args.a_label
        verdict = f"{better} better" if (lo > 0 or hi < 0) else "no difference"
        star = " *" if key in PRIMARY else "  "
        print(f"{key:<20}{mean_a[key]:>12.4f}{mean_b[key]:>12.4f}{np.mean(d):>+10.4f}"
              f"   [{lo:>+7.4f}, {hi:>+7.4f}]{p:>8.3f}  {verdict}{star}")

    print("\n* = primary metric. 'no difference' means the 95% interval of the paired")
    print("  difference contains zero, i.e. this validation set cannot separate them.")


if __name__ == "__main__":
    main()
