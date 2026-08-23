#!/usr/bin/env python3
"""
Report RSNA grading performance with metrics that this validation set can actually
resolve, instead of per-class F1.

Why this exists
---------------
The validation split holds ~80 Severe cases per foraminal condition. A Monte-Carlo
check at that size puts SD(Severe F1) at roughly 0.045-0.050 and the 95% interval at
about +/-0.09, so the 0.02-0.03 differences chased throughout Phase 3 sit at 0.4-0.6
standard errors. Worse, the epoch-to-epoch swing in F1 (SD ~0.007-0.010 over the last
five epochs) is almost entirely threshold-selection noise: the same runs measured by
Severe AUPRC swing by SD ~0.0006, an order of magnitude less.

So this script leads with threshold-free and ordinal-aware measures:

  * Severe AUPRC (one-vs-rest)  -- threshold-free, uses the ranking over all samples.
    Preferred over ROC-AUC under heavy imbalance (Saito & Rehmsmeier, PLOS ONE 2015).
  * Quadratic-weighted kappa    -- the measure the clinical literature reports, so it
    is directly comparable to published inter-reader agreement (Lurie et al., Spine
    2008: canal 0.73, foraminal 0.58).
  * Sample-weighted log loss    -- the actual RSNA 2024 competition metric, with
    per-sample weights 1 / 2 / 4 for Normal-Mild / Moderate / Severe. Comparable with
    published leaderboard work.
  * Per-class F1                -- kept, but demoted to a secondary column.

Confidence intervals are bootstrapped by resampling *patients*, not rows. Severe
foraminal findings cluster within a patient (adjacent levels, and bilaterally), so a
row-level bootstrap understates the interval.

Input is the .npz written by dump_logits.py, which must contain probs_<cond>,
labels_<cond> and study_id.

Usage:
    python3 experiments/f1_improvement/robust_metrics.py \
        --npz experiments/f1_improvement/logits/hybrid_seed42_val.npz \
        --label "Hybrid published seed42"
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

CONDITIONS = ("spinal_canal", "left_foraminal", "right_foraminal")
CLASS_NAMES = ("Normal/Mild", "Moderate", "Severe")
# RSNA 2024 competition sample weights, indexed by true class.
COMPETITION_WEIGHTS = np.array([1.0, 2.0, 4.0])


def sample_weighted_log_loss(probs, labels):
    """The RSNA 2024 metric: mean log loss with per-sample weights 1/2/4 by true class.

    Weights are applied per sample according to its ground-truth class, which is what
    the competition did -- not as a static per-class multiplier inside the loss.
    """
    eps = 1e-15
    p = np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)
    w = COMPETITION_WEIGHTS[labels]
    return float(np.sum(-w * np.log(p)) / np.sum(w))


def f1_for_class(preds, labels, cls):
    tp = int(np.sum((preds == cls) & (labels == cls)))
    fp = int(np.sum((preds == cls) & (labels != cls)))
    fn = int(np.sum((preds != cls) & (labels == cls)))
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def metrics_for_condition(probs, labels):
    """All reported measures for one condition, on rows with a valid (non -1) label."""
    valid = labels >= 0
    probs, labels = probs[valid], labels[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None

    preds = probs.argmax(axis=1)
    severe_present = (labels == 2).any()

    out = {
        "n": int(len(labels)),
        "n_severe": int(np.sum(labels == 2)),
        "severe_auprc": float(average_precision_score((labels == 2).astype(int), probs[:, 2]))
        if severe_present else float("nan"),
        "severe_auc": float(roc_auc_score((labels == 2).astype(int), probs[:, 2]))
        if severe_present else float("nan"),
        "macro_auprc": float(np.mean([
            average_precision_score((labels == c).astype(int), probs[:, c])
            for c in range(3) if (labels == c).any()
        ])),
        # Quadratic weights make an adjacent-grade error cost 1/4 of a two-grade error,
        # which is the clinically meaningful cost structure for a severity scale.
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "weighted_logloss": sample_weighted_log_loss(probs, labels),
        "severe_f1": f1_for_class(preds, labels, 2),
        "severe_recall": float(np.sum((preds == 2) & (labels == 2)) / max(1, np.sum(labels == 2))),
        "severe_precision": float(np.sum((preds == 2) & (labels == 2)) / max(1, np.sum(preds == 2))),
    }
    return out


def evaluate(data, cond_subset=CONDITIONS, row_mask=None, probs_override=None):
    """Per-condition metrics plus the macro average across conditions.

    `probs_override` swaps in externally computed probabilities (e.g. temperature
    calibrated ones) while keeping the same labels and row selection.
    """
    per_cond, collected = {}, {}
    for cond in cond_subset:
        if probs_override is not None and cond in probs_override:
            probs = probs_override[cond]
        else:
            probs = data[f"probs_{cond}"]
        labels = data[f"labels_{cond}"]
        if row_mask is not None:
            probs, labels = probs[row_mask], labels[row_mask]
        m = metrics_for_condition(probs, labels)
        if m is None:
            continue
        per_cond[cond] = m
        for k, v in m.items():
            if k not in ("n", "n_severe"):
                collected.setdefault(k, []).append(v)
    mean = {k: float(np.nanmean(v)) for k, v in collected.items()}
    return per_cond, mean


def clustered_bootstrap(data, cond_subset, n_boot, seed):
    """Resample PATIENTS with replacement; return the bootstrap distribution of the
    condition-averaged metrics."""
    rng = np.random.default_rng(seed)
    study_ids = data["study_id"]
    patients = np.unique(study_ids)
    # Row indices grouped by patient, so a resampled patient contributes all its rows.
    rows_by_patient = {p: np.flatnonzero(study_ids == p) for p in patients}

    draws = {}
    for _ in range(n_boot):
        picked = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([rows_by_patient[p] for p in picked])
        try:
            _, mean = evaluate(data, cond_subset, row_mask=idx)
        except ValueError:
            continue
        for k, v in mean.items():
            draws.setdefault(k, []).append(v)
    return {k: np.array(v) for k, v in draws.items()}


PRIMARY = ("severe_auprc", "macro_auprc", "qwk", "weighted_logloss")
SECONDARY = ("severe_f1", "severe_precision", "severe_recall", "severe_auc")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="dump_logits.py output")
    ap.add_argument("--label", default=None, help="name for this model in the report")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--calibrate", action="store_true",
                    help="Also report temperature-scaled numbers. The temperature is "
                         "cross-fitted over patient halves, so no row is scored by a "
                         "temperature fitted on its own patient. Proper scores move a "
                         "lot; argmax-based metrics (F1, QWK) cannot move at all; "
                         "ranking metrics move slightly -- see the note in "
                         "temperature_calibration.py.")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    label = args.label or Path(args.npz).stem
    conds = [c for c in args.conditions if f"probs_{c}" in data.files]

    per_cond, mean = evaluate(data, conds)

    print("=" * 78)
    print(f"{label}")
    print(f"  {len(np.unique(data['study_id']))} patients, {len(data['study_id'])} rows")
    print("=" * 78)

    print("\nPER CONDITION")
    hdr = f"{'condition':<17}{'sevAUPRC':>10}{'macAUPRC':>10}{'QWK':>8}{'wLogLoss':>10}{'sevF1':>8}{'n_sev':>7}"
    print(hdr)
    print("-" * len(hdr))
    for cond, m in per_cond.items():
        print(f"{cond:<17}{m['severe_auprc']:>10.4f}{m['macro_auprc']:>10.4f}"
              f"{m['qwk']:>8.4f}{m['weighted_logloss']:>10.4f}{m['severe_f1']:>8.4f}{m['n_severe']:>7d}")

    print(f"\nBootstrapping {args.n_boot} draws, resampled by patient...")
    draws = clustered_bootstrap(data, conds, args.n_boot, args.seed)

    def ci(key):
        d = draws.get(key)
        if d is None or len(d) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    print("\nMEAN ACROSS CONDITIONS, with patient-clustered 95% CI")
    print(f"{'metric':<20}{'value':>10}{'95% CI':>22}{'CI width':>10}{'boot SD':>10}")
    print("-" * 72)
    report = {}
    for key in PRIMARY + SECONDARY:
        if key not in mean:
            continue
        lo, hi = ci(key)
        sd = float(np.std(draws[key])) if key in draws else float("nan")
        marker = "  <- primary" if key in PRIMARY else ""
        print(f"{key:<20}{mean[key]:>10.4f}   [{lo:>7.4f}, {hi:>7.4f}]{hi - lo:>10.4f}{sd:>10.4f}{marker}")
        report[key] = {"value": mean[key], "ci_low": lo, "ci_high": hi, "boot_sd": sd}

    if "severe_f1" in report and "severe_auprc" in report:
        f1w = report["severe_f1"]["ci_high"] - report["severe_f1"]["ci_low"]
        apw = report["severe_auprc"]["ci_high"] - report["severe_auprc"]["ci_low"]
        print(f"\nSevere F1 interval is {f1w / apw:.1f}x wider than Severe AUPRC's.")

    if args.calibrate:
        from temperature_calibration import crossfit_calibrated_probs
        cal_probs, temps = crossfit_calibrated_probs(data, conds, seed=args.seed)
        _, cal_mean = evaluate(data, conds, probs_override=cal_probs)
        print("\nAFTER TEMPERATURE SCALING (cross-fitted over patient halves)")
        print(f"{'metric':<20}{'raw':>10}{'calibrated':>13}{'change':>10}")
        print("-" * 53)
        for key in PRIMARY + SECONDARY:
            if key in mean and key in cal_mean:
                delta = cal_mean[key] - mean[key]
                print(f"{key:<20}{mean[key]:>10.4f}{cal_mean[key]:>13.4f}{delta:>+10.4f}")
        tstr = "  ".join(f"{c}: " + "/".join(f"{t:.3f}" for t in v)
                         for c, v in temps.items())
        print(f"\nT per condition (one per fold) -- {tstr}")
        print("T below 1 means the head was under-confident and the logits needed "
              "sharpening.")
        print("F1/QWK are unchanged because T cannot move an argmax. AUPRC/AUC shift "
              "slightly:\nwith a multi-class softmax the denominator differs per "
              "sample, so T does reorder samples.")
        report["calibrated"] = cal_mean
        report["temperatures"] = temps

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(
            {"label": label, "per_condition": per_cond, "mean": mean,
             "bootstrap": report, "n_boot": args.n_boot}, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
