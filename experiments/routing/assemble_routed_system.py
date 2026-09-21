#!/usr/bin/env python3
"""Join the T2 canal branch and the T1 foraminal branch into one system.

The thesis claims foraminal narrowing should be graded on Sagittal T1 and canal
stenosis on Sagittal T2, and trains a branch for each. Neither branch alone is
the proposed system: the system is the two of them routed by condition. This
assembles that, with no retraining, by joining stored predictions on
(study, level).

TWO THINGS MAKE THIS HARDER THAN A MERGE, AND BOTH ARE HANDLED HERE.

1. The two runs do not share a validation set. Both used mode="random" with
   seed 42, but over different patient lists (the T1 crop set covers 1972
   studies, the T2 set 1974), and sklearn's split is a property of the list,
   not of the patient. Measured: 214 of 395 validation patients in common, so
   181 patients sit in one run's validation and the other's TRAINING. Scoring
   the routed system on those would read a model's own training data as though
   it were held out. So everything below is restricted to the 214 shared
   patients, and the script refuses to run if that set is empty.

   This is why spinenet/patient_split.py grew a "hash" mode. A re-run under
   that mode would restore the full validation set; until then, this is the
   largest honestly usable subset.

2. The T1 crops are per side: one row for the left foramen and one for the
   right, each carrying a label for its own side only. The T2 rows carry all
   three labels. So the T1 side is pivoted back to one row per (study, level)
   before the join.

WHAT IS AND IS NOT COMPARABLE. The routed-versus-T2-only comparison below is
paired: same patients, same rows, same seed, differing only in where the
foraminal prediction comes from. That is the number this script exists for.
The absolute values are NOT comparable to the Phase 2 table, which is three
seeds over a different and larger validation set.

Usage:
    python3 experiments/routing/assemble_routed_system.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from spinenet.patient_split import split_patients  # noqa: E402

CONDS = ("spinal_canal", "left_foraminal", "right_foraminal")
FORAMINAL = ("left_foraminal", "right_foraminal")


def load_branch(npz, meta_csv, expect_conds):
    """Stored predictions plus the (study, level) key they belong to.

    The dumps carry study_id but not level, so the level column is recovered by
    rebuilding the same split over the same metadata and checking that the
    resulting row order matches the dump's study_id column exactly. If it does
    not, the recovered levels would be silently wrong, so this raises.
    """
    d = np.load(npz, allow_pickle=True)
    md = pd.read_csv(meta_csv)
    _, val = split_patients(md.study_id.unique(), 0.2, 42, "random")
    v = md[md.study_id.isin(val)].reset_index(drop=True)

    n = d["probs_" + expect_conds[0]].shape[0]
    if len(v) != n:
        raise SystemExit(f"{npz.name}: metadata gives {len(v)} rows, dump has {n}")
    if not (v.study_id.to_numpy() == d["study_id"]).all():
        raise SystemExit(f"{npz.name}: study order differs from the dump")
    return d, v


def severe_metrics(y, p):
    """Severe-class numbers on one condition, ignoring unlabelled rows."""
    keep = y != -1
    y, p = y[keep], p[keep]
    if len(y) == 0 or (y == 2).sum() == 0:
        return {}
    yhat = p.argmax(1)
    bin_y = (y == 2).astype(int)
    tp = int(((yhat == 2) & (y == 2)).sum())
    fp = int(((yhat == 2) & (y != 2)).sum())
    fn = int(((yhat != 2) & (y == 2)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(y)), "n_severe": int(bin_y.sum()),
        "severe_precision": prec, "severe_recall": rec,
        "severe_f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
        "severe_auc": float(roc_auc_score(bin_y, p[:, 2])),
        "severe_auprc": float(average_precision_score(bin_y, p[:, 2])),
        # Macro one-vs-rest, which is what the Phase 2 and Phase 3 report
        # tables mean by "AUPRC" and "AUC". Reporting the Severe-only figure
        # under the same name would compare two different quantities.
        "macro_auprc": float(average_precision_score(
            label_binarize(y, classes=[0, 1, 2]), p, average="macro")),
        "macro_auc": float(roc_auc_score(
            label_binarize(y, classes=[0, 1, 2]), p, average="macro", multi_class="ovr")),
        "macro_f1": float(f1_score(y, yhat, average="macro", labels=[0, 1, 2],
                                   zero_division=0)),
        "accuracy": float((yhat == y).mean()),
    }


def system_scores(labels, probs):
    """Per-condition metrics plus the macro figures the Phase 2 table reports."""
    per = {c: severe_metrics(labels[c], probs[c]) for c in CONDS}
    ok = [c for c in CONDS if per[c]]
    return {
        "per_condition": per,
        "mean_f1_macro": float(np.mean([per[c]["macro_f1"] for c in ok])),
        "mean_accuracy": float(np.mean([per[c]["accuracy"] for c in ok])),
        "mean_severe_f1": float(np.mean([per[c]["severe_f1"] for c in ok])),
        "mean_severe_recall": float(np.mean([per[c]["severe_recall"] for c in ok])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t2-npz", type=Path,
                    default=REPO / "experiments/f1_improvement/logits/runc_final_val.npz")
    ap.add_argument("--t1-npz", type=Path,
                    default=REPO / "experiments/f1_improvement/logits/t1_final_val.npz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "routed_system.json")
    args = ap.parse_args()

    t2, v2 = load_branch(args.t2_npz, REPO / "rsna_preprocessed/train_metadata.csv", CONDS)
    t1, v1 = load_branch(args.t1_npz, REPO / "rsna_preprocessed_t1/t1_metadata.csv", FORAMINAL)

    shared = sorted(set(v2.study_id) & set(v1.study_id))
    if not shared:
        raise SystemExit("the two runs share no validation patients; nothing can be joined")
    print(f"validation patients: T2 {v2.study_id.nunique()}, T1 {v1.study_id.nunique()}, "
          f"shared {len(shared)}")
    print(f"  {v2.study_id.nunique() - len(shared)} patients sit in one run's validation "
          f"and the other's training, and are excluded")

    keep2 = v2.study_id.isin(shared).to_numpy()
    base = v2[keep2].reset_index(drop=True)
    labels = {c: t2["labels_" + c][keep2] for c in CONDS}
    probs_t2 = {c: t2["probs_" + c][keep2] for c in CONDS}

    # T1 rows are per side; pivot to one row per (study, level) per side.
    v1 = v1.copy()
    v1["side"] = np.where(v1.condition.str.contains("left"), "left", "right")
    v1["row"] = np.arange(len(v1))
    t1_probs = {c: t1["probs_" + c] for c in FORAMINAL}

    probs_routed = dict(probs_t2)
    coverage = {}
    for c, side in zip(FORAMINAL, ("left", "right")):
        side_rows = v1[(v1.side == side) & v1.study_id.isin(shared)]
        lut = {(s, l): r for s, l, r in
               zip(side_rows.study_id, side_rows.level, side_rows.row)}
        out = probs_t2[c].copy()
        hit = 0
        for i, (s, l) in enumerate(zip(base.study_id, base.level)):
            r = lut.get((s, l))
            if r is not None:
                out[i] = t1_probs[c][r]
                hit += 1
        probs_routed[c] = out
        coverage[c] = hit / len(base)
        print(f"  {c}: T1 prediction available for {hit}/{len(base)} rows "
              f"({100*coverage[c]:.1f}%); the rest keep the T2 prediction")

    res = {"n_shared_patients": len(shared), "n_rows": int(len(base)),
           "t1_coverage": coverage,
           "t2_only": system_scores(labels, probs_t2),
           "routed": system_scores(labels, probs_routed)}

    # Paired bootstrap over patients: both systems are scored on the same
    # resample, so the shared sampling noise cancels.
    pats = base.study_id.to_numpy()
    uniq = np.unique(pats)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(args.n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(pats == p) for p in pick])
        a = system_scores({c: labels[c][idx] for c in CONDS},
                          {c: probs_t2[c][idx] for c in CONDS})["mean_f1_macro"]
        b = system_scores({c: labels[c][idx] for c in CONDS},
                          {c: probs_routed[c][idx] for c in CONDS})["mean_f1_macro"]
        diffs.append(b - a)
    diffs = np.array(diffs)
    res["mean_f1_macro_gain"] = float(res["routed"]["mean_f1_macro"]
                                      - res["t2_only"]["mean_f1_macro"])
    res["gain_ci95"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    res["gain_p_two_sided"] = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))

    print(f"\n{'metric':22s}{'T2 only':>10s}{'routed':>10s}{'delta':>10s}")
    for k in ("mean_f1_macro", "mean_accuracy", "mean_severe_f1", "mean_severe_recall"):
        a, b = res["t2_only"][k], res["routed"][k]
        print(f"{k:22s}{a:10.4f}{b:10.4f}{b-a:+10.4f}")
    print(f"\nMean F1 macro gain {res['mean_f1_macro_gain']:+.4f}  "
          f"95% CI [{res['gain_ci95'][0]:+.4f}, {res['gain_ci95'][1]:+.4f}]  "
          f"p={res['gain_p_two_sided']:.3f}")

    print(f"\n{'condition':18s}{'Severe F1':>12s}{'Severe Rec':>12s}{'Severe Prec':>13s}")
    for c in CONDS:
        a, b = res["t2_only"]["per_condition"][c], res["routed"]["per_condition"][c]
        print(f"{c:18s}{a['severe_f1']:6.3f}->{b['severe_f1']:5.3f}"
              f"{a['severe_recall']:6.3f}->{b['severe_recall']:5.3f}"
              f"{a['severe_precision']:7.3f}->{b['severe_precision']:5.3f}"
              f"   (n_sev {a['n_severe']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")
    print("\nThese absolute values are NOT comparable to the Phase 2 table: that is "
          "three seeds over the full validation set, this is one seed over the 214 "
          "patients both branches held out. The routed-versus-T2 delta is the "
          "comparable quantity, because it is paired on identical rows.")


if __name__ == "__main__":
    main()
