#!/usr/bin/env python3
"""Does the crop geometry actually explain the foraminal deficit?

The routing argument currently rests on two facts that sit side by side without
being connected:

  geometry:  32.2% (left) and 47.2% (right) of foraminal annotations fall
             outside the 9-slice depth of the T2 crop (geometry_evidence_mm.py)
  results:   the T1 specialist beats the T2 model on foraminal AUPRC by
             +0.054 and +0.039

A committee can reasonably ask why the second follows from the first, rather
than the T1 crops simply being different images. This script tests the link with
a prediction that can fail.

If the T2 crop genuinely cannot see the foramen in those cases, then the T2
model should perform WORSE on exactly them. That is a within-model comparison:
one model, one validation set, split by a property of the data that the model
knows nothing about. No cross-model comparison, so the patient-split defect
between the T2 and T1 metadata does not enter, and no saliency method is
involved either.

Outcomes and what each would mean:
  gap present   the geometry explains the deficit; the routing fix addresses a
                cause rather than a correlate
  no gap        the geometry does not explain the deficit. The 32-47% figure
                would still be true but would stop supporting the routing
                claim, and the thesis would have to say so.

Usage:
    python3 experiments/explainability/geometry_vs_performance.py \\
        --npz experiments/f1_improvement/logits/runc_final_val.npz
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from spinenet.patient_split import split_patients  # noqa: E402
from geometry_evidence_mm import slice_geometry, to_patient_mm  # noqa: E402

RAW = REPO_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"


def norm_level(s):
    """'L1/L2' (coordinate file) and 'l1_l2' (crop metadata) name the same level.

    Joining on the raw strings silently yields zero rows, which reads like "no
    effect" rather than like a bug, so both sides go through this.
    """
    return (s.astype(str).str.lower()
            .str.replace("/", "_", regex=False)
            .str.replace("-", "_", regex=False)
            .str.strip())


def build_geometry_table(study_ids, verbose=True):
    """Per (study, level, side): is the foraminal annotation beyond +/-4 slices
    of the T2 crop centre, measured in patient space?"""
    coords = pd.read_csv(RAW / "train_label_coordinates.csv")
    desc = pd.read_csv(RAW / "train_series_descriptions.csv")
    c = coords.merge(desc, on=["study_id", "series_id"], how="left")
    c["cond"] = c["condition"].str.lower()
    c["level"] = norm_level(c["level"])
    c = c[c.study_id.isin(study_ids)]

    canal = c[c["cond"].str.contains("canal")]
    foram = c[c["cond"].str.contains("neural")]

    cache, recs = {}, []
    for i, sid in enumerate(sorted(set(canal.study_id) & set(foram.study_id))):
        if verbose and i % 50 == 0:
            print(f"  ...{i} studies", flush=True)
        ca, fo = canal[canal.study_id == sid], foram[foram.study_id == sid]
        for level in set(ca.level) & set(fo.level):
            crow = ca[ca.level == level].iloc[0]
            key = (sid, crow.series_id, crow.instance_number)
            if key not in cache:
                cache[key] = slice_geometry(*key)
            gc = cache[key]
            if gc is None or not gc["slice_gap"]:
                continue
            canal_mm = to_patient_mm(crow.x, crow.y, gc)
            normal = np.cross(gc["row_dir"], gc["col_dir"])
            half_slice_mm = 4 * gc["slice_gap"]
            for _, frow in fo[fo.level == level].iterrows():
                fkey = (sid, frow.series_id, frow.instance_number)
                if fkey not in cache:
                    cache[fkey] = slice_geometry(*fkey)
                gf = cache[fkey]
                if gf is None:
                    continue
                d_norm = float(np.dot(to_patient_mm(frow.x, frow.y, gf) - canal_mm,
                                      normal))
                recs.append({
                    "study_id": int(sid), "level": level,
                    "side": "left" if "left" in frow["cond"] else "right",
                    "d_normal_mm": abs(d_norm),
                    "d_normal_slices": abs(d_norm) / gc["slice_gap"],
                    "outside_slice_window": abs(d_norm) > half_slice_mm,
                })
    return pd.DataFrame(recs)


def scores(y, p_severe):
    """Severe-class discrimination on one subset.

    AUC is the primary number here and AUPRC is not comparable across these
    subsets, which is a trap this script originally fell into. AUPRC has a floor
    equal to the positive prevalence, and the two subsets have very different
    prevalence (right side: 2.9% inside against 5.8% outside). The subset with
    more positives therefore scores higher AUPRC mechanically, regardless of the
    model. AUC is invariant to prevalence, so it answers the question actually
    being asked: does the model rank Severe cases better in one subset than the
    other?

    auprc_lift = AUPRC / prevalence is reported as well, since a raw AUPRC is
    still worth seeing and dividing by the floor makes it roughly comparable.
    """
    y = np.asarray(y)
    keep = y != -1
    y, p_severe = y[keep], p_severe[keep]
    n_sev = int((y == 2).sum())
    if len(y) == 0 or n_sev == 0 or n_sev == len(y):
        return dict(auc=float("nan"), auprc=float("nan"),
                    auprc_lift=float("nan"), prevalence=float("nan"),
                    n_severe=n_sev, n=int(len(y)))
    bin_y = (y == 2).astype(int)
    prev = float(bin_y.mean())
    ap_ = float(average_precision_score(bin_y, p_severe))
    return dict(auc=float(roc_auc_score(bin_y, p_severe)), auprc=ap_,
                auprc_lift=ap_ / prev, prevalence=prev,
                n_severe=n_sev, n=int(len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path,
                    default=REPO_ROOT / "experiments/f1_improvement/logits/runc_final_val.npz")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "rsna_preprocessed")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--split-mode", type=str, default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path,
                    default=_HERE / "results" / "geometry_vs_performance.json")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(d["meta_json"]))
    print(f"dump: {args.npz.name}  split={meta.get('split')}  "
          f"conditions={[k[6:] for k in d.files if k.startswith('probs_')]}")

    md = pd.read_csv(args.data_dir / f"{args.split}_metadata.csv")
    _, val_patients = split_patients(md.study_id.unique(), args.val_split,
                                     args.seed, args.split_mode)
    v = md[md.study_id.isin(val_patients)].reset_index(drop=True)
    n_rows = d["probs_left_foraminal"].shape[0]
    if len(v) != n_rows:
        raise SystemExit(f"row mismatch: metadata {len(v)} vs npz {n_rows}. "
                         f"--split-mode / --seed must match the dump.")
    if not (v.study_id.to_numpy() == d["study_id"]).all():
        raise SystemExit("study_id order differs between metadata and dump")

    print(f"\nreading DICOM geometry for {v.study_id.nunique()} val studies...")
    geo = build_geometry_table(set(v.study_id))
    if geo.empty:
        raise SystemExit("no geometry rows; is the raw DICOM download present?")
    print(f"  {len(geo)} (study, level, side) geometry rows")

    out = {"dump": args.npz.name, "conditions": {}}
    rng = np.random.default_rng(0)

    for side in ("left", "right"):
        cond = f"{side}_foraminal"
        g = geo[geo.side == side][["study_id", "level", "outside_slice_window",
                                   "d_normal_slices"]]
        # The T2 dump has one row per (study, level) carrying both sides' labels,
        # so joining on (study, level) attaches this side's geometry to it.
        j = v[["study_id", "level"]].copy()
        j["level"] = norm_level(j["level"])
        j["row"] = np.arange(len(j))
        before = len(j)
        j = j.merge(g, on=["study_id", "level"], how="inner")
        if j.empty:
            raise SystemExit(
                f"join produced 0 rows for {cond}. Level naming differs between "
                f"the coordinate file and the crop metadata ('L1/L2' vs 'l1_l2') "
                f"and norm_level should have handled it; an empty join here would "
                f"otherwise look like a null result rather than a bug.")
        if len(j) < 0.5 * before:
            print(f"  WARNING {cond}: only {len(j)}/{before} rows matched geometry")

        labels = d[f"labels_{cond}"][j.row.to_numpy()]
        p_sev = d[f"probs_{cond}"][j.row.to_numpy()][:, 2]
        inside_m = ~j.outside_slice_window.to_numpy()
        outside_m = j.outside_slice_window.to_numpy()

        res = {}
        for name, m in (("inside", inside_m), ("outside", outside_m)):
            res[name] = scores(labels[m], p_sev[m])

        # Paired bootstrap over patients on the DIFFERENCE, so the two subsets'
        # shared sampling noise cancels.
        pats = j.study_id.to_numpy()
        uniq = np.unique(pats)
        diffs = []
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.flatnonzero(pats == p) for p in pick])
            a = scores(labels[idx][inside_m[idx]], p_sev[idx][inside_m[idx]])["auc"]
            b = scores(labels[idx][outside_m[idx]], p_sev[idx][outside_m[idx]])["auc"]
            if not (np.isnan(a) or np.isnan(b)):
                diffs.append(a - b)
        diffs = np.array(diffs)
        res["auc_gap_inside_minus_outside"] = float(
            res["inside"]["auc"] - res["outside"]["auc"])
        res["auprc_lift_gap_inside_minus_outside"] = float(
            res["inside"]["auprc_lift"] - res["outside"]["auprc_lift"])
        if len(diffs):
            res["gap_ci95"] = [float(np.percentile(diffs, 2.5)),
                               float(np.percentile(diffs, 97.5))]
            res["gap_p_two_sided"] = float(2 * min((diffs <= 0).mean(),
                                                   (diffs >= 0).mean()))
        res["median_slices_outside_subset"] = float(
            j[outside_m].d_normal_slices.median())
        out["conditions"][cond] = res

        print(f"\n{cond}")
        for name in ("inside", "outside"):
            r = res[name]
            print(f"  {name:8s} n={r['n']:5d} severe={r['n_severe']:4d} "
                  f"prev={100*r['prevalence']:.1f}%  AUC={r['auc']:.4f}  "
                  f"AUPRC={r['auprc']:.4f} (lift {r['auprc_lift']:.2f}x)")
        print(f"  AUC gap (inside - outside) = "
              f"{res['auc_gap_inside_minus_outside']:+.4f}", end="")
        if "gap_ci95" in res:
            print(f"  95% CI [{res['gap_ci95'][0]:+.4f}, {res['gap_ci95'][1]:+.4f}]"
                  f"  p={res['gap_p_two_sided']:.3f}")
        else:
            print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("\nA positive AUC gap means the T2 model ranks Severe cases worse "
          "where the foraminal finding lies outside its 9-slice window, which is "
          "what the routing argument predicts. Near zero means the geometry does "
          "not explain the deficit. Read the AUC column, not AUPRC: the two "
          "subsets have different Severe prevalence and AUPRC has a floor at the "
          "prevalence, so raw AUPRC favours the subset with more positives no "
          "matter what the model does.")


if __name__ == "__main__":
    main()
