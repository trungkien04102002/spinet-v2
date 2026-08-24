#!/usr/bin/env python3
"""Does the model's evidence sit where the radiologist pointed?

Ground truth here is a single point per (study, level, condition): 48692 rows,
100% with exactly one coordinate, and no mask, box or radius anywhere in
train_label_coordinates.csv. So IoU and the original Pointing Game (Zhang et al.
2018), which need an extent, are unavailable. What is available is the distance
from the saliency peak to that point, thresholded at a tolerance, following the
TorchRay PointingGame convention and reported at several tolerances rather than
one, with the continuous distance distribution alongside it as Saporta et al.
2022 (CheXlocalize) do.

TWO BASELINES ARE MANDATORY, not optional. The crop is centred on an annotated
coordinate by construction (prep_t1_crops.py:230), so a model that only ever
looked at the middle of the frame would score well. Without a chance baseline
the hit rate says nothing at all:

    random    peak drawn uniformly inside the crop
    centre    peak fixed at the crop centre

If saliency does not beat "always guess the middle", it has added no information.

The coordinate chain goes through patient space, and it has to. The foraminal
annotation lives on Sagittal T1 while the T2 crop's origin comes from the canal
coordinate on Sagittal T2, and those series have different pixel grids (measured:
384x384 at 0.78 mm against 640x640 at 0.47 mm). Subtracting pixel coordinates
across them is meaningless. So:

    annotation pixel (T1 slice) -> patient mm
    canal pixel     (T2 slice) -> patient mm
    displacement projected onto the T2 slice's row/col/normal axes
    -> T2 pixels via T2 pixel spacing
    -> crop pixels, then scaled by 224/crop_w and 112/crop_h
    -> volume voxel index

Distances are reported in millimetres, since a voxel is not isotropic here: the
in-plane voxel is about 1 mm after the 240x120 -> 224x112 resize, while the slice
step is about 4.5 mm.

Usage:
    python3 experiments/explainability/localization_audit.py --n-cases 120
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parents[1]
for p in (str(REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import saliency_methods as S  # noqa: E402
from geometry_evidence_mm import slice_geometry, to_patient_mm  # noqa: E402
from geometry_vs_performance import norm_level  # noqa: E402
from spinenet.patient_split import split_patients  # noqa: E402
from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402

RAW = REPO_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"
CROP_W, CROP_H = 240, 120
VOL_D, VOL_H, VOL_W = 9, 112, 224
TAUS_MM = (5.0, 10.0, 15.0, 20.0)

COND_KEY = {
    "spinal_canal": "spinal canal",
    "left_foraminal": "left neural",
    "right_foraminal": "right neural",
}


def annotation_in_volume(study_id, level, condition, coords):
    """Map one annotation into (slice, row, col) of the (9,112,224) volume.

    The crop is centred on the CANAL coordinate for a T2 crop, so the canal row
    supplies the crop origin and the target condition's row supplies the point
    being located. Returns None when either is missing or lacks geometry.
    """
    sub = coords[(coords.study_id == study_id) & (coords.level == level)]
    canal = sub[sub.cond.str.contains("spinal canal")]
    target = sub[sub.cond.str.contains(COND_KEY[condition])]
    if canal.empty or target.empty:
        return None
    cr, tr = canal.iloc[0], target.iloc[0]

    gc = slice_geometry(study_id, cr.series_id, cr.instance_number)
    gt = slice_geometry(study_id, tr.series_id, tr.instance_number)
    if gc is None or gt is None or not gc["slice_gap"]:
        return None

    delta = to_patient_mm(tr.x, tr.y, gt) - to_patient_mm(cr.x, cr.y, gc)
    normal = np.cross(gc["row_dir"], gc["col_dir"])
    d_col_mm = float(np.dot(delta, gc["col_dir"]))
    d_row_mm = float(np.dot(delta, gc["row_dir"]))
    d_nrm_mm = float(np.dot(delta, normal))

    # Displacement in source pixels on the T2 grid, then into crop pixels. The
    # crop is centred on the canal point, so the canal sits at (CROP_W/2, CROP_H/2).
    col_crop = CROP_W / 2 + d_col_mm / gc["spacing"][1]
    row_crop = CROP_H / 2 + d_row_mm / gc["spacing"][0]
    # The resize maps CROP_W -> VOL_W and CROP_H -> VOL_H.
    col_vol = col_crop * VOL_W / CROP_W
    row_vol = row_crop * VOL_H / CROP_H
    slice_vol = (VOL_D - 1) / 2 + d_nrm_mm / gc["slice_gap"]

    return {
        "slice": slice_vol, "row": row_vol, "col": col_vol,
        # mm per voxel, for converting a voxel distance back to millimetres
        "mm_per_col": gc["spacing"][1] * CROP_W / VOL_W,
        "mm_per_row": gc["spacing"][0] * CROP_H / VOL_H,
        "mm_per_slice": gc["slice_gap"],
        "inside_volume": bool(0 <= col_vol < VOL_W and 0 <= row_vol < VOL_H
                              and 0 <= slice_vol < VOL_D),
    }


def dist_mm(peak, gt):
    """Euclidean distance in millimetres between a voxel peak and the target."""
    return float(np.sqrt(((peak[0] - gt["slice"]) * gt["mm_per_slice"]) ** 2
                         + ((peak[1] - gt["row"]) * gt["mm_per_row"]) ** 2
                         + ((peak[2] - gt["col"]) * gt["mm_per_col"]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam-checkpoint", type=str,
                    default="checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "rsna_preprocessed")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--split-mode", type=str, default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", nargs="+",
                    default=["spinal_canal", "left_foraminal", "right_foraminal"])
    ap.add_argument("--methods", nargs="+", default=["grad_cam", "layer_cam"])
    ap.add_argument("--stage", type=str, default="cbam4")
    ap.add_argument("--layer-cam-stage", type=str, default="layer2",
                    help="Layer-CAM is included precisely to attach shallower, "
                         "where plain Grad-CAM collapses.")
    ap.add_argument("--target-class", type=int, default=2)
    ap.add_argument("--n-cases", type=int, default=120)
    ap.add_argument("--out", type=Path,
                    default=_HERE / "results" / "localization_audit.json")
    args = ap.parse_args()

    coords = pd.read_csv(RAW / "train_label_coordinates.csv")
    coords["cond"] = coords["condition"].str.lower()
    coords["level"] = norm_level(coords["level"])

    md = pd.read_csv(args.data_dir / f"{args.split}_metadata.csv")
    _, val_p = split_patients(md.study_id.unique(), 0.2, args.seed, args.split_mode)
    v = md[md.study_id.isin(val_p)].reset_index(drop=True)
    v["level_n"] = norm_level(v["level"])

    model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    ck = torch.load(args.cbam_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model.eval()

    rng = np.random.default_rng(0)
    out = {"taus_mm": list(TAUS_MM), "target_class": args.target_class,
           "conditions": {}}

    for condition in args.conditions:
        # Only cases the model was asked to call Severe: a localisation claim
        # about a Normal case has no target to localise.
        pos = v[v[condition] == args.target_class]
        if pos.empty:
            continue
        rows = pos.sample(n=min(args.n_cases, len(pos)), random_state=args.seed)
        recs = []
        for r in rows.itertuples():
            gt = annotation_in_volume(r.study_id, r.level_n, condition, coords)
            if gt is None or not gt["inside_volume"]:
                continue
            vol = torch.from_numpy(
                np.load(args.data_dir / "volumes" / r.filepath)).float()[None, None]
            rec = {"study_id": int(r.study_id), "level": r.level_n}
            for method in args.methods:
                stage = (args.layer_cam_stage if method == "layer_cam"
                         else args.stage)
                m = S.cam(model, vol, condition, args.target_class, method, stage)
                up = S.upsample_to_input(m, (VOL_D, VOL_H, VOL_W))
                rec[method] = dist_mm(S.peak_location(up), gt)
            # Baselines, on the same case so the comparison is paired.
            rec["baseline_random"] = dist_mm(
                (rng.integers(VOL_D), rng.integers(VOL_H), rng.integers(VOL_W)), gt)
            rec["baseline_centre"] = dist_mm(
                ((VOL_D - 1) / 2, (VOL_H - 1) / 2, (VOL_W - 1) / 2), gt)
            recs.append(rec)

        if not recs:
            print(f"{condition}: no usable cases")
            continue
        df = pd.DataFrame(recs)
        entry = {"n": int(len(df)), "estimators": {}}
        for col in list(args.methods) + ["baseline_random", "baseline_centre"]:
            d = df[col].to_numpy()
            entry["estimators"][col] = {
                "median_mm": float(np.median(d)),
                "mean_mm": float(np.mean(d)),
                **{f"hit@{t:g}mm": float((d <= t).mean()) for t in TAUS_MM},
            }
        out["conditions"][condition] = entry

        print(f"\n{condition}  (n={len(df)} cases)")
        hdr = "  ".join(f"hit@{t:g}mm" for t in TAUS_MM)
        print(f"  {'estimator':<18s}{'median mm':>11s}  {hdr}")
        for col, e in entry["estimators"].items():
            hits = "  ".join(f"{e[f'hit@{t:g}mm']:9.3f}" for t in TAUS_MM)
            print(f"  {col:<18s}{e['median_mm']:>11.1f}  {hits}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")
    print("\nA method only carries information if it beats BOTH baselines. The "
          "centre baseline is the demanding one, because the crop is centred on "
          "an annotation by construction.")


if __name__ == "__main__":
    main()
