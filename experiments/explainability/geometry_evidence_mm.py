#!/usr/bin/env python3
"""Geometric evidence, computed in patient space instead of pixel space.

Supersedes the pixel-space comparison in geometry_evidence.py, which was wrong.

That script subtracted a foraminal annotation's (x, y) from a canal
annotation's (x, y) and treated the difference as a displacement. But foraminal
findings are annotated on Sagittal T1 and canal findings on Sagittal T2, and
those are different series with different geometry. Measured on four studies,
three had mismatched pixel grids:

    384x384 @ 0.78 mm   vs   640x640 @ 0.47 mm
    384x384 @ 0.78 mm   vs   320x320 @ 0.94 mm

A pixel offset therefore does not correspond to a fixed physical distance, and
the two coordinate systems do not even share an origin. Any number derived that
way is uninterpretable.

The fix is to map each annotation into the DICOM patient coordinate system,
which every series carries and which is common to all of them:

    P = IPP + c * PixelSpacing[1] * R + r * PixelSpacing[0] * C

where IPP is ImagePositionPatient for the annotated slice and R, C are the row
and column direction cosines from ImageOrientationPatient. All 40 sampled
studies carry both tags.

With both annotations in millimetres in the same frame, the question "is the
foraminal finding inside the T2 crop?" becomes well posed: convert the crop's
half-extent (120 x 60 px on the T2 grid) into millimetres using the T2 series'
own pixel spacing, and ask whether the foraminal point falls within it.

Usage:
    python3 experiments/explainability/geometry_evidence_mm.py --limit 300
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"
OUT = Path(__file__).resolve().parent / "results" / "geometry_evidence_mm.json"

CROP_W, CROP_H = 240, 120  # pixels on the source grid, from prep_t1_crops.py


def slice_geometry(study_id, series_id, instance_number):
    """IPP, row/col direction cosines and pixel spacing for one annotated slice."""
    d = RAW / "train_images" / str(study_id) / str(series_id)
    f = d / f"{instance_number}.dcm"
    if not f.exists():
        cand = sorted(d.glob("*.dcm"))
        if not cand:
            return None
        f = cand[len(cand) // 2]
    ds = pydicom.dcmread(str(f), stop_before_pixels=True)
    if not hasattr(ds, "ImagePositionPatient") or not hasattr(ds, "ImageOrientationPatient"):
        return None
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    slice_gap = getattr(ds, "SpacingBetweenSlices", None) or getattr(ds, "SliceThickness", None)
    return {
        "ipp": np.array(ds.ImagePositionPatient, dtype=float),
        "row_dir": iop[3:6],   # direction along increasing row index
        "col_dir": iop[0:3],   # direction along increasing column index
        "spacing": np.array(ds.PixelSpacing, dtype=float),  # [row, col] mm
        "slice_gap": float(slice_gap) if slice_gap else None,
        "rows": int(ds.Rows),
        "cols": int(ds.Columns),
    }


def to_patient_mm(x_col, y_row, g):
    """Pixel (column, row) on one slice -> 3D patient coordinates in mm."""
    return (g["ipp"]
            + x_col * g["spacing"][1] * g["col_dir"]
            + y_row * g["spacing"][0] * g["row_dir"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300,
                    help="Studies to process. Each needs DICOM header reads, so "
                         "the full set is slow; 300 already gives tight CIs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    coords = pd.read_csv(RAW / "train_label_coordinates.csv")
    desc = pd.read_csv(RAW / "train_series_descriptions.csv")
    c = coords.merge(desc, on=["study_id", "series_id"], how="left")
    c["cond"] = c["condition"].str.lower()

    canal = c[c["cond"].str.contains("canal")]
    foram = c[c["cond"].str.contains("neural")]

    studies = np.array(sorted(set(canal.study_id) & set(foram.study_id)))
    rng = np.random.default_rng(args.seed)
    if args.limit and args.limit < len(studies):
        studies = rng.choice(studies, size=args.limit, replace=False)
    print(f"{len(studies)} studies with both canal and foraminal annotations")

    recs = []
    geo_cache = {}
    skipped = 0
    for sid in studies:
        ca = canal[canal.study_id == sid]
        fo = foram[foram.study_id == sid]
        for level in set(ca.level) & set(fo.level):
            crow = ca[ca.level == level].iloc[0]
            key = (sid, crow.series_id, crow.instance_number)
            if key not in geo_cache:
                geo_cache[key] = slice_geometry(*key)
            gc = geo_cache[key]
            if gc is None:
                skipped += 1
                continue
            canal_mm = to_patient_mm(crow.x, crow.y, gc)
            # The crop half-extent in mm, on the T2 grid the crop is taken from.
            half_col_mm = (CROP_W / 2) * gc["spacing"][1]
            half_row_mm = (CROP_H / 2) * gc["spacing"][0]

            for _, frow in fo[fo.level == level].iterrows():
                fkey = (sid, frow.series_id, frow.instance_number)
                if fkey not in geo_cache:
                    geo_cache[fkey] = slice_geometry(*fkey)
                gf = geo_cache[fkey]
                if gf is None:
                    skipped += 1
                    continue
                for_mm = to_patient_mm(frow.x, frow.y, gf)
                delta = for_mm - canal_mm
                # Project the displacement onto the T2 slice's own in-plane axes,
                # which are the axes the crop window is defined along.
                d_col = float(np.dot(delta, gc["col_dir"]))
                d_row = float(np.dot(delta, gc["row_dir"]))
                d_norm = float(np.dot(delta, np.cross(gc["row_dir"], gc["col_dir"])))
                side = "left" if "left" in frow["cond"] else "right"
                # The crop also spans only 9 slices: offsets -4..+4 around the
                # annotated slice. Its through-plane half-extent is therefore
                # 4 * the T2 series' slice gap.
                half_slice_mm = (4 * gc["slice_gap"]) if gc["slice_gap"] else None
                inplane_ok = (abs(d_col) <= half_col_mm
                              and abs(d_row) <= half_row_mm)
                slice_ok = (half_slice_mm is None
                            or abs(d_norm) <= half_slice_mm)
                recs.append({
                    "study_id": int(sid), "level": level, "side": side,
                    "d_col_mm": d_col, "d_row_mm": d_row, "d_normal_mm": d_norm,
                    "inplane_mm": float(np.hypot(d_col, d_row)),
                    "inside_inplane": bool(inplane_ok),
                    "inside_slices": bool(slice_ok),
                    "inside": bool(inplane_ok and slice_ok),
                    "half_col_mm": float(half_col_mm),
                    "half_row_mm": float(half_row_mm),
                    "half_slice_mm": half_slice_mm,
                    "d_normal_slices": (abs(d_norm) / gc["slice_gap"]
                                        if gc["slice_gap"] else None),
                })

    df = pd.DataFrame(recs)
    if df.empty:
        raise SystemExit("no usable pairs")

    out = {"n_pairs": int(len(df)), "n_studies": int(df.study_id.nunique()),
           "skipped_missing_geometry": skipped,
           "crop_window_px": [CROP_W, CROP_H], "sides": {}}
    print(f"\n{len(df)} pairs over {df.study_id.nunique()} studies "
          f"({skipped} skipped for missing geometry)")
    print(f"crop half-extent, median: "
          f"{df.half_col_mm.median():.1f} x {df.half_row_mm.median():.1f} mm")

    for side, g in df.groupby("side"):
        out["sides"][side] = {
            "n": int(len(g)),
            "inplane_mm_median": float(g.inplane_mm.median()),
            "inplane_mm_p90": float(g.inplane_mm.quantile(0.9)),
            "d_col_mm_median": float(g.d_col_mm.abs().median()),
            "d_row_mm_median": float(g.d_row_mm.abs().median()),
            "d_normal_mm_median": float(g.d_normal_mm.abs().median()),
            "d_normal_slices_median": float(g.d_normal_slices.median()),
            "d_normal_slices_p90": float(g.d_normal_slices.quantile(0.9)),
            "half_slice_mm_median": float(g.half_slice_mm.median()),
            "outside_inplane_fraction": float(1 - g.inside_inplane.mean()),
            "outside_slices_fraction": float(1 - g.inside_slices.mean()),
            "outside_fraction": float(1 - g.inside.mean()),
        }
        s = out["sides"][side]
        print(f"\n{side} foraminal (n={s['n']})")
        print(f"  in-plane from canal:   median {s['inplane_mm_median']:.1f} mm, "
              f"p90 {s['inplane_mm_p90']:.1f} mm  "
              f"(window half-extent {g.half_col_mm.median():.0f}x{g.half_row_mm.median():.0f} mm)")
        print(f"  through-plane:         median {s['d_normal_mm_median']:.1f} mm "
              f"= {s['d_normal_slices_median']:.1f} slices, "
              f"p90 {s['d_normal_slices_p90']:.1f} slices  "
              f"(window is +/-4 slices = {s['half_slice_mm_median']:.1f} mm)")
        print(f"  outside IN-PLANE window:      {100*s['outside_inplane_fraction']:.1f}%")
        print(f"  outside 9-SLICE window:       {100*s['outside_slices_fraction']:.1f}%   <-- the real gap")
        print(f"  outside EITHER:               {100*s['outside_fraction']:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
