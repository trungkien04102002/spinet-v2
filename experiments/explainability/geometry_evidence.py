#!/usr/bin/env python3
"""SUPERSEDED AND WRONG. Use geometry_evidence_mm.py instead.

Kept only so the error is traceable. This script compares a foraminal
annotation's pixel (x, y) against a canal annotation's pixel (x, y) as if they
shared a coordinate frame. They do not: foraminal findings are annotated on
Sagittal T1 and canal findings on Sagittal T2, and three of four sampled
studies had different pixel grids (384x384 at 0.78 mm against 640x640 at
0.47 mm; 384x384 at 0.78 mm against 320x320 at 0.94 mm). A pixel offset across
two such grids is not a physical distance, so the 23.4% / 23.2% it reports is
meaningless.

geometry_evidence_mm.py redoes the measurement in the DICOM patient coordinate
system and finds the real gap is along the slice axis, not in plane: 0.3%
outside in plane, but 32.2% (left) and 47.2% (right) outside the 9-slice window.

Original docstring follows.

Model-free evidence that the T2 crop cannot contain the foraminal finding.

This is the strongest single piece of evidence for the series-routing claim, and
the cheapest: it uses no model, no saliency method and no threshold, so there is
nothing in it to dispute except the geometry itself.

The pipeline cut ONE crop per disc level from Sagittal T2, centred on the canal
coordinate, and attached all three condition labels to it. But radiologists
annotate foraminal narrowing on Sagittal T1 and canal stenosis on Sagittal T2,
and the two annotations are not in the same place. This script measures how
often the foraminal annotation falls outside the crop window entirely -- that
is, how often the region being graded is not in the image the model was given.

Reads train_label_coordinates.csv from the raw RSNA download. That file is
gitignored (it lives inside the ~33 GB competition directory) and is NOT
redistributed here, so the derived aggregates are written to
results/geometry_evidence.json and committed instead. If the raw data is gone,
the JSON still holds the numbers; if the raw data is present, this script
regenerates and overwrites them.

Usage:
    python3 experiments/explainability/geometry_evidence.py
    python3 experiments/explainability/geometry_evidence.py --no-clip
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COORDS = (REPO_ROOT / "rsna-2024-lumbar-spine-degenerative-classification"
          / "train_label_coordinates.csv")
OUT = Path(__file__).resolve().parent / "results" / "geometry_evidence.json"

# The crop window used by rsna_dataloader.py and prep_t1_crops.py, in pixels.
CROP_W, CROP_H = 240, 120


def realised_window(x, y, img_w, img_h, clip):
    """Where the crop actually lands, accounting for border clipping.

    prep_t1_crops.py does:
        x1 = max(0, x - CROP_W // 2); x2 = min(img_w, x1 + CROP_W)
    so for a coordinate near an edge the window shifts inward rather than
    extending past the border, and the annotation stops being centred. Ignoring
    that biases the "outside" rate downward.
    """
    if not clip:
        return x - CROP_W / 2, x + CROP_W / 2, y - CROP_H / 2, y + CROP_H / 2
    x1 = np.maximum(0, x - CROP_W // 2)
    y1 = np.maximum(0, y - CROP_H // 2)
    x2 = np.minimum(img_w, x1 + CROP_W)
    y2 = np.minimum(img_h, y1 + CROP_H)
    # If clipped at the far edge, the window slides back rather than shrinking.
    x1 = np.maximum(0, x2 - CROP_W)
    y1 = np.maximum(0, y2 - CROP_H)
    return x1, x2, y1, y2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", type=Path, default=COORDS)
    ap.add_argument("--img-w", type=int, default=640,
                    help="Assumed source image width for border clipping. RSNA "
                         "sagittal series are commonly 640x640; pass the real "
                         "value if known per-series.")
    ap.add_argument("--img-h", type=int, default=640)
    ap.add_argument("--no-clip", action="store_true",
                    help="Skip border clipping (upper bound on centredness).")
    args = ap.parse_args()

    if not args.coords.exists():
        raise SystemExit(
            f"{args.coords} not found. It is part of the gitignored raw RSNA "
            f"download and is not redistributed in this repository. The derived "
            f"numbers are already committed at {OUT.relative_to(REPO_ROOT)}; "
            f"re-download the competition data to regenerate them.")

    c = pd.read_csv(args.coords)
    c["cond"] = c["condition"].str.lower()

    canal = (c[c["cond"].str.contains("canal")][["study_id", "level", "x", "y"]]
             .rename(columns={"x": "cx", "y": "cy"}))

    out = {
        "crop_window_px": {"width": CROP_W, "height": CROP_H},
        "border_clipping_applied": not args.no_clip,
        "assumed_image_size": [args.img_w, args.img_h],
        "conditions": {},
    }

    for side in ("left", "right"):
        f = c[c["cond"].str.contains(f"{side} neural")][
            ["study_id", "level", "x", "y"]]
        m = f.merge(canal, on=["study_id", "level"])
        if m.empty:
            continue

        x1, x2, y1, y2 = realised_window(
            m["cx"].to_numpy(), m["cy"].to_numpy(),
            args.img_w, args.img_h, clip=not args.no_clip)
        inside = ((m["x"].to_numpy() >= x1) & (m["x"].to_numpy() <= x2)
                  & (m["y"].to_numpy() >= y1) & (m["y"].to_numpy() <= y2))

        dx = (m["x"] - m["cx"]).abs()
        dy = (m["y"] - m["cy"]).abs()
        dist = np.sqrt((m["x"] - m["cx"]) ** 2 + (m["y"] - m["cy"]) ** 2)

        out["conditions"][f"{side}_foraminal"] = {
            "n": int(len(m)),
            "offset_from_canal_px": {
                "dx_median": float(dx.median()),
                "dy_median": float(dy.median()),
                "dx_p90": float(dx.quantile(0.9)),
                "dy_p90": float(dy.quantile(0.9)),
                "euclidean_median": float(np.median(dist)),
                "euclidean_p90": float(np.percentile(dist, 90)),
            },
            "inside_t2_crop_fraction": float(inside.mean()),
            "outside_t2_crop_fraction": float(1 - inside.mean()),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"crop window {CROP_W}x{CROP_H} px, "
          f"clipping={'on' if not args.no_clip else 'off'}")
    for name, d in out["conditions"].items():
        o = d["offset_from_canal_px"]
        print(f"\n{name}  (n={d['n']})")
        print(f"  offset from canal: dx median {o['dx_median']:.0f} px, "
              f"dy median {o['dy_median']:.0f} px, "
              f"euclidean median {o['euclidean_median']:.0f} px")
        print(f"  OUTSIDE the T2 crop window: "
              f"{100 * d['outside_t2_crop_fraction']:.1f}%")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
