"""Split the paper's Grad-CAM figure into its two panels for the infographic.

The published PNG bakes "Without CBAM" and "With CBAM" into the image and sits
on a white ground. On a page that is not white that reads as a pasted box, and
the baked titles would repeat labels the layout can set itself. This crops the
two panels out so the infographic can label them in its own type.

Pixels are untouched: same figure, same case, same overlay as Figure 5 of the
paper.

    python3 figures/make_gradcam_panels.py
"""
import numpy as np
from PIL import Image
from pathlib import Path

SRC = Path("paper/camera_ready/figures/grad_cam_nocbam_vs_cbam.png")
OUT = Path("paper/camera_ready/figures")

im = np.array(Image.open(SRC).convert("RGB"))
# Non-white columns and rows mark where the two image panels actually are.
nonwhite = (im < 240).any(axis=2)
cols = np.flatnonzero(nonwhite.any(axis=0))
rows = np.flatnonzero(nonwhite.any(axis=1))

# Drop the title band: keep rows from the first tall run of dark pixels.
col_runs, start = [], None
for c in range(im.shape[1]):
    dense = nonwhite[:, c].sum() > 0.5 * len(rows)
    if dense and start is None:
        start = c
    elif not dense and start is not None:
        if c - start > 100:
            col_runs.append((start, c))
        start = None
if start is not None and im.shape[1] - start > 100:
    col_runs.append((start, im.shape[1]))

assert len(col_runs) == 2, f"expected two panels, found {len(col_runs)}"
r0 = next(r for r in rows if nonwhite[r, col_runs[0][0]:col_runs[0][1]].sum()
          > 0.8 * (col_runs[0][1] - col_runs[0][0]))
r1 = rows[-1]

for name, (c0, c1) in zip(("gradcam_nocbam.png", "gradcam_cbam.png"), col_runs):
    Image.fromarray(im[r0:r1 + 1, c0:c1 + 1]).save(OUT / name)
    print(f"wrote {name}  {c1-c0}x{r1-r0+1}")
