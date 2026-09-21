"""Export one mid-sagittal lumbar T2 slice for the conference infographic.

RSNA 2024 data is licensed for non-commercial academic research and education,
with an obligation not to attempt to identify the subjects. This exports pixel
data only: no DICOM header travels with the PNG, and the slice is inspected for
burned-in annotation before use.

    python3 figures/make_mri_panel.py
"""
import numpy as np, pydicom
from pathlib import Path
from PIL import Image

RAW = Path("rsna-2024-lumbar-spine-degenerative-classification")
STUDY, SERIES = "4003253", "702807833"
OUT = Path("paper/camera_ready/figures/mri_sagittal_panel.png")

files = sorted((RAW / "train_images" / STUDY / SERIES).glob("*.dcm"),
               key=lambda p: int(p.stem))
ds = pydicom.dcmread(str(files[len(files) // 2]))
img = ds.pixel_array.astype(np.float32)

# Percentile windowing. A flat 1st to 99th clip left this slice muddy on a
# pale page, so the window is tightened and the midtones lifted with a gamma
# below 1. This is display windowing, the same knob a radiologist turns; it
# changes no pixel ordering and no measurement.
lo, hi = np.percentile(img, [2, 99.5])
img = np.clip((img - lo) / (hi - lo), 0, 1) ** 0.82

# Crop to a portrait frame around the spine. The lumbar column sits in the
# middle third horizontally on these sagittal series.
h, w = img.shape
x0, x1 = int(w * 0.30), int(w * 0.72)
y0, y1 = int(h * 0.16), int(h * 0.92)
img = img[y0:y1, x0:x1]

Image.fromarray((img * 255).astype(np.uint8)).save(OUT)
print(f"wrote {OUT}  {img.shape[1]}x{img.shape[0]}  "
      f"aspect {img.shape[1]/img.shape[0]:.2f}")
