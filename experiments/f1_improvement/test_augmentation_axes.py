#!/usr/bin/env python3
"""
Checks that each geometric augmentation touches the axis it claims to.

The bug being guarded against: RandomHorizontalFlip flips dims=[-1], which on a
(9, 112, 224) sagittal crop is the anterior-posterior axis, NOT left-right --
laterality lives on the slice axis, dim 0. The old code swapped left_*/right_*
labels on that flip, so half the foraminal samples got their labels corrupted
whenever the default was used.

Run:  python3 experiments/f1_improvement/test_augmentation_axes.py
Exits non-zero on failure, so it works as a pre-run smoke test.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spinenet.augmentation import (  # noqa: E402
    RandomHorizontalFlip,
    RandomSliceReverse,
    get_training_augmentation,
)

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILURES.append(name)


def marked_volume():
    """(9, 112, 224) where every voxel encodes its own (slice, row, col), so any
    axis permutation is detectable rather than merely plausible."""
    d, h, w = 9, 112, 224
    s = torch.arange(d).view(d, 1, 1).float()
    r = torch.arange(h).view(1, h, 1).float()
    c = torch.arange(w).view(1, 1, w).float()
    return s * 1e6 + r * 1e3 + c


def labels():
    return {"spinal_canal": 0, "left_foraminal": 1, "right_foraminal": 2}


def main():
    vol = marked_volume()

    print("\nRandomHorizontalFlip -- must touch the LAST axis only")
    out, lab = RandomHorizontalFlip(p=1.0)(vol.clone(), labels())
    check("flips the anterior-posterior (last) axis",
          torch.equal(out, torch.flip(vol, dims=[-1])))
    # Compare the SLICE component of the encoding, not a fixed column: after a
    # column flip, column 0 legitimately holds what used to be in column 223.
    # What must not change is which slice each voxel came from.
    check("leaves the slice axis untouched",
          torch.equal(torch.div(out, 1e6, rounding_mode="floor"),
                      torch.div(vol, 1e6, rounding_mode="floor")),
          "slice ordering changed, which would make it a laterality flip")
    check("does NOT swap left/right labels by default",
          lab["left_foraminal"] == 1 and lab["right_foraminal"] == 2)

    print("\nRandomHorizontalFlip -- label swapping must be refused")
    try:
        RandomHorizontalFlip(p=1.0, swap_labels=True)
        check("raises when asked to swap L/R on the AP flip", False,
              "constructor accepted a label-corrupting configuration")
    except ValueError:
        check("raises when asked to swap L/R on the AP flip", True)
    try:
        RandomHorizontalFlip(p=1.0, swap_labels=True, allow_wrong_axis_swap=True)
        check("escape hatch still allows reproducing an old run", True)
    except ValueError:
        check("escape hatch still allows reproducing an old run", False)

    print("\nRandomSliceReverse -- must mirror left-right and swap labels together")
    out, lab = RandomSliceReverse(p=1.0)(vol.clone(), labels())
    check("reverses the slice axis", torch.equal(out, torch.flip(vol, dims=[0])))
    check("leaves the anterior-posterior axis untouched",
          torch.equal(out[0, 0, :], vol[-1, 0, :]))
    check("swaps left/right labels",
          lab["left_foraminal"] == 2 and lab["right_foraminal"] == 1)
    check("leaves the midline condition alone", lab["spinal_canal"] == 0)

    print("\nRandomSliceReverse -- applying it twice must be the identity")
    t = RandomSliceReverse(p=1.0)
    once, lab1 = t(vol.clone(), labels())
    twice, lab2 = t(once, lab1)
    check("image returns to the original", torch.equal(twice, vol))
    check("labels return to the original", lab2 == labels())

    print("\nget_training_augmentation -- defaults must reproduce published runs")
    default_geo = [type(t).__name__ for t in get_training_augmentation('medium').transforms]
    check("AP flip present by default", "RandomHorizontalFlip" in default_geo)
    check("slice reverse absent by default", "RandomSliceReverse" not in default_geo)

    with_reverse = [type(t).__name__
                    for t in get_training_augmentation('medium', slice_reverse=True).transforms]
    check("--slice-reverse adds it", "RandomSliceReverse" in with_reverse)

    without_ap = [type(t).__name__
                  for t in get_training_augmentation('medium', ap_flip=False).transforms]
    check("--no-ap-flip removes it", "RandomHorizontalFlip" not in without_ap)

    both = [type(t).__name__ for t in
            get_training_augmentation('medium', ap_flip=False,
                                      slice_reverse=True).transforms]
    check("the two flags are independent",
          "RandomSliceReverse" in both and "RandomHorizontalFlip" not in both,
          f"got {both}")

    print("\nIntensity transforms must survive all modes")
    for mode in ("light", "medium", "heavy"):
        names = [type(t).__name__ for t in get_training_augmentation(mode).transforms]
        check(f"{mode}: brightness/contrast kept",
              "RandomBrightnessContrast" in names)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {', '.join(FAILURES)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
