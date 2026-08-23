"""Patient-level train/val splits that stay consistent across datasets.

The default split calls `train_test_split` on whatever patient list the loaded
metadata happens to contain. That is fine while every run reads the same file,
but it breaks the moment two datasets describe the same patients:

    rsna_preprocessed/train_metadata.csv   1973 studies
    rsna_preprocessed_t1/t1_metadata.csv   1972 studies

Those lists differ by three studies. `train_test_split(..., random_state=42)`
on arrays of different lengths does not produce two slightly different
partitions -- it produces two unrelated ones. Measured on the pair above: the
two validation sets share 214 of 395 patients, and 181 patients sit in one
run's validation set while being in the other run's training set.

Nothing about a single run is wrong: each model is still scored on patients it
never trained on. What breaks is any statement that compares the two, or any
system that routes one condition to one model and another condition to the
other, since for those 181 patients one branch has memorised the case.

`hash` mode fixes this at the source: the partition is a property of the
patient id, not of the list it arrives in. A patient always lands in the same
fold, in every dataset, forever -- including datasets that do not exist yet.
"""
import hashlib

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_MODES = ("random", "hash")


def _unit_interval(study_id, seed: int) -> float:
    """Map a study id to a stable float in [0, 1).

    Python's built-in hash() is salted per process, so it cannot be used: the
    same patient would change fold between runs. MD5 of the id plus the seed is
    stable across processes, machines and Python versions.
    """
    digest = hashlib.md5(f"{seed}:{study_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_patients(patients, val_split: float = 0.2, seed: int = 42,
                   mode: str = "random"):
    """Return (train_patients, val_patients) as numpy arrays.

    mode="random" reproduces the historical behaviour exactly, so existing
    checkpoints and dumps stay comparable.
    mode="hash" assigns each patient by a hash of its id, so two datasets
    listing overlapping patients agree on who is held out.
    """
    if mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode {mode!r}; expected one of {SPLIT_MODES}")

    patients = np.asarray(patients)
    if mode == "random":
        return train_test_split(patients, test_size=val_split, random_state=seed)

    scores = np.array([_unit_interval(p, seed) for p in patients])
    is_val = scores < val_split
    return patients[~is_val], patients[is_val]


def describe_split(train_patients, val_patients, mode: str) -> str:
    total = len(train_patients) + len(val_patients)
    frac = len(val_patients) / total if total else float("nan")
    note = "" if mode == "random" else "  (stable across datasets)"
    return (f"split={mode}  train={len(train_patients)}  val={len(val_patients)} "
            f"({frac:.1%}){note}")
