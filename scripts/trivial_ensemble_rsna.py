#!/usr/bin/env python3
"""Trivial-ensemble control for RSNA (reviewer concern C2).

Question a reviewer will ask: "Is the Hybrid's gain just a basic ensemble of
the two independent branches?" i.e. does averaging the softmax of the plain
SpineNetV2 baseline and the BMC-only model match the learned-fusion Hybrid?

This script needs NO training. It loads existing checkpoints, runs each on the
SAME RSNA validation split (patient-level, shuffle=False so samples align),
and reports the 9 paper metrics for:
    - SpineNetV2 (baseline)
    - BMC-only   (hybrid with CBAM branch zeroed)
    - Ensemble   (mean of the two softmax outputs)   <-- the control
    - Hybrid     (learned concat-MLP fusion)         [optional, --hybrid-ckpt]

If Ensemble < Hybrid, the learned fusion adds value beyond trivial averaging.

Metric machinery is reused from eval_rsna_auc.py / spinenet.auc_metrics so the
numbers are directly comparable to Table 1.

Example (seed 42):
  export PYTHONPATH=$PYTHONPATH:$(pwd)
  python3 scripts/trivial_ensemble_rsna.py \
      --baseline-ckpt checkpoints/v3_20260503/baseline/best_model.pth \
      --bmc-ckpt      checkpoints/v3_20260503/bmc_only/best_model_hybrid_biomedclip_only.pth \
      --cbam-checkpoint checkpoints/v3_20260503/cbam/best_model_attention.pth \
      --hybrid-ckpt   checkpoints/v3_20260503/hybrid/best_model_hybrid.pth \
      --output experiments/v3_rsna_multiseed/trivial_ensemble_seed42.json
"""
import argparse
import json
import sys
from pathlib import Path

# This script lives in scripts/; put the repo root on sys.path so the root-level
# modules (eval_rsna_auc, train_rsna_hybrid) import regardless of PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from eval_rsna_auc import (
    CONDITIONS,
    build_val_loader,
    infer_baseline_or_cbam,
    infer_hybrid,
)
from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)

CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]
SEVERE = 2


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_nine(per_cond):
    """per_cond: {cond: {'probs': [N,3], 'labels': [N]}} -> dict of 9 metrics."""
    accs, f1m, recm, precm, sevf1 = [], [], [], [], []
    for c in CONDITIONS:
        probs = per_cond[c]["probs"]
        labels = per_cond[c]["labels"]
        mask = labels >= 0
        y = labels[mask]
        p = probs[mask].argmax(1)
        accs.append(accuracy_score(y, p))
        f1m.append(f1_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
        recm.append(recall_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
        precm.append(precision_score(y, p, labels=[0, 1, 2], average="macro", zero_division=0))
        sevf1.append(f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)[SEVERE])

    probs_dict = {c: per_cond[c]["probs"] for c in CONDITIONS}
    labels_dict = {c: per_cond[c]["labels"] for c in CONDITIONS}
    per = compute_auc_auprc_per_condition(probs_dict, labels_dict, CLASS_NAMES)
    o = aggregate_overall_auprc(per, CLASS_NAMES)

    return {
        "Mean Accuracy": float(np.mean(accs) * 100.0),
        "Mean F1 macro": float(np.mean(f1m)),
        "Mean Recall macro": float(np.mean(recm)),
        "Mean Precision macro": float(np.mean(precm)),
        "Mean AUC macro": float(o["macro_auc"]),
        "Mean AUPRC macro": float(o["macro_auprc"]),
        "Severe F1": float(np.mean(sevf1)),
        "Severe AUC": float(o["severe_auc"]),
        "Severe AUPRC": float(o["severe_auprc"]),
    }


def _check_aligned(a, b):
    """The two inference passes must see identical samples/labels in order."""
    for c in CONDITIONS:
        if not np.array_equal(a[c]["labels"], b[c]["labels"]):
            raise RuntimeError(
                f"Label mismatch on '{c}': the two models saw different val samples. "
                "Use the same --val-split and --seed; loader must be shuffle=False.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--bmc-ckpt", required=True)
    ap.add_argument("--cbam-checkpoint", required=True,
                    help="CBAM ckpt used to construct the BMC-only / Hybrid model")
    ap.add_argument("--hybrid-ckpt", default=None,
                    help="Optional: also eval the real learned-fusion Hybrid")
    ap.add_argument("--data-dir", default="rsna_preprocessed")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--slice-strategy", default="static", choices=["static", "dynamic"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = _pick_device()
    print(f"Device: {device}")
    loader, n_val = build_val_loader(
        args.data_dir, args.val_split, args.seed, args.batch_size, args.num_workers)
    print(f"Val samples: {n_val}\n")

    from spinenet.models.grading_baseline import GradingModelBaseline
    from spinenet.models.grading_hybrid import SpineNetHybrid
    from train_rsna_hybrid import build_text_database

    # ---- 1. baseline (SpineNetV2) ----
    print("[1] SpineNetV2 baseline ...")
    base = GradingModelBaseline(format="rsna")
    st = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    base.load_state_dict(st.get("model_state_dict", st), strict=False)
    base.to(device)
    base_pred = infer_baseline_or_cbam(base, loader, device)

    # ---- 2. BMC-only (CBAM branch zeroed at forward) ----
    print("[2] BMC-only ...")
    bmc = SpineNetHybrid(
        cbam_checkpoint_path=args.cbam_checkpoint,
        biomedclip_device=str(device),
        slice_strategy=args.slice_strategy,
        ablate_branch="biomedclip_only",
    ).to(device)
    st = torch.load(args.bmc_ckpt, map_location=device, weights_only=False)
    bmc.load_state_dict(st.get("model_state_dict", st), strict=False)
    bmc_text = build_text_database(bmc, device)
    bmc_pred = infer_hybrid(bmc, loader, bmc_text, device)

    _check_aligned(base_pred, bmc_pred)

    # ---- 3. trivial ensemble = mean of the two softmaxes ----
    ens_pred = {
        c: {"probs": 0.5 * (base_pred[c]["probs"] + bmc_pred[c]["probs"]),
            "labels": base_pred[c]["labels"]}
        for c in CONDITIONS
    }

    results = {
        "SpineNetV2": compute_nine(base_pred),
        "BMC-only": compute_nine(bmc_pred),
        "Ensemble(Base+BMC)": compute_nine(ens_pred),
    }

    # ---- 4. optional real Hybrid ----
    if args.hybrid_ckpt:
        print("[4] Hybrid (learned fusion) ...")
        hyb = SpineNetHybrid(
            cbam_checkpoint_path=args.cbam_checkpoint,
            biomedclip_device=str(device),
            slice_strategy=args.slice_strategy,
            ablate_branch="none",
        ).to(device)
        st = torch.load(args.hybrid_ckpt, map_location=device, weights_only=False)
        hyb.load_state_dict(st.get("model_state_dict", st), strict=False)
        hyb_text = build_text_database(hyb, device)
        hyb_pred = infer_hybrid(hyb, loader, hyb_text, device)
        _check_aligned(base_pred, hyb_pred)
        results["Hybrid(learned)"] = compute_nine(hyb_pred)

    # ---- print comparison ----
    metric_keys = list(next(iter(results.values())).keys())
    cols = list(results.keys())
    print("\n" + "=" * (22 + 20 * len(cols)))
    print(f"{'Metric':<22}" + "".join(f"{c:>20}" for c in cols))
    print("-" * (22 + 20 * len(cols)))
    for m in metric_keys:
        row = "".join(f"{results[c][m]:>20.3f}" for c in cols)
        print(f"{m:<22}{row}")
    print("=" * (22 + 20 * len(cols)))
    print("\nInterpretation: if Ensemble(Base+BMC) is BELOW Hybrid(learned) on")
    print("Mean F1 / Severe F1, the learned concat-MLP fusion adds value beyond a")
    print("trivial average -> answers reviewer concern C2.\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"seed": args.seed, "n_val": n_val, "results": results},
                  open(args.output, "w"), indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
