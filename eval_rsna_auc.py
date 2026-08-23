"""Standalone AUC + AUPRC evaluator for trained RSNA checkpoints.

Why this script exists
----------------------
The training scripts (train_rsna_baseline.py / train_rsna_attention.py /
train_rsna_hybrid.py) only persist argmax-based metrics (Acc / Recall /
Precision / F1) to ``best_metrics_*.txt``. They do NOT keep softmax
probabilities, so AUC and AUPRC cannot be computed from the saved logs alone.

This script reruns *inference only* on the same patient-level val split that
training used (``train_test_split(unique_patients, test_size=val_split,
random_state=seed)``), collects softmax probabilities, and writes per-class
AUC / AUPRC / Brier-score to a JSON next to the checkpoint.

Inference takes ~5-15 minutes on a single GPU per checkpoint — no training.

Usage
-----
    python3 eval_rsna_auc.py \\
        --model baseline \\
        --checkpoint checkpoints/fresh_baseline/best_model_baseline_full_e25.pth

    python3 eval_rsna_auc.py \\
        --model cbam \\
        --checkpoint checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth

    python3 eval_rsna_auc.py \\
        --model hybrid \\
        --checkpoint checkpoints/hybrid/best_model_hybrid_fixed_e7.pth

Output
------
JSON file at ``<checkpoint_dir>/auc_auprc_<checkpoint_stem>.json`` with:

    {
      "model": "hybrid",
      "checkpoint": "...",
      "val_split": 0.2,
      "seed": 42,
      "n_val_samples": <int>,
      "per_condition": {
        "spinal_canal": {
          "per_class": {
            "Normal/Mild":  {"auc": ..., "auprc": ..., "support": ...},
            "Moderate":     {"auc": ..., "auprc": ..., "support": ...},
            "Severe":       {"auc": ..., "auprc": ..., "support": ...}
          },
          "macro_auc":   ...,
          "macro_auprc": ...,
          "brier":       ...
        }, ...
      },
      "overall": {
        "macro_auc":   ...,
        "macro_auprc": ...,
        "popular_auprc": ...,    # Normal/Mild averaged over conditions
        "rare_auprc":    ...,    # mean(Moderate, Severe) averaged over conditions
        "severe_auprc":  ...     # Severe averaged over conditions (clinical priority)
      }
    }

Probabilities are also dumped to ``probs_<stem>.npz`` for downstream analysis.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)


CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=["baseline", "cbam", "hybrid"],
                   help="Architecture used for the checkpoint")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to the .pth checkpoint")
    p.add_argument("--data-dir", default="rsna_preprocessed",
                   help="Preprocessed RSNA data directory")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Patient-level val split (matches training)")
    p.add_argument("--seed", type=int, default=42,
                   help="random_state for train_test_split (matches training)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Inference batch size")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers")
    # Hybrid-specific
    p.add_argument("--cbam-checkpoint", default=None, type=str,
                   help="(hybrid only) frozen CBAM checkpoint path. Defaults to "
                        "fresh_cbam/best_model_attention_sqrt_cw_e20.pth")
    p.add_argument("--slice-strategy", default="static", choices=["static", "dynamic"],
                   help="(hybrid only) slice selection strategy used at training time")
    p.add_argument("--output", default=None, type=str,
                   help="Override output JSON path. Default: alongside checkpoint")
    p.add_argument("--save-probs", action="store_true", default=True,
                   help="Also save raw softmax probabilities to .npz")
    return p.parse_args()


# --------------------------------------------------------------------------
# Val dataset + loader (same patient-level split as training)
# --------------------------------------------------------------------------

def build_val_loader(data_dir: str, val_split: float, seed: int,
                     batch_size: int, num_workers: int) -> Tuple[DataLoader, int]:
    from rsna_preprocessed_dataloader import RSNAPreprocessedDataset

    full = RSNAPreprocessedDataset(data_dir=data_dir, split="train", transform=None)
    unique_patients = full.metadata["study_id"].unique()
    _, val_patients = train_test_split(
        unique_patients, test_size=val_split, random_state=seed,
    )
    val_indices = full.metadata[full.metadata["study_id"].isin(val_patients)].index.tolist()
    val_subset = Subset(full, val_indices)

    loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, len(val_subset)


# --------------------------------------------------------------------------
# Inference per architecture — returns per-condition (probs, labels) tensors
# --------------------------------------------------------------------------

def _collect_init() -> Dict[str, Dict[str, List]]:
    return {c: {"probs": [], "labels": []} for c in CONDITIONS}


def infer_baseline_or_cbam(model, loader, device) -> Dict[str, Dict[str, np.ndarray]]:
    """Both baseline and CBAM use linear-head architecture; same inference path."""
    model.eval()
    buf = _collect_init()
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Inference", leave=False):
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]
            outputs = model(volumes)  # dict[str, [B, 3]]
            for cond in CONDITIONS:
                probs = F.softmax(outputs[cond], dim=1).cpu().numpy()  # [B, 3]
                buf[cond]["probs"].append(probs)
                buf[cond]["labels"].append(labels[cond].numpy())
    return _stack(buf)


def infer_hybrid(model, loader, text_db, device) -> Dict[str, Dict[str, np.ndarray]]:
    model.eval()
    buf = _collect_init()
    scale = model.logit_scale.exp().clamp(max=100.0)
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Inference", leave=False):
            volumes = volumes.unsqueeze(1).to(device)
            image_emb = model.encode_image(volumes)  # [B, 512]
            for cond in CONDITIONS:
                logits = scale * (image_emb @ text_db[cond].T)  # [B, 3]
                probs = F.softmax(logits, dim=1).cpu().numpy()
                buf[cond]["probs"].append(probs)
                buf[cond]["labels"].append(labels[cond].numpy())
    return _stack(buf)


def _stack(buf: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, np.ndarray]]:
    return {c: {"probs": np.concatenate(buf[c]["probs"], axis=0),
                "labels": np.concatenate(buf[c]["labels"], axis=0)}
            for c in CONDITIONS}


# --------------------------------------------------------------------------
# Metrics: per-class AUC + AUPRC + Brier (one-vs-rest)
# --------------------------------------------------------------------------

# Per-class AUC/AUPRC/Brier computation lives in spinenet.auc_metrics so the
# training scripts and this standalone eval share a single implementation.


# --------------------------------------------------------------------------
# Pretty print
# --------------------------------------------------------------------------

def print_summary(payload: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  AUC / AUPRC summary — model={payload['model']}")
    print(f"  checkpoint: {payload['checkpoint']}")
    print(f"  n_val_samples: {payload['n_val_samples']}")
    print("=" * 78)

    for cond in CONDITIONS:
        m = payload["per_condition"][cond]
        print(f"\n{cond} (n_valid={m['n_valid']}):")
        print(f"  {'Class':<14} {'AUC':>7} {'AUPRC':>8} {'Brier':>8} {'Support':>8}")
        print(f"  {'-' * 50}")
        for cls in CLASS_NAMES:
            r = m["per_class"][cls]
            auc = "  nan " if np.isnan(r["auc"]) else f"{r['auc']:7.3f}"
            ap = "  nan  " if np.isnan(r["auprc"]) else f"{r['auprc']:8.3f}"
            br = "  nan  " if np.isnan(r["brier"]) else f"{r['brier']:8.3f}"
            print(f"  {cls:<14} {auc} {ap} {br} {r['support']:>8d}")
        print(f"  {'macro':<14} {m['macro_auc']:7.3f} {m['macro_auprc']:8.3f}")

    o = payload["overall"]
    print("\n" + "-" * 78)
    print("OVERALL (averaged across 3 conditions)")
    print("-" * 78)
    print(f"  macro AUC       : {o['macro_auc']:.3f}")
    print(f"  macro AUPRC     : {o['macro_auprc']:.3f}")
    print(f"  popular AUPRC   : {o['popular_auprc']:.3f}   (Normal/Mild only)")
    print(f"  rare    AUPRC   : {o['rare_auprc']:.3f}   (Moderate + Severe)")
    print(f"  Severe  AUPRC   : {o['severe_auprc']:.3f}   (clinical priority)")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = _pick_device()
    print(f"Device: {device}")

    print(f"\n[1/4] Building val loader (val_split={args.val_split}, seed={args.seed})...")
    val_loader, n_val = build_val_loader(
        args.data_dir, args.val_split, args.seed, args.batch_size, args.num_workers,
    )
    print(f"  ✓ {n_val} val samples")

    print(f"\n[2/4] Loading model + checkpoint...")
    if args.model in ("baseline", "cbam"):
        if args.model == "baseline":
            from spinenet.models.grading_baseline import GradingModelBaseline
            model = GradingModelBaseline(format="rsna")
        else:
            from spinenet.models.grading_attention import GradingModelWithCBAM
            model = GradingModelWithCBAM(format="rsna", use_cbam=True)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device)
    else:  # hybrid
        from spinenet.models.grading_hybrid import SpineNetHybrid
        cbam_ckpt = args.cbam_checkpoint or "checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth"
        model = SpineNetHybrid(
            cbam_checkpoint_path=cbam_ckpt,
            biomedclip_device=str(device),
            slice_strategy=args.slice_strategy,
        ).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model loaded ({n_train:,} trainable params)")

    print(f"\n[3/4] Running inference...")
    t0 = time.perf_counter()
    if args.model == "hybrid":
        from train_rsna_hybrid import build_text_database
        text_db = build_text_database(model, device)
        per_cond_data = infer_hybrid(model, val_loader, text_db, device)
    else:
        per_cond_data = infer_baseline_or_cbam(model, val_loader, device)
    elapsed = time.perf_counter() - t0
    throughput = n_val / elapsed
    print(f"  ✓ Inference done in {elapsed:.1f}s ({throughput:.1f} samples/s)")

    print(f"\n[4/4] Computing AUC + AUPRC + Brier...")
    probs_dict = {cond: per_cond_data[cond]["probs"] for cond in CONDITIONS}
    labels_dict = {cond: per_cond_data[cond]["labels"] for cond in CONDITIONS}
    per_condition = compute_auc_auprc_per_condition(probs_dict, labels_dict, CLASS_NAMES)
    # ``per_condition[cond]`` lacks the ``n_valid`` field we used in pretty-print;
    # back-fill it so print_summary keeps working.
    for cond in CONDITIONS:
        per_condition[cond]["n_valid"] = int((labels_dict[cond] != -1).sum())
    overall = aggregate_overall_auprc(per_condition, CLASS_NAMES)

    payload = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "val_split": args.val_split,
        "seed": args.seed,
        "n_val_samples": int(n_val),
        "inference_seconds": float(elapsed),
        "inference_throughput_samples_per_sec": float(throughput),
        "per_condition": per_condition,
        "overall": overall,
    }

    # ---- save JSON
    ckpt_path = Path(args.checkpoint)
    out_path = (Path(args.output) if args.output
                else ckpt_path.parent / f"auc_auprc_{ckpt_path.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  ✓ JSON: {out_path}")

    # ---- save raw probs
    if args.save_probs:
        npz_path = ckpt_path.parent / f"probs_{ckpt_path.stem}.npz"
        np.savez_compressed(npz_path, **{
            f"{cond}_probs": per_cond_data[cond]["probs"]
            for cond in CONDITIONS
        }, **{
            f"{cond}_labels": per_cond_data[cond]["labels"]
            for cond in CONDITIONS
        })
        print(f"  ✓ Probs: {npz_path}")

    print_summary(payload)


if __name__ == "__main__":
    main()
