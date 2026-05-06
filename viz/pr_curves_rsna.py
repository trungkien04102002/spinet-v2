"""
Precision-Recall curves for the Severe class on RSNA val.

Runs inference once per checkpoint, collects per-sample Severe-class softmax
probability for each condition, and plots a 1x3 row of PR curves overlaying
all checkpoints provided. Intended for the CBAM-only vs Hybrid v3 comparison
(extend with --baseline-ckpt if a v3 baseline ckpt becomes available).

Output:
    experiments/v3_20260503/figures/pr_curves_severe.png
    experiments/v3_20260503/figures/pr_curves_severe.csv  (raw AUPRC table)

Usage:
    python3 viz/pr_curves_rsna.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset  # noqa: E402
from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402
from spinenet.models.grading_hybrid import SpineNetHybrid  # noqa: E402

CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
SEVERE_CLASS = 2

RSNA_PROMPTS = {
    "spinal_canal": [
        "normal or mild spinal canal stenosis",
        "moderate spinal canal stenosis",
        "severe spinal canal stenosis",
    ],
    "left_foraminal": [
        "normal or mild left neural foraminal narrowing",
        "moderate left neural foraminal narrowing",
        "severe left neural foraminal narrowing",
    ],
    "right_foraminal": [
        "normal or mild right neural foraminal narrowing",
        "moderate right neural foraminal narrowing",
        "severe right neural foraminal narrowing",
    ],
}


def build_val_loader(data_dir: Path, batch_size: int, num_workers: int):
    full = RSNAPreprocessedDataset(data_dir=str(data_dir), split="train")
    patients = full.metadata["study_id"].unique()
    _, val_pat = train_test_split(patients, test_size=0.2, random_state=42)
    val_idx = full.metadata[full.metadata["study_id"].isin(val_pat)].index.tolist()
    val = Subset(full, val_idx)
    loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print(f"  Val: {len(val)} samples ({len(val_pat)} patients)")
    return loader


def load_cbam(ckpt: Path, device: torch.device) -> nn.Module:
    model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_hybrid(cbam_ckpt: Path, hybrid_ckpt: Path, device: torch.device):
    model = SpineNetHybrid(
        cbam_checkpoint_path=str(cbam_ckpt),
        biomedclip_device=str(device),
        slice_strategy="static",
    )
    state = torch.load(hybrid_ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def collect_severe_probs_cbam(model, loader, device):
    probs = {c: [] for c in CONDITIONS}
    gts = {c: [] for c in CONDITIONS}
    for i, (vol, labels) in enumerate(loader):
        vol = vol.unsqueeze(1).to(device)
        out = model(vol)
        for c in CONDITIONS:
            p = torch.softmax(out[c], dim=-1)[:, SEVERE_CLASS].cpu().numpy()
            g = labels[c].numpy()
            probs[c].append(p)
            gts[c].append(g)
        if (i + 1) % 20 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    for c in CONDITIONS:
        probs[c] = np.concatenate(probs[c])
        gts[c] = np.concatenate(gts[c])
    return probs, gts


@torch.no_grad()
def collect_severe_probs_hybrid(model, loader, device):
    text_dbs = {c: model.encode_text(RSNA_PROMPTS[c]).to(device) for c in CONDITIONS}
    probs = {c: [] for c in CONDITIONS}
    gts = {c: [] for c in CONDITIONS}
    for i, (vol, labels) in enumerate(loader):
        vol = vol.unsqueeze(1).to(device)
        image_emb = model.encode_image(vol)
        scale = model.logit_scale.exp().clamp(max=100.0)
        for c in CONDITIONS:
            logits = scale * (image_emb @ text_dbs[c].T)
            p = torch.softmax(logits, dim=-1)[:, SEVERE_CLASS].cpu().numpy()
            g = labels[c].numpy()
            probs[c].append(p)
            gts[c].append(g)
        if (i + 1) % 10 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    for c in CONDITIONS:
        probs[c] = np.concatenate(probs[c])
        gts[c] = np.concatenate(gts[c])
    return probs, gts


def plot_pr_curves(model_results: dict, out_path: Path):
    """
    model_results: {model_name: (probs_dict, gts_dict)}
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    rows = []
    colors = {"cbam": "C0", "hybrid": "C1", "baseline": "C2"}
    for ax, cond in zip(axes, CONDITIONS):
        for name, (probs, gts) in model_results.items():
            mask = gts[cond] >= 0
            y_true = (gts[cond][mask] == SEVERE_CLASS).astype(int)
            y_score = probs[cond][mask]
            if y_true.sum() == 0:
                print(f"  no Severe positives for {cond} — skipping")
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            label = f"{name} (AUPRC={ap:.3f})"
            ax.plot(rec, prec, color=colors.get(name, None), label=label, lw=1.6)
            rows.append({"model": name, "condition": cond,
                         "n_pos": int(y_true.sum()), "n_total": int(len(y_true)),
                         "auprc": float(ap)})
        ax.set_title(cond.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Recall (Severe)")
        ax.set_ylabel("Precision (Severe)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Precision-Recall curves for Severe class on RSNA val",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  saved {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cbam-ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/v3_20260503/cbam_best.pth",
    )
    ap.add_argument(
        "--hybrid-ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/v3_20260503/hybrid_best.pth",
    )
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "rsna_preprocessed")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--skip-hybrid", action="store_true",
                    help="Only run CBAM-only (faster sanity check)")
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/v3_20260503/figures/pr_curves_severe.png",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    print("\n[1/N] Building val loader...")
    loader = build_val_loader(args.data_dir, args.batch_size, args.num_workers)

    results = {}

    print("\n[CBAM] Loading + predicting...")
    cbam = load_cbam(args.cbam_ckpt, device)
    results["cbam"] = collect_severe_probs_cbam(cbam, loader, device)
    del cbam

    if not args.skip_hybrid:
        print("\n[Hybrid] Loading + predicting...")
        hybrid = load_hybrid(args.cbam_ckpt, args.hybrid_ckpt, device)
        results["hybrid"] = collect_severe_probs_hybrid(hybrid, loader, device)
        del hybrid

    print("\n[plot]")
    plot_pr_curves(results, args.output)
    print("[done]")


if __name__ == "__main__":
    main()
