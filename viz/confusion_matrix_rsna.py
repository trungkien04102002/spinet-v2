"""
Confusion matrices for Hybrid v3 on RSNA val split.

Runs inference on the full RSNA val set (random_state=42, test_size=0.2),
collects argmax predictions per condition, and plots a 1x3 row of confusion
matrices (Spinal canal / Left foraminal / Right foraminal). Each cell shows
absolute count and the column-normalized recall % (true class -> predicted).

Output:
    experiments/v3_20260503/figures/confusion_matrix_<model>.png
    experiments/v3_20260503/figures/confusion_matrix_<model>.csv  (raw counts)

Usage:
    python3 viz/confusion_matrix_rsna.py --model hybrid
    python3 viz/confusion_matrix_rsna.py --model cbam
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset  # noqa: E402
from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402
from spinenet.models.grading_hybrid import SpineNetHybrid  # noqa: E402

CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
CLASS_NAMES = ["Normal", "Moderate", "Severe"]

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
def predict_cbam(model, loader, device):
    preds = {c: [] for c in CONDITIONS}
    gts = {c: [] for c in CONDITIONS}
    for i, (vol, labels) in enumerate(loader):
        vol = vol.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]
        out = model(vol)
        for c in CONDITIONS:
            p = out[c].argmax(dim=-1).cpu().numpy()
            g = labels[c].numpy()
            preds[c].append(p)
            gts[c].append(g)
        if (i + 1) % 10 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    for c in CONDITIONS:
        preds[c] = np.concatenate(preds[c])
        gts[c] = np.concatenate(gts[c])
    return preds, gts


@torch.no_grad()
def predict_hybrid(model, loader, device):
    text_dbs = {
        c: model.encode_text(RSNA_PROMPTS[c]).to(device) for c in CONDITIONS
    }
    preds = {c: [] for c in CONDITIONS}
    gts = {c: [] for c in CONDITIONS}
    for i, (vol, labels) in enumerate(loader):
        vol = vol.unsqueeze(1).to(device)
        image_emb = model.encode_image(vol)  # [B, 512]
        scale = model.logit_scale.exp().clamp(max=100.0)
        for c in CONDITIONS:
            logits = scale * (image_emb @ text_dbs[c].T)
            p = logits.argmax(dim=-1).cpu().numpy()
            g = labels[c].numpy()
            preds[c].append(p)
            gts[c].append(g)
        if (i + 1) % 5 == 0:
            print(f"    batch {i + 1}/{len(loader)}")
    for c in CONDITIONS:
        preds[c] = np.concatenate(preds[c])
        gts[c] = np.concatenate(gts[c])
    return preds, gts


def plot_confusion_matrices(gts, preds, model_name: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    rows = []
    for ax, cond in zip(axes, CONDITIONS):
        # Drop missing labels (-1) — RSNA uses -1 for missing.
        mask = gts[cond] >= 0
        y_true = gts[cond][mask]
        y_pred = preds[cond][mask]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(cond.replace("_", " ").title(), fontsize=11)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=20)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                txt = f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
        for true_cls in range(3):
            row = {
                "model": model_name,
                "condition": cond,
                "true_class": CLASS_NAMES[true_cls],
                "n_true": int(cm[true_cls].sum()),
            }
            for pred_cls in range(3):
                row[f"pred_{CLASS_NAMES[pred_cls]}"] = int(cm[true_cls, pred_cls])
            rows.append(row)

    fig.suptitle(
        f"Confusion matrices on RSNA val ({model_name}) — counts and row-normalized recall",
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
    ap.add_argument("--model", choices=["cbam", "hybrid"], default="hybrid")
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "experiments/v3_20260503/figures")
    args = ap.parse_args()

    device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Model: {args.model}")

    print("\n[1/3] Building val loader...")
    loader = build_val_loader(args.data_dir, args.batch_size, args.num_workers)

    if args.model == "cbam":
        print("\n[2/3] Loading CBAM-only model...")
        model = load_cbam(args.cbam_ckpt, device)
        print("\n[3/3] Predicting...")
        preds, gts = predict_cbam(model, loader, device)
    else:
        print("\n[2/3] Loading Hybrid model...")
        model = load_hybrid(args.cbam_ckpt, args.hybrid_ckpt, device)
        print("\n[3/3] Predicting...")
        preds, gts = predict_hybrid(model, loader, device)

    out_path = args.output_dir / f"confusion_matrix_{args.model}.png"
    plot_confusion_matrices(gts, preds, args.model, out_path)
    print("\n[done]")


if __name__ == "__main__":
    main()
