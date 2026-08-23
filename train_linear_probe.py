"""Linear probe comparison: BiomedCLIP vs ImageNet ViT on RSNA IVD grading.

Why this experiment exists
--------------------------
Advisor's challenge: "BiomedCLIP could just be doing what any vision encoder
would do. Prove its medical pretraining adds spine-relevant features."

This script tests two frozen 2D encoders, each followed by a linear head, on
the same patient-level RSNA val split as the main training scripts:

    --backbone biomedclip       :: BiomedCLIP (Zhang 2023, PMC-15M medical)
    --backbone imagenet_vit     :: ViT-B/16 (ImageNet-1K, generic vision)

If BiomedCLIP linear-probe > ImageNet linear-probe on RSNA F1/AUPRC, BiomedCLIP
has spine-relevant features beyond generic vision. If they tie, BiomedCLIP's
medical pretraining adds nothing for this task and we cannot defend its use.
Either result is publishable evidence; both directly answer the advisor.

Method (kept deliberately simple for a quick, reproducible ablation)
-------------------------------------------------------------------
1. For each sample (9, 112, 224) volume, take the 3 center slices (3,4,5),
   encode each with the frozen 2D encoder, mean-pool the slice embeddings.
2. Multi-task linear head: one ``Linear(D, 3)`` per condition, trained jointly
   with cross-entropy and ``ignore_index=-1`` for missing labels.
3. Same train/val split (patient-level, ``random_state=42``) as training.
4. Compute Acc/Precision/Recall/F1 + AUC/AUPRC + Brier; write to
   ``best_metrics.{json,txt}`` via the shared ``MetricsLogger``.

Usage
-----
    python3 train_linear_probe.py --backbone biomedclip
    python3 train_linear_probe.py --backbone imagenet_vit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.auc_metrics import (
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)
from spinenet.metrics_logger import MetricsLogger


CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
NUM_CLASSES = 3
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--backbone", required=True, choices=["biomedclip", "imagenet_vit"],
                   help="Which frozen encoder to probe")
    p.add_argument("--data-dir", default="rsna_preprocessed")
    p.add_argument("--save-dir", default="checkpoints/linear_probe")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30,
                   help="Linear head training epochs (encoder frozen the whole time)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--num-slices", type=int, default=3,
                   help="Center slices to encode per volume (3 = slices 3,4,5)")
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Encoders — both expose ``encode(slices_rgb_224) -> [B, K, D]``
# ---------------------------------------------------------------------------

class BiomedCLIPProbe(nn.Module):
    """Frozen BiomedCLIP image encoder. Output dim 512."""
    def __init__(self, device: torch.device):
        super().__init__()
        from spinenet.models.biomedclip_wrapper import BiomedCLIPWrapper
        self.wrapper = BiomedCLIPWrapper(device=str(device))
        self.feat_dim = 512

    @torch.no_grad()
    def encode(self, slices: torch.Tensor) -> torch.Tensor:
        """slices: [B, K, H=112, W=224] grayscale in [0,1] -> [B, K, 512]."""
        B, K, H, W = slices.shape
        flat = slices.reshape(B * K, H, W)
        rgb = torch.stack(
            [self.wrapper.preprocess_slice(flat[i]) for i in range(B * K)], dim=0
        ).to(slices.device)
        embs = self.wrapper.encode_image(rgb)
        return embs.reshape(B, K, -1)


class ImageNetViTProbe(nn.Module):
    """Frozen ImageNet ViT-B/16. Output dim 768 (CLS token).

    Uses torchvision's pretrained weights. Replaces the classification head
    with identity so we get the pooled CLS token directly.
    """
    def __init__(self, device: torch.device):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights).to(device).eval()
        self.model.heads = nn.Identity().to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.feat_dim = 768
        self._device = device

    @torch.no_grad()
    def encode(self, slices: torch.Tensor) -> torch.Tensor:
        """slices: [B, K, H=112, W=224] grayscale in [0,1] -> [B, K, 768]."""
        B, K, H, W = slices.shape
        flat = slices.reshape(B * K, 1, H, W)
        # ImageNet ViT expects 3 x 224 x 224, normalized with ImageNet stats.
        rgb = flat.expand(-1, 3, -1, -1)
        rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(rgb.device)
        std = IMAGENET_STD.view(1, 3, 1, 1).to(rgb.device)
        rgb = (rgb - mean) / std
        embs = self.model(rgb)  # [B*K, 768]
        return embs.reshape(B, K, -1)


# ---------------------------------------------------------------------------
# Head: shared backbone -> mean-pool slice features -> 3 linear classifiers
# ---------------------------------------------------------------------------

class LinearProbeHead(nn.Module):
    """One ``Linear(D, 3)`` per condition. Encoder is frozen; only this trains."""
    def __init__(self, feat_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.heads = nn.ModuleDict({
            cond: nn.Linear(feat_dim, num_classes) for cond in CONDITIONS
        })

    def forward(self, image_emb: torch.Tensor) -> dict:
        return {cond: self.heads[cond](image_emb) for cond in CONDITIONS}


# ---------------------------------------------------------------------------
# Center-slice extraction
# ---------------------------------------------------------------------------

def take_center_slices(volumes: torch.Tensor, k: int) -> torch.Tensor:
    """volumes: [B, 9, 112, 224] -> [B, k, 112, 224] picking the k center slices."""
    n_total = volumes.shape[1]
    start = (n_total - k) // 2
    return volumes[:, start:start + k]


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def encode_features(encoder, loader, device, num_slices: int):
    """One frozen forward pass over the entire split. Mean-pool over slices.

    Returns (features [N, D], labels {cond: [N]}).
    """
    feats, lbls = [], {c: [] for c in CONDITIONS}
    encoder.eval()
    for volumes, labels in tqdm(loader, desc="Encoding", leave=False):
        volumes = volumes.to(device)  # [B, 9, 112, 224]
        slices = take_center_slices(volumes, num_slices)
        slice_emb = encoder.encode(slices)  # [B, K, D]
        pooled = slice_emb.mean(dim=1)  # [B, D]
        feats.append(pooled.cpu())
        for c in CONDITIONS:
            lbls[c].append(labels[c].numpy())
    return (torch.cat(feats, dim=0).numpy(),
            {c: np.concatenate(lbls[c], axis=0) for c in CONDITIONS})


def train_linear_head(head: LinearProbeHead, X_train, y_train_dict,
                      X_val, y_val_dict, device, epochs: int, lr: float,
                      batch_size: int):
    """Train the linear heads on cached features. Encoder already done."""
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(ignore_index=-1)

    X_train_t = torch.from_numpy(X_train).float().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_train_t = {c: torch.from_numpy(y_train_dict[c]).long().to(device) for c in CONDITIONS}
    y_val_t = {c: torch.from_numpy(y_val_dict[c]).long().to(device) for c in CONDITIONS}

    n_train = X_train_t.shape[0]
    print(f"\nTraining linear head: {n_train} train, {X_val_t.shape[0]} val, "
          f"{epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n_train, device=device)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x = X_train_t[idx]
            opt.zero_grad()
            outs = head(x)
            loss = sum(crit(outs[c], y_train_t[c][idx]) for c in CONDITIONS) / len(CONDITIONS)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.shape[0]

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            train_loss = total_loss / n_train
            head.eval()
            with torch.no_grad():
                val_outs = head(X_val_t)
            sf1 = []
            for c in CONDITIONS:
                preds = val_outs[c].argmax(dim=1).cpu().numpy()
                y = y_val_dict[c]
                m = y != -1
                if m.sum() > 0:
                    _, _, f1, _ = precision_recall_fscore_support(
                        y[m], preds[m], labels=[0, 1, 2], average=None, zero_division=0,
                    )
                    sf1.append(f1[2])  # Severe F1
            print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}  "
                  f"avg_severe_F1={np.mean(sf1):.4f}")

    return head


def eval_linear_head(head, X_val, y_val_dict, device):
    head.eval()
    X_val_t = torch.from_numpy(X_val).float().to(device)
    with torch.no_grad():
        outs = head(X_val_t)

    val_accuracies = {}
    per_class_metrics = {}
    probs_dict, labels_dict = {}, {}

    for c in CONDITIONS:
        logits = outs[c]
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y = y_val_dict[c]
        m = y != -1

        val_accuracies[c] = float(accuracy_score(y[m], preds[m])) if m.sum() else 0.0
        if m.sum():
            P, R, F1, S = precision_recall_fscore_support(
                y[m], preds[m], labels=[0, 1, 2], average=None, zero_division=0,
            )
            per_class_metrics[c] = {
                "precision": P, "recall": R, "f1": F1, "support": S,
            }
        probs_dict[c] = probs
        labels_dict[c] = y

    auc_auprc = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    overall = aggregate_overall_auprc(auc_auprc)
    return val_accuracies, per_class_metrics, auc_auprc, overall


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_loaders(args) -> Tuple[DataLoader, DataLoader, int, int]:
    full = RSNAPreprocessedDataset(data_dir=args.data_dir, split="train", transform=None)
    unique = full.metadata["study_id"].unique()
    train_p, val_p = train_test_split(unique, test_size=args.val_split, random_state=args.seed)
    tr_idx = full.metadata[full.metadata["study_id"].isin(train_p)].index.tolist()
    va_idx = full.metadata[full.metadata["study_id"].isin(val_p)].index.tolist()
    tr_loader = DataLoader(Subset(full, tr_idx), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=False)
    va_loader = DataLoader(Subset(full, va_idx), batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=False)
    return tr_loader, va_loader, len(tr_idx), len(va_idx)


def main():
    args = parse_args()
    device = pick_device()
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")

    print("\n[1/4] Building loaders...")
    tr_loader, va_loader, n_train, n_val = build_loaders(args)
    print(f"  ✓ train={n_train}  val={n_val}")

    print(f"\n[2/4] Loading frozen encoder...")
    if args.backbone == "biomedclip":
        encoder = BiomedCLIPProbe(device)
    else:
        encoder = ImageNetViTProbe(device)
    print(f"  ✓ {args.backbone}: feat_dim={encoder.feat_dim}")

    print(f"\n[3/4] Extracting features (one frozen pass each split)...")
    t0 = time.perf_counter()
    X_train, y_train = encode_features(encoder, tr_loader, device, args.num_slices)
    X_val, y_val = encode_features(encoder, va_loader, device, args.num_slices)
    elapsed = time.perf_counter() - t0
    print(f"  ✓ X_train={X_train.shape}  X_val={X_val.shape}  ({elapsed:.1f}s)")

    print(f"\n[4/4] Training linear head + evaluating...")
    head = LinearProbeHead(encoder.feat_dim)
    head = train_linear_head(head, X_train, y_train, X_val, y_val,
                             device, args.epochs, args.lr, args.batch_size)
    val_acc, per_class, auc_auprc, overall = eval_linear_head(head, X_val, y_val, device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"linear_probe_{args.backbone}"

    severe_f1 = float(np.mean([
        per_class[c]["f1"][2] for c in per_class
        if "f1" in per_class[c] and len(per_class[c]["f1"]) >= 3
    ])) if per_class else float("nan")

    logger = MetricsLogger(save_dir=save_dir, prefix=prefix)
    logger.save_best(
        epoch=args.epochs, train_loss=float("nan"), val_loss=float("nan"),
        val_accuracies=val_acc, per_class_metrics=per_class,
        avg_severe_f1=severe_f1, auc_auprc_metrics=auc_auprc,
        auc_auprc_overall=overall,
        extra={
            "backbone": args.backbone, "feat_dim": encoder.feat_dim,
            "n_train": int(n_train), "n_val": int(n_val),
            "encode_seconds": float(elapsed), "args": vars(args),
        },
    )

    print(f"\n{'=' * 70}\nDONE — {args.backbone} linear probe\n{'=' * 70}")
    for c in CONDITIONS:
        print(f"  {c:18s} acc={val_acc[c]:.3f}")
    print(f"\nOverall:")
    print(f"  Severe F1 (mean):  {severe_f1:.3f}")
    print(f"  Macro AUPRC:       {overall.get('macro_auprc', float('nan')):.3f}")
    print(f"  Severe AUPRC:      {overall.get('severe_auprc', float('nan')):.3f}")
    print(f"\nSaved to: {save_dir}/{prefix}_best_metrics.{{json,txt}}")


if __name__ == "__main__":
    main()
