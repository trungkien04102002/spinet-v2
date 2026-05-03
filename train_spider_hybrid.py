"""
Phase 4 — Train Hybrid (CBAM + BiomedCLIP) on SPIDER with text-aligned heads.

Same architecture as train_rsna_hybrid.py but:
  - 4 SPIDER conditions instead of 3 RSNA conditions
  - 13 SPIDER text prompts (pfirrmann 5 + modic 4 + disc_narrowing 2 + spondylolisthesis 2)
  - Init from RSNA Hybrid checkpoint (carries projection MLP + slice_pool + logit_scale)
  - Preserves zero-shot capability: model still uses cosine sim with text prompts;
    new SPIDER labels can be added at eval time by encoding new prompts.

Usage:
    python3 train_spider_hybrid.py \\
        --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid.pth \\
        --cbam-checkpoint checkpoints/best_model_attention.pth \\
        --epochs 15 --batch-size 16 --lr 1e-4
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from spider_dataloader import SPIDERDataset
from spinenet.models.grading_hybrid import SpineNetHybrid


def _slim_state_dict(state):
    """Drop frozen biomedclip.* keys before saving — they're re-loadable from
    the HF hub and would otherwise add ~750 MB to every checkpoint file."""
    return {k: v for k, v in state.items() if not k.startswith("biomedclip.")}


# ============================ SPIDER text prompts ============================
# Index = class label (0-based). Pfirrmann labels are stored 0-4 in the dataloader
# (originally 1-5 in the CSV).
SPIDER_PROMPTS = {
    "pfirrmann": [
        "pfirrmann grade 1 normal disc",
        "pfirrmann grade 2 mild disc degeneration",
        "pfirrmann grade 3 moderate disc degeneration",
        "pfirrmann grade 4 severe disc degeneration",
        "pfirrmann grade 5 end-stage disc degeneration",
    ],
    "modic": [
        "no modic changes",
        "modic type 1 endplate inflammation",
        "modic type 2 fatty endplate degeneration",
        "modic type 3 sclerotic endplate changes",
    ],
    "disc_narrowing": [
        "normal disc height",
        "disc space narrowing",
    ],
    "spondylolisthesis": [
        "no spondylolisthesis",
        "spondylolisthesis with vertebral slippage",
    ],
}
SPIDER_CONDITIONS = list(SPIDER_PROMPTS.keys())
SPIDER_NUM_CLASSES = {c: len(SPIDER_PROMPTS[c]) for c in SPIDER_CONDITIONS}
DISPLAY_NAMES = {
    "pfirrmann": "Pfirrmann",
    "modic": "Modic",
    "disc_narrowing": "Disc Narrowing",
    "spondylolisthesis": "Spondylolisthesis",
}
CLASS_NAMES = {
    "pfirrmann": ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"],
    "modic": ["Type 0", "Type 1", "Type 2", "Type 3"],
    "disc_narrowing": ["No", "Yes"],
    "spondylolisthesis": ["No", "Yes"],
}
PROMPT_TEMPLATE = "a magnetic resonance image of {label}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid-checkpoint", type=str, required=True,
                   help="RSNA-trained Hybrid checkpoint (provides image_projection, slice_pool, logit_scale)")
    p.add_argument("--cbam-checkpoint", type=str, required=True,
                   help="RSNA CBAM checkpoint to init the 3D backbone")
    p.add_argument("--data-dir", type=str, default="spider")
    p.add_argument("--modality", type=str, default="t2")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-freq", type=int, default=5,
                   help="Save periodic checkpoint every N epochs")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--unfreeze-cbam", action="store_true",
                   help="Also fine-tune the CBAM 3D backbone (default: keep frozen)")
    p.add_argument("--loss", type=str, default="weighted",
                   choices=["weighted", "standard"])
    p.add_argument("--slice-strategy", type=str, default="static",
                   choices=["static", "dynamic"])
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints_spider")
    p.add_argument("--metrics-dir", type=str, default="experiments/spider_phase4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablate-branch", type=str, default="none",
                   choices=["none", "cbam_only", "biomedclip_only"],
                   help='Zero out one branch for component ablation. '
                        '"cbam_only" = drop BiomedCLIP image features; '
                        '"biomedclip_only" = drop CBAM 3D features.')
    return p.parse_args()


def build_text_database(model: SpineNetHybrid, device: torch.device):
    """
    Returns dict: condition -> [num_classes, 512] L2-normalized tensor on device.
    Encoded once before training and reused every batch.
    """
    db = {}
    for cond in SPIDER_CONDITIONS:
        texts = [PROMPT_TEMPLATE.format(label=p) for p in SPIDER_PROMPTS[cond]]
        with torch.no_grad():
            embs = model.encode_text(texts).to(device)
        db[cond] = embs
    return db


def compute_class_weights(dataset):
    """sqrt-inverse-frequency class weights per condition (normalized).

    Uses _resolve_get_labels to skip full .mha volume loads at startup.
    """
    from spinenet.augmentation import _resolve_get_labels

    counts = {c: torch.zeros(SPIDER_NUM_CLASSES[c]) for c in SPIDER_CONDITIONS}
    get_labels = _resolve_get_labels(dataset)

    for idx in range(len(dataset)):
        if get_labels is not None:
            labels = get_labels(idx)
        else:
            _, labels = dataset[idx]
        for c in SPIDER_CONDITIONS:
            counts[c][labels[c]] += 1

    weights = {}
    for c in SPIDER_CONDITIONS:
        w = 1.0 / counts[c].clamp(min=1).sqrt()
        w = w / w.sum() * SPIDER_NUM_CLASSES[c]
        weights[c] = w
    return weights, counts


def cosine_logits(image_emb: torch.Tensor, text_emb: torch.Tensor,
                  logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled cosine similarity logits.
    image_emb: [B, 512] L2-normalized
    text_emb:  [num_classes, 512] L2-normalized
    Returns: [B, num_classes]
    """
    cos = image_emb @ text_emb.T  # [B, num_classes]
    scale = torch.exp(logit_scale).clamp(max=100.0)
    return cos * scale


def evaluate(model: SpineNetHybrid, dataloader, text_db, criteria, device):
    model.eval()
    all_losses = {c: [] for c in SPIDER_CONDITIONS}
    all_preds  = {c: [] for c in SPIDER_CONDITIONS}
    all_labels = {c: [] for c in SPIDER_CONDITIONS}

    with torch.no_grad():
        for volumes, labels in tqdm(dataloader, desc="Validating", leave=False):
            volumes = volumes.unsqueeze(1).to(device)
            image_emb = model.encode_image(volumes)  # [B, 512]
            for c in SPIDER_CONDITIONS:
                logits = cosine_logits(image_emb, text_db[c], model.logit_scale)
                target = labels[c].to(device)
                loss = criteria[c](logits, target)
                all_losses[c].append(loss.item())
                all_preds[c].extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels[c].extend(labels[c].numpy())

    val_accuracies = {}
    per_class_metrics = {}
    for c in SPIDER_CONDITIONS:
        y_true = np.array(all_labels[c])
        y_pred = np.array(all_preds[c])
        val_accuracies[c] = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred,
            labels=list(range(SPIDER_NUM_CLASSES[c])),
            average=None, zero_division=0,
        )
        per_class_metrics[c] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }

    avg_loss = float(np.mean([np.mean(all_losses[c]) for c in SPIDER_CONDITIONS]))
    return avg_loss, val_accuracies, per_class_metrics


def print_per_class_metrics(per_class_metrics):
    print("\n" + "=" * 72)
    print("Per-Class Metrics (Precision / Recall / F1)")
    print("=" * 72)
    for c in SPIDER_CONDITIONS:
        m = per_class_metrics.get(c)
        if m is None:
            continue
        print(f"\n{DISPLAY_NAMES[c]} ({SPIDER_NUM_CLASSES[c]} classes):")
        print(f"  {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
        print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
        for i, name in enumerate(CLASS_NAMES[c]):
            print(f"  {name:<12} {m['precision'][i]:<10.3f} "
                  f"{m['recall'][i]:<10.3f} {m['f1'][i]:<10.3f} "
                  f"{int(m['support'][i]):<8}")
    print("=" * 72)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Tag for output filenames so frozen/unfreeze + multi-seed + ablation runs
    # don't overwrite each other.
    _tag_parts = ["unfreeze" if args.unfreeze_cbam else "frozen"]
    if args.seed != 42:
        _tag_parts.append(f"seed{args.seed}")
    if args.ablate_branch != "none":
        _tag_parts.append(args.ablate_branch)
    tag = "_".join(_tag_parts)

    print("=" * 72)
    print("Phase 4: Train Hybrid CBAM + BiomedCLIP on SPIDER")
    print("=" * 72)
    print(f"Device:               {device}")
    print(f"Hybrid checkpoint:    {args.hybrid_checkpoint}")
    print(f"CBAM checkpoint:      {args.cbam_checkpoint}")
    print(f"Modality:             {args.modality}")
    print(f"Conditions:           {SPIDER_CONDITIONS}")
    print(f"Unfreeze CBAM:        {args.unfreeze_cbam}")
    print(f"Loss mode:            {args.loss}")
    print(f"Output filename tag:  _{tag}")
    print("=" * 72)

    # ---- Dataset ----
    print("\n[1/5] Loading SPIDER dataset (training subset)...")
    full_dataset = SPIDERDataset(data_dir=args.data_dir, split="training",
                                 modality=args.modality)

    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=args.val_split,
        random_state=args.seed,
    )
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"  Train: {len(train_dataset)}    Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ---- Model ----
    print("\n[2/5] Building Hybrid model...")
    model = SpineNetHybrid(
        cbam_checkpoint_path=args.cbam_checkpoint,
        biomedclip_device=str(device),
        slice_strategy=args.slice_strategy,
        ablate_branch=args.ablate_branch,
    ).to(device)
    if args.ablate_branch != "none":
        print(f"  Ablation mode: {args.ablate_branch} "
              f"(other branch zeroed before fusion)")

    # Load Hybrid trainable weights (image_projection + slice_pool + logit_scale)
    print(f"  Loading Hybrid trainable weights from {args.hybrid_checkpoint}")
    ckpt = torch.load(args.hybrid_checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded. Missing keys (expected for frozen modules): {len(missing)}, "
          f"Unexpected: {len(unexpected)}")

    # SpineNetHybrid constructor freezes CBAM by default (requires_grad=False).
    # When --unfreeze-cbam is set, we must actively re-enable gradients for it
    # AND switch the module out of eval() so BatchNorm running stats update.
    if args.unfreeze_cbam:
        for p in model.cbam.parameters():
            p.requires_grad = True
        model.cbam.train()
        print("  CBAM backbone: TRAINABLE (gradients enabled)")
    else:
        for p in model.cbam.parameters():
            p.requires_grad = False
        model.cbam.eval()
        print("  CBAM backbone: FROZEN")
    # BiomedCLIP is always frozen (handled internally by SpineNetHybrid)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_train:,} / {n_total:,}")

    # ---- Text database ----
    print("\n[3/5] Building SPIDER text database...")
    text_db = build_text_database(model, device)
    for c in SPIDER_CONDITIONS:
        print(f"  {c:20s} -> {tuple(text_db[c].shape)}")

    # ---- Loss ----
    print("\n[4/5] Setting up losses...")
    if args.loss == "weighted":
        weights, counts = compute_class_weights(train_dataset)
        criteria = {c: nn.CrossEntropyLoss(weight=weights[c].to(device))
                    for c in SPIDER_CONDITIONS}
        for c in SPIDER_CONDITIONS:
            print(f"  {DISPLAY_NAMES[c]:<20s} counts={counts[c].int().tolist()}  "
                  f"weights={[f'{w:.2f}' for w in weights[c].tolist()]}")
    else:
        criteria = {c: nn.CrossEntropyLoss() for c in SPIDER_CONDITIONS}
        print("  Standard CrossEntropyLoss (no class weights)")

    # ---- Optimizer ----
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=args.lr, weight_decay=args.weight_decay)
    # PyTorch 2.x removed `verbose` arg.
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # ---- Training ----
    print("\n[5/5] Starting training...")
    print("=" * 72)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        # Keep BiomedCLIP and (optionally) CBAM in eval mode for BatchNorm stability
        model.biomedclip.eval()
        if not args.unfreeze_cbam:
            model.cbam.eval()

        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for volumes, labels in pbar:
            volumes = volumes.unsqueeze(1).to(device)
            image_emb = model.encode_image(volumes)

            losses = []
            for c in SPIDER_CONDITIONS:
                logits = cosine_logits(image_emb, text_db[c], model.logit_scale)
                target = labels[c].to(device)
                losses.append(criteria[c](logits, target))
            loss = sum(losses) / len(losses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / max(1, len(train_loader))

        # Validation
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, text_db, criteria, device,
        )
        scheduler.step(val_loss)

        mean_acc = float(np.mean(list(val_acc.values())))
        elapsed = time.time() - t0

        # Per-condition F1 macro (mean of per-class F1) — primary metric on
        # imbalanced classes where accuracy is misleading.
        f1_macro = {c: float(np.mean(per_class[c]['f1'])) for c in SPIDER_CONDITIONS}
        mean_f1 = float(np.mean(list(f1_macro.values())))

        print(f"\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")
        print(f"  Val per-condition (accuracy | F1 macro):")
        for c in SPIDER_CONDITIONS:
            print(f"    - {DISPLAY_NAMES[c]:<20s}  acc={val_acc[c]*100:6.2f}%  F1={f1_macro[c]:.3f}")
        print(f"    - {'Mean':<20s}  acc={mean_acc*100:6.2f}%  F1={mean_f1:.3f}")
        if (epoch + 1) % 5 == 0:
            print_per_class_metrics(per_class)

        # Track history (CSV row per epoch)
        row = {"epoch": epoch + 1, "train_loss": avg_train_loss,
               "val_loss": val_loss, "val_mean_acc": mean_acc,
               "val_mean_f1_macro": mean_f1}
        for c in SPIDER_CONDITIONS:
            row[f"val_acc_{c}"] = float(val_acc[c])
            row[f"val_f1_macro_{c}"] = f1_macro[c]
            for i, f in enumerate(per_class[c]["f1"]):
                row[f"f1_{c}_class{i}"] = float(f)
            for i, p in enumerate(per_class[c]["precision"]):
                row[f"precision_{c}_class{i}"] = float(p)
            for i, r in enumerate(per_class[c]["recall"]):
                row[f"recall_{c}_class{i}"] = float(r)
        history.append(row)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_path = ckpt_dir / f"best_model_hybrid_spider_{tag}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": _slim_state_dict(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracies": val_acc,
                "val_f1_macro": f1_macro,
                "per_class_metrics": {c: {k: v.tolist() if hasattr(v, "tolist") else v
                                          for k, v in m.items()}
                                      for c, m in per_class.items()},
                "args": vars(args),
            }, best_path)
            print(f"  ✓ Saved best model: {best_path}")

            # Persist best metrics text
            with open(metrics_dir / f"best_metrics_hybrid_spider_{tag}.txt", "w") as f:
                f.write(f"=== BEST HYBRID SPIDER MODEL ({tag}) ===\n")
                f.write(f"Saved at: {datetime.now().isoformat(timespec='seconds')}\n")
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Val loss: {val_loss:.4f}\n")
                f.write(f"Val mean acc:      {mean_acc*100:.2f}%\n")
                f.write(f"Val mean F1 macro: {mean_f1:.3f}\n\n")
                for c in SPIDER_CONDITIONS:
                    f.write(f"{DISPLAY_NAMES[c]:<20s}  acc={val_acc[c]*100:6.2f}%  "
                            f"F1_macro={f1_macro[c]:.3f}  "
                            f"per-class F1={[round(float(x), 3) for x in per_class[c]['f1']]}\n")
                    f.write(f"{'':>20s}  precision={[round(float(x), 3) for x in per_class[c]['precision']]}  "
                            f"recall={[round(float(x), 3) for x in per_class[c]['recall']]}\n")

            # Persist best metrics JSON (machine readable)
            import json as _json
            with open(metrics_dir / f"best_metrics_hybrid_spider_{tag}.json", "w") as f:
                _json.dump({
                    'tag': tag,
                    'epoch': epoch + 1,
                    'train_loss': float(avg_train_loss),
                    'val_loss': float(val_loss),
                    'val_mean_acc': float(mean_acc),
                    'val_mean_f1_macro': mean_f1,
                    'val_accuracies': {c: float(val_acc[c]) for c in SPIDER_CONDITIONS},
                    'val_f1_macro': f1_macro,
                    'per_class_f1':       {c: [float(x) for x in per_class[c]['f1']]        for c in SPIDER_CONDITIONS},
                    'per_class_precision':{c: [float(x) for x in per_class[c]['precision']] for c in SPIDER_CONDITIONS},
                    'per_class_recall':   {c: [float(x) for x in per_class[c]['recall']]    for c in SPIDER_CONDITIONS},
                    'args': vars(args),
                }, f, indent=2)
        else:
            epochs_no_improve += 1

        # Periodic snapshot
        if (epoch + 1) % args.save_freq == 0:
            snap = ckpt_dir / f"checkpoint_hybrid_spider_{tag}_epoch{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": _slim_state_dict(model.state_dict()),
                "args": vars(args),
            }, snap)
            print(f"  Saved snapshot: {snap}")

        # Save running training log
        with open(metrics_dir / f"training_log_hybrid_spider_{tag}.json", "w") as f:
            json.dump(history, f, indent=2)

        if epochs_no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no improvement for {args.early_stop_patience} epochs)")
            break

        print("-" * 72)

    print("\n" + "=" * 72)
    print("Training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir}/best_model_hybrid_spider_{tag}.pth")
    print(f"Metrics: {metrics_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
