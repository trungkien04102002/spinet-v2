#!/usr/bin/env python3
"""
Train / evaluate FORAMINAL grading on Sagittal T1 crops (#1 T1-foraminal experiment).

Why this exists
----------------
Research (see docs/ + advisor notes, F1 improvement plan item #1) showed
foraminal narrowing should be graded on Sagittal T1, not the Sagittal T2
midline crop the rest of the pipeline uses. ``experiments/f1_improvement/
prep_t1_crops.py`` extracts per-side (9, 112, 224) crops centered on the
actual T1 foraminal coordinate (see that script's docstring for why T1
coordinates differ from the T2-anchored canal pipeline).

This script reuses the existing training harness pattern from
``train_rsna_baseline.py`` (patient-level split by ``study_id``, same
``random_state=seed`` -- keep ``--seed 42`` to match the T2 baseline/CBAM/
hybrid runs already reported), but:

  - Trains on ``rsna_preprocessed_t1`` instead of ``rsna_preprocessed``.
  - Only computes loss/metrics on the two foraminal heads
    (``left_foraminal``, ``right_foraminal``). ``spinal_canal`` is always -1
    in the T1 metadata (this experiment doesn't touch canal grading) so it's
    dropped from loss/eval entirely -- simplest defensible design, per task
    spec ("spinal_canal can be ignored").
  - Reuses ``GradingModelWithCBAM`` / ``GradingModelBaseline`` unmodified
    (still outputs a spinal_canal head; it's just never trained/evaluated
    here since every spinal_canal label is -1).

Caveat on "identical seed-42 split": the *split code* (train_test_split with
random_state=42 over unique study_id) is byte-for-byte identical to
train_rsna_baseline.py. Whether the resulting *patient sets* are identical
to the T2 run depends on whether prep_t1_crops.py was run over the exact
same universe of study_ids as the T2 preprocessing (i.e. no --limit, and no
studies dropped for lacking a usable T1 series). Run the full T1 prep first
if you need the val patients to match the T2 baseline's val split.

Usage:
    # Smoke test (tiny, CPU, 1 epoch) on a small T1 subset:
    python3 experiments/f1_improvement/train_t1_foraminal.py --fast-dev \\
        --data-dir rsna_preprocessed_t1 --save-dir /tmp/t1_smoke

    # Full run (GPU, seed 42, after running the full T1 prep):
    python3 experiments/f1_improvement/train_t1_foraminal.py \\
        --model cbam --epochs 30 --batch-size 32 --lr 1e-3 --seed 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Repo root must be on sys.path when running this script directly from
# experiments/f1_improvement/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_baseline import GradingModelBaseline
from spinenet.models.grading_attention import GradingModelWithCBAM
from spinenet.metrics_logger import MetricsLogger
from spinenet.auc_metrics import aggregate_overall_auprc, compute_auc_auprc_per_condition
from spinenet.losses import compute_class_weights


CONDITIONS = ["left_foraminal", "right_foraminal"]  # spinal_canal intentionally dropped


def build_model(name: str):
    if name == "cbam":
        return GradingModelWithCBAM(format="rsna", use_cbam=True)
    elif name == "baseline":
        return GradingModelBaseline(format="rsna")
    else:
        raise ValueError(f"Unknown model: {name}")


def _stratified_head(metadata, indices, n_total):
    """Pick a small subset of `indices` that still covers all three severity
    classes (for --fast-dev).

    A plain ``indices[:n]`` slice usually lands on Normal-only rows, because
    Severe is under 5% of the data. compute_class_weights would then divide by
    a zero count -- ``1/0`` -> inf -> NaN weights after normalisation -- and the
    smoke test would "pass" without ever exercising the class-weight path it
    exists to check.
    """
    n_per_class = max(2, n_total // 3)
    sub = metadata.loc[indices]
    # Each T1 row is valid on exactly one side; the other side is -1, so max()
    # recovers that row's severity class (-1 only if both are missing).
    row_class = sub[["left_foraminal", "right_foraminal"]].max(axis=1)

    picked = []
    for cls in (0, 1, 2):
        picked.extend(sub.index[row_class == cls][:n_per_class].tolist())
    return picked or list(indices[:n_total])


def evaluate(model, dataloader, device):
    """Evaluate model on validation set -- foraminal heads only."""
    model.eval()

    all_preds = {c: [] for c in CONDITIONS}
    all_labels = {c: [] for c in CONDITIONS}
    all_outputs = {c: [] for c in CONDITIONS}

    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validating", leave=False)
        for volumes, labels in val_pbar:
            volumes = volumes.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]

            labels_device = {c: labels[c].to(device) for c in CONDITIONS}

            outputs = model(volumes)

            # Each T1 metadata row is centered on exactly ONE side (left OR
            # right) -- the other side is -1 for that row. A batch can
            # therefore contain zero valid labels for one condition (all -1),
            # and CrossEntropyLoss(ignore_index=-1) returns NaN when its
            # entire target batch is ignored. Only accumulate the loss term
            # for conditions that have >=1 valid label in this batch.
            loss = 0.0
            n_valid_conditions = 0
            for condition in CONDITIONS:
                labels_c = labels_device[condition]
                if (labels_c != -1).any():
                    loss = loss + criterion(outputs[condition], labels_c)
                    n_valid_conditions += 1

            if n_valid_conditions == 0:
                continue  # both conditions fully missing in this batch (shouldn't happen)
            loss = loss / n_valid_conditions

            total_loss += loss.item()
            num_batches += 1

            for condition in CONDITIONS:
                preds = torch.argmax(outputs[condition], dim=1)
                all_preds[condition].extend(preds.cpu().numpy())
                all_labels[condition].extend(labels[condition].numpy())
                all_outputs[condition].append(outputs[condition].cpu())

    avg_loss = total_loss / max(num_batches, 1)

    accuracies = {}
    per_class_metrics = {}

    for condition in CONDITIONS:
        labels_arr = np.array(all_labels[condition])
        preds_arr = np.array(all_preds[condition])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            labels_filtered = labels_arr[valid_mask]
            preds_filtered = preds_arr[valid_mask]

            accuracies[condition] = accuracy_score(labels_filtered, preds_filtered)

            precision, recall, f1, support = precision_recall_fscore_support(
                labels_filtered, preds_filtered, labels=[0, 1, 2], average=None, zero_division=0
            )
            per_class_metrics[condition] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        else:
            accuracies[condition] = 0.0
            per_class_metrics[condition] = None

    all_outputs_concat = {k: torch.cat(v, dim=0) for k, v in all_outputs.items() if v}
    probs_dict = {k: torch.softmax(v, dim=1).numpy() for k, v in all_outputs_concat.items()}
    labels_dict = {k: np.array(v) for k, v in all_labels.items()}
    auc_auprc_metrics = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    auc_auprc_overall = aggregate_overall_auprc(auc_auprc_metrics)

    return (avg_loss, accuracies, all_labels, all_preds, per_class_metrics,
            auc_auprc_metrics, auc_auprc_overall)


def print_per_class_metrics(per_class_metrics):
    condition_names = {"left_foraminal": "Left Foraminal", "right_foraminal": "Right Foraminal"}
    class_names = ["Normal/Mild", "Moderate", "Severe"]

    print("\n" + "=" * 70)
    print("Per-Class Metrics (Precision, Recall, F1-Score) -- T1 foraminal")
    print("=" * 70)

    for condition, name in condition_names.items():
        metrics = per_class_metrics.get(condition)
        if metrics is not None:
            print(f"\n{name}:")
            print(f"{'Class':<15} {'Support':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
            print("-" * 70)
            for i, class_name in enumerate(class_names):
                print(
                    f"{class_name:<15} {int(metrics['support'][i]):>8} "
                    f"{metrics['precision'][i]:>10.3f} {metrics['recall'][i]:>10.3f} {metrics['f1'][i]:>10.3f}"
                )
            macro_p = np.mean(metrics["precision"])
            macro_r = np.mean(metrics["recall"])
            macro_f1 = np.mean(metrics["f1"])
            print("-" * 70)
            print(f"{'Macro Avg':<15} {'':<8} {macro_p:>10.3f} {macro_r:>10.3f} {macro_f1:>10.3f}")
        else:
            print(f"\n{name}: No valid labels")


def print_confusion_matrices(all_labels, all_preds):
    condition_names = {"left_foraminal": "Left Foraminal", "right_foraminal": "Right Foraminal"}

    print("\n" + "=" * 70)
    print("Confusion Matrices -- T1 foraminal")
    print("=" * 70)

    for condition, name in condition_names.items():
        labels_arr = np.array(all_labels[condition])
        preds_arr = np.array(all_preds[condition])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            cm = confusion_matrix(labels_arr[valid_mask], preds_arr[valid_mask], labels=[0, 1, 2])
            print(f"\n{name} (excluding {(~valid_mask).sum()} missing labels):")
            print("              Pred: Normal  Moderate  Severe")
            print(f"  True: Normal     {cm[0, 0]:6d}    {cm[0, 1]:6d}  {cm[0, 2]:6d}")
            print(f"        Moderate   {cm[1, 0]:6d}    {cm[1, 1]:6d}  {cm[1, 2]:6d}")
            print(f"        Severe     {cm[2, 0]:6d}    {cm[2, 1]:6d}  {cm[2, 2]:6d}")
        else:
            print(f"\n{name}: No valid labels")


def main():
    parser = argparse.ArgumentParser(description="Train FORAMINAL grading on Sagittal T1 crops")

    # Data
    parser.add_argument("--data-dir", type=str, default="rsna_preprocessed_t1",
                        help="Path to T1-preprocessed data (output of prep_t1_crops.py)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default: 0.2)")

    # Model
    parser.add_argument("--model", type=str, default="cbam", choices=["cbam", "baseline"],
                        help="Which grading backbone to train (default: cbam)")
    parser.add_argument(
        "--cbam-checkpoint", type=str, default=None,
        help="Warm-start from a trained CBAM checkpoint "
             "(e.g. checkpoints/rsna/best_model_attention.pth) instead of the "
             "generic backbone. Continues from your existing model.")
    parser.add_argument("--use-pretrained", action="store_true", default=True,
                        help="Use pretrained 3D ResNet34 backbone (default: True)")
    parser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false",
                        help="Train from scratch")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Unfreeze backbone for fine-tuning")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--class-weight-mode", type=str, default="none",
                        choices=["none", "sqrt", "inverse", "effective"],
                        help="Class weight mode for CrossEntropyLoss (default: none). "
                             "Leave at 'none' and the Severe head collapses to the "
                             "majority class -- Severe F1 stays 0.")
    parser.add_argument("--select-by", type=str, default="val_loss",
                        choices=["val_loss", "severe_f1"],
                        help="Which metric picks the saved 'best' epoch. val_loss "
                             "rewards the majority-class collapse; use severe_f1 when "
                             "Severe F1 is what the run is for (default: val_loss)")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="checkpoints/t1_foraminal",
                        help="Directory to save checkpoints + metrics")
    parser.add_argument("--save-freq", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Early stopping patience")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split + torch + numpy")

    # Fast dev / smoke test
    parser.add_argument("--fast-dev", action="store_true",
                        help="Smoke-test mode: 1 epoch, tiny batch, num_workers=0, "
                             "and a tiny class-stratified subset of the data. Proves "
                             "the training loop + class weights + metrics JSON work "
                             "end-to-end in ~a minute. Not for real results.")
    parser.add_argument("--fast-dev-samples", type=int, default=24,
                        help="Approximate number of training samples to keep in "
                             "--fast-dev (split evenly across the 3 severity "
                             "classes, default: 24)")

    args = parser.parse_args()

    if args.fast_dev:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0
        args.early_stop_patience = 1
        args.save_freq = 1
        print("[fast-dev] epochs=1, batch_size<=4, num_workers=0 (smoke test only)")

    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    prefix = f"t1_foraminal_{args.model}{run_tag}"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("T1-Foraminal Grading Training")
    print("=" * 70)
    print(f"  Data directory:   {args.data_dir}")
    print(f"  Model:            {args.model}")
    print(f"  Heads trained:    {CONDITIONS} (spinal_canal ignored -- T1 metadata has it as -1)")
    print(f"  Validation split: {args.val_split:.1%}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Seed:             {args.seed}")

    # Load dataset. RSNAPreprocessedDataset looks for f'{split}_metadata.csv',
    # so split='t1' -> 't1_metadata.csv' (the file prep_t1_crops.py writes).
    print("\n[1/6] Loading T1 dataset...")
    dataset = RSNAPreprocessedDataset(data_dir=args.data_dir, split="t1")
    print(f"  Loaded {len(dataset)} samples")

    # Split by patient (no data leakage!) -- identical code path to
    # train_rsna_baseline.py's split, same random_state=seed.
    print("\n[2/6] Splitting dataset by patient...")
    unique_patients = dataset.metadata["study_id"].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=args.val_split, random_state=args.seed
    )

    train_indices = dataset.metadata[dataset.metadata["study_id"].isin(train_patients)].index.tolist()
    val_indices = dataset.metadata[dataset.metadata["study_id"].isin(val_patients)].index.tolist()

    if args.fast_dev:
        # Without this the "smoke test" runs a full epoch over ~15.7k crops
        # (batch 4 -> ~3.9k steps), which takes 10-20 minutes and reads every
        # .npy on disk. Stratified so Moderate/Severe survive the cut.
        train_indices = _stratified_head(dataset.metadata, train_indices, args.fast_dev_samples)
        val_indices = _stratified_head(
            dataset.metadata, val_indices, max(6, args.fast_dev_samples // 2)
        )
        print(f"  [fast-dev] Truncated to {len(train_indices)} train / "
              f"{len(val_indices)} val samples (class-stratified)")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"  Train: {len(train_dataset)} samples ({len(train_patients)} patients)")
    print(f"  Val:   {len(val_dataset)} samples ({len(val_patients)} patients)")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    print("\n[3/6] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = build_model(args.model)

    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_severe_f1 = checkpoint.get("best_severe_f1", float("-inf"))
        print(f"  Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        best_val_loss = float("inf")
        best_severe_f1 = float("-inf")

        import os
        if getattr(args, "cbam_checkpoint", None):
            if not os.path.isfile(args.cbam_checkpoint):
                raise FileNotFoundError(
                    f"\n  --cbam-checkpoint not found: {args.cbam_checkpoint}\n"
                    "  Upload best_model_attention.pth to the box first, or fix the\n"
                    "  path. Refusing to silently train from the generic backbone."
                )
            print(f"  Warm-starting from trained CBAM: {args.cbam_checkpoint}")
            model.load_trained_cbam(args.cbam_checkpoint, verbose=True)
        else:
            print("=" * 70)
            print("  ⚠️  NOT warm-starting from your trained CBAM.")
            print("  Training from the GENERIC backbone instead.")
            print("  To continue from your model (the intended setup), pass:")
            print("    --cbam-checkpoint checkpoints/rsna/best_model_attention.pth")
            print("=" * 70)
            if args.use_pretrained:
                print("  Loading pretrained backbone...")
                weights_dir = os.path.expanduser("~/.spinenet/weights")
                try:
                    model.load_pretrained_backbone(weights_dir, verbose=False)
                    print("  Pretrained backbone loaded")
                except Exception as e:
                    print(f"  Warning: could not load pretrained weights ({e}) -> training from scratch")

    if args.unfreeze_backbone:
        print("  Unfreezing backbone (fine-tuning mode)")
        model.freeze_backbone(freeze=False)
    else:
        print("  Freezing backbone (training heads only)")
        model.freeze_backbone(freeze=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    model = model.to(device)

    print("\n[4/6] Setting up training...")
    class_alpha = None
    if args.class_weight_mode != "none":
        print(f"  Computing class weights (mode={args.class_weight_mode}) from train set...")
        class_alpha = compute_class_weights(train_dataset, num_classes=3, mode=args.class_weight_mode)
        if not torch.isfinite(class_alpha).all():
            raise ValueError(
                f"Class weights are not finite: {class_alpha.tolist()}\n"
                "  A severity class has ZERO samples in the training split, so "
                "1/count blew up.\n"
                "  Raise --fast-dev-samples, or check that --data-dir really "
                "holds Moderate/Severe rows."
            )
        print(f"  Class weights: Normal={class_alpha[0]:.3f} Moderate={class_alpha[1]:.3f} Severe={class_alpha[2]:.3f}")
        if args.fast_dev:
            print("  [fast-dev] These come from a tiny class-stratified subset, so they "
                  "do NOT reflect the real dataset imbalance. Check only that they are "
                  "finite and printed; a real run puts Severe far above Normal.")
        # .to(device): the weight is a buffer on the loss module, and nothing
        # ever moves the criterion, so a CPU weight against CUDA logits is a
        # hard RuntimeError on the first batch.
        criterion = nn.CrossEntropyLoss(weight=class_alpha.to(device), ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"  Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  Loss: CrossEntropyLoss over {CONDITIONS}")

    print(f"\n[5/6] Training for {args.epochs} epochs...")
    print("=" * 70)

    best_epoch = 0
    epochs_without_improvement = 0
    metrics_logger = MetricsLogger(save_dir=save_dir, prefix=prefix)
    total_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", leave=False)
        for volumes, labels in train_pbar:
            volumes = volumes.unsqueeze(1).to(device)
            labels_device = {c: labels[c].to(device) for c in CONDITIONS}

            optimizer.zero_grad()
            outputs = model(volumes)

            # Same all-ignored-in-batch guard as evaluate() above (see that
            # comment) -- a batch can be all-left or all-right, which would
            # make CrossEntropyLoss(ignore_index=-1) return NaN for the
            # fully-missing condition.
            loss = 0.0
            n_valid_conditions = 0
            for condition in CONDITIONS:
                labels_c = labels_device[condition]
                if (labels_c != -1).any():
                    loss = loss + criterion(outputs[condition], labels_c)
                    n_valid_conditions += 1

            if n_valid_conditions == 0:
                continue  # both conditions fully missing in this batch (shouldn't happen)
            loss = loss / n_valid_conditions

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / max(num_batches, 1)

        val_start = time.time()
        (val_loss, val_accuracies, all_labels, all_preds, val_per_class_metrics,
         val_auc_auprc_metrics, val_auc_auprc_overall) = evaluate(model, val_loader, device)
        val_elapsed = time.time() - val_start
        n_val_samples = len(val_loader.dataset)
        val_throughput = float(n_val_samples / val_elapsed) if val_elapsed > 0 else 0.0
        val_ms_per_sample = float(1000 * val_elapsed / n_val_samples) if n_val_samples > 0 else 0.0

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs} Summary ({epoch_time:.1f}s)")
        print("=" * 70)
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print("\nValidation Accuracies:")
        for c in CONDITIONS:
            print(f"  {c}: {val_accuracies[c]:.2%}")

        if (epoch + 1) % 5 == 0 or args.fast_dev:
            print_per_class_metrics(val_per_class_metrics)
            print_confusion_matrices(all_labels, all_preds)

        avg_severe_f1 = float("nan")
        severe_f1s = [
            val_per_class_metrics[c]["f1"][2]
            for c in val_per_class_metrics
            if val_per_class_metrics.get(c) is not None and len(val_per_class_metrics[c]["f1"]) >= 3
        ]
        if severe_f1s:
            avg_severe_f1 = float(np.mean(severe_f1s))

        if args.select_by == "severe_f1":
            is_best = (not np.isnan(avg_severe_f1)) and avg_severe_f1 > best_severe_f1
        else:
            is_best = val_loss < best_val_loss
        # Track both regardless of which one drives selection, so the saved
        # metrics describe the whole run rather than only the chosen criterion.
        best_val_loss = min(best_val_loss, val_loss)
        if not np.isnan(avg_severe_f1):
            best_severe_f1 = max(best_severe_f1, avg_severe_f1)

        if is_best:
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            best_path = save_dir / f"best_model_{prefix}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracies": val_accuracies,
                "best_val_loss": best_val_loss,
                "best_severe_f1": best_severe_f1,
                "select_by": args.select_by,
            }, best_path)
            print(f"\n Saved best model to {best_path} "
                  f"(selected by {args.select_by})")

            elapsed_total = time.time() - total_train_start
            metrics_logger.save_best(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                val_accuracies=val_accuracies,
                per_class_metrics=val_per_class_metrics,
                avg_severe_f1=avg_severe_f1,
                auc_auprc_metrics=val_auc_auprc_metrics,
                auc_auprc_overall=val_auc_auprc_overall,
                extra={
                    "best_path": str(best_path),
                    "total_train_seconds": float(elapsed_total),
                    "avg_epoch_seconds": float(elapsed_total / (epoch + 1)),
                    "eval_seconds": float(val_elapsed),
                    "eval_samples": int(n_val_samples),
                    "eval_throughput_samples_per_sec": val_throughput,
                    "eval_ms_per_sample": val_ms_per_sample,
                    "args": vars(args),
                },
            )
        else:
            epochs_without_improvement += 1

        metrics_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            val_accuracies=val_accuracies,
            per_class_metrics=val_per_class_metrics,
            avg_severe_f1=avg_severe_f1,
            is_best=is_best,
            auc_auprc_metrics=val_auc_auprc_metrics,
            auc_auprc_overall=val_auc_auprc_overall,
            extra={
                "epoch_seconds": float(epoch_time),
                "eval_seconds": float(val_elapsed),
                "eval_throughput_samples_per_sec": val_throughput,
            },
        )

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_dir / f"checkpoint_{prefix}_epoch_{epoch + 1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracies": val_accuracies,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            print(f" Saved checkpoint to {checkpoint_path}")

        print(f"\nBest Val Loss: {best_val_loss:.4f} | "
              f"Best Severe F1: {best_severe_f1:.4f} "
              f"(saved epoch {best_epoch}, by {args.select_by})")
        print(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n Early stopping triggered after {args.early_stop_patience} epochs without improvement")
            break

        print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("[6/6] Training Complete!")
    print("=" * 70)
    print(f"  Best epoch: {best_epoch} (selected by {args.select_by})")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best Severe F1: {best_severe_f1:.4f}")
    print(f"  Metrics JSON: {save_dir / f'{prefix}_best_metrics.json'}")


if __name__ == "__main__":
    main()
