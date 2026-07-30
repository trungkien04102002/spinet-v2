#!/usr/bin/env python3
"""
Train GradingMultiView (T2 + T1 fusion) on RSNA lumbar grading.

Mirrors train_rsna_baseline.py's split/loss/eval/metrics-logging pattern
EXACTLY (same seed-42 patient split via sklearn train_test_split) so results
are directly comparable to the single-view harness. See
docs/LVTN_phase3/MULTIVIEW_RESEARCH.md for the architecture rationale.

Usage (once T1 crops exist for real):
    # Late fusion (Week-2 baseline)
    python3 experiments/multiview/train_multiview.py \\
        --fusion concat --seed 42 \\
        --data-dir rsna_preprocessed --t1-dir rsna_preprocessed_t1 \\
        --epochs 30 --batch-size 16 --lr 1e-3

    # Gated leader/supporter fusion (the advisor's core ask)
    python3 experiments/multiview/train_multiview.py \\
        --fusion gated --seed 42 \\
        --data-dir rsna_preprocessed --t1-dir rsna_preprocessed_t1 \\
        --epochs 30 --batch-size 16 --lr 1e-3

Smoke test (CPU, dummy T1 zeros, tiny subset, 1 epoch):
    python3 experiments/multiview/train_multiview.py \\
        --fast-dev --allow-missing-t1 --fusion concat
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_recall_fscore_support,
)
from tqdm import tqdm

# Make the repo root importable regardless of cwd (this file lives under
# experiments/multiview/, two levels below the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Also make this file's own directory importable for the sibling modules
# (multiview_dataloader, grading_multiview) when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from multiview_dataloader import RSNAMultiViewDataset, make_patient_split  # noqa: E402
from grading_multiview import GradingMultiView, CONDITIONS  # noqa: E402
from spinenet.metrics_logger import MetricsLogger  # noqa: E402
from spinenet.losses import compute_class_weights  # noqa: E402
from spinenet.auc_metrics import (  # noqa: E402
    aggregate_overall_auprc,
    compute_auc_auprc_per_condition,
)


def compute_weighted_log_loss(outputs_dict, labels_dict):
    """Weighted log loss (RSNA competition metric), equal weights per
    condition — identical to train_rsna_baseline.py's version."""
    weights = {cond: 1.0 for cond in CONDITIONS}
    total_loss = 0.0
    total_weight = 0.0

    for condition in CONDITIONS:
        probs = torch.softmax(outputs_dict[condition], dim=1).cpu().numpy()
        labels = labels_dict[condition].cpu().numpy()
        valid_mask = labels != -1
        if valid_mask.sum() > 0:
            condition_loss = log_loss(
                labels[valid_mask], probs[valid_mask], labels=[0, 1, 2]
            )
            total_loss += condition_loss * weights[condition]
            total_weight += weights[condition]

    return total_loss / total_weight if total_weight > 0 else float("nan")


def evaluate(model, dataloader, device):
    """Evaluate model on validation set. Mirrors train_rsna_baseline.py's
    evaluate(), adapted for the three-input (T2, T1-left, T1-right) forward
    signature."""
    model.eval()

    all_preds = {cond: [] for cond in CONDITIONS}
    all_labels = {cond: [] for cond in CONDITIONS}
    all_outputs = {cond: [] for cond in CONDITIONS}

    total_loss = 0.0
    num_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    with torch.no_grad():
        val_pbar = tqdm(dataloader, desc="Validating", leave=False)
        for t2_vol, t1_left_vol, t1_right_vol, labels in val_pbar:
            t2_vol = t2_vol.unsqueeze(1).to(device)  # [B, 1, 9, 112, 224]
            t1_left_vol = t1_left_vol.unsqueeze(1).to(device)
            t1_right_vol = t1_right_vol.unsqueeze(1).to(device)

            labels_device = {
                cond: labels[cond].to(device) for cond in CONDITIONS
            }

            outputs = model(t2_vol, t1_left_vol, t1_right_vol)

            loss = 0.0
            for cond in CONDITIONS:
                loss += criterion(outputs[cond], labels_device[cond])
            loss /= len(CONDITIONS)

            total_loss += loss.item()
            num_batches += 1

            for cond in CONDITIONS:
                preds = torch.argmax(outputs[cond], dim=1)
                all_preds[cond].extend(preds.cpu().numpy())
                all_labels[cond].extend(labels[cond].numpy())
                all_outputs[cond].append(outputs[cond].cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else float("nan")

    accuracies = {}
    per_class_metrics = {}

    for cond in CONDITIONS:
        labels_arr = np.array(all_labels[cond])
        preds_arr = np.array(all_preds[cond])
        valid_mask = labels_arr != -1

        if valid_mask.sum() > 0:
            labels_filtered = labels_arr[valid_mask]
            preds_filtered = preds_arr[valid_mask]

            accuracies[cond] = accuracy_score(labels_filtered, preds_filtered)

            precision, recall, f1, support = precision_recall_fscore_support(
                labels_filtered, preds_filtered,
                labels=[0, 1, 2], average=None, zero_division=0,
            )
            per_class_metrics[cond] = {
                "precision": precision, "recall": recall,
                "f1": f1, "support": support,
            }
        else:
            accuracies[cond] = 0.0
            per_class_metrics[cond] = None

    all_outputs_concat = {k: torch.cat(v, dim=0) for k, v in all_outputs.items()}
    all_labels_tensor = {k: torch.tensor(v) for k, v in all_labels.items()}
    weighted_logloss = compute_weighted_log_loss(all_outputs_concat, all_labels_tensor)

    probs_dict = {
        k: torch.softmax(v, dim=1).numpy() for k, v in all_outputs_concat.items()
    }
    labels_dict = {k: np.array(v) for k, v in all_labels.items()}
    auc_auprc_metrics = compute_auc_auprc_per_condition(probs_dict, labels_dict)
    auc_auprc_overall = aggregate_overall_auprc(auc_auprc_metrics)

    return (avg_loss, accuracies, weighted_logloss, all_labels, all_preds,
            per_class_metrics, auc_auprc_metrics, auc_auprc_overall)


def main():
    parser = argparse.ArgumentParser(description="Train multi-view (T2+T1) RSNA grading model")

    # Data
    parser.add_argument("--data-dir", type=str, default="rsna_preprocessed",
                        help="Path to T2 preprocessed data (existing pipeline)")
    parser.add_argument("--t1-dir", type=str, default="rsna_preprocessed_t1",
                        help="Path to T1 preprocessed data (sibling task; may not exist yet)")
    parser.add_argument("--allow-missing-t1", action="store_true",
                        help="Fall back to dummy-zeros T1 volumes if T1 dir/metadata is missing")
    parser.add_argument("--val-split", type=float, default=0.2)

    # Model
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "gated"],
                        help="Fusion mode: 'concat' (late fusion) or 'gated' (GMU leader/supporter)")
    parser.add_argument("--no-cbam", dest="use_cbam", action="store_false", default=True,
                        help="Disable CBAM in the per-sequence encoders")
    parser.add_argument("--use-pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="use_pretrained", action="store_false")
    parser.add_argument(
        "--cbam-checkpoint", type=str, default=None,
        help="Warm-start BOTH encoders from a trained CBAM checkpoint "
             "(e.g. checkpoints/rsna/best_model_attention.pth). Continues from "
             "your existing model instead of the generic backbone. Overrides "
             "--use-pretrained.")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--save-dir", type=str, default="experiments/multiview/checkpoints")
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=10)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split + torch + numpy (IDENTICAL split logic to "
                             "train_rsna_baseline.py's seed-42 harness)")

    # Dev / smoke-test mode
    parser.add_argument("--fast-dev", action="store_true",
                        help="Smoke-test mode: tiny subset, 1 epoch, small batch, CPU-friendly. "
                             "Overrides --epochs/--batch-size/--num-workers.")
    parser.add_argument("--fast-dev-samples", type=int, default=24,
                        help="Total samples to keep (before train/val split) in --fast-dev mode")

    args = parser.parse_args()

    if args.fast_dev:
        args.epochs = 1
        args.batch_size = min(args.batch_size, 4)
        args.num_workers = 0
        args.use_pretrained = False  # avoid requiring ~/.spinenet/weights on a dev box

    # Reproducibility — before any DataLoader / model construction.
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_tag = "" if args.seed == 42 else f"_seed{args.seed}"
    prefix = f"multiview_{args.fusion}{run_tag}"

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RSNA Multi-View (T2+T1) Grading Model Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Fusion mode: {args.fusion}")
    print(f"  T2 data dir: {args.data_dir}")
    print(f"  T1 data dir: {args.t1_dir} (allow_missing={args.allow_missing_t1})")
    print(f"  Fast-dev: {args.fast_dev}")
    print(f"  Epochs: {args.epochs}  Batch size: {args.batch_size}  LR: {args.lr}")
    print(f"  Seed: {args.seed}")

    # --- Dataset ---
    print("\n[1/6] Loading dataset...")
    dataset = RSNAMultiViewDataset(
        data_dir=args.data_dir,
        t1_dir=args.t1_dir,
        split="train",
        allow_missing_t1=args.allow_missing_t1,
    )
    print(f"Loaded {len(dataset)} samples")

    # --- Patient-level split (IDENTICAL logic to train_rsna_baseline.py) ---
    print("\n[2/6] Splitting dataset by patient (seed={})...".format(args.seed))
    train_indices, val_indices = make_patient_split(
        dataset.metadata, val_split=args.val_split, seed=args.seed
    )

    if args.fast_dev:
        # Keep a tiny, deterministic slice for the smoke test only.
        n = args.fast_dev_samples
        train_indices = train_indices[: max(1, int(n * (1 - args.val_split)))]
        val_indices = val_indices[: max(1, int(n * args.val_split))]
        print(f"  [fast-dev] Truncated to {len(train_indices)} train / "
              f"{len(val_indices)} val samples")

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    print(f"Train: {len(train_dataset)} samples   Val: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # --- Model ---
    print("\n[3/6] Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = GradingMultiView(fusion=args.fusion, use_cbam=args.use_cbam)

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

        if args.cbam_checkpoint:
            if not os.path.isfile(args.cbam_checkpoint):
                raise FileNotFoundError(
                    f"\n  --cbam-checkpoint not found: {args.cbam_checkpoint}\n"
                    "  Upload best_model_attention.pth to the box first, or fix the\n"
                    "  path. Refusing to silently train from the generic backbone."
                )
            print(f"  Warm-starting both encoders from trained CBAM: {args.cbam_checkpoint}")
            model.load_trained_cbam_encoders(args.cbam_checkpoint, verbose=True)
        else:
            print("=" * 70)
            print("  ⚠️  NOT warm-starting from your trained CBAM.")
            print("  Training both encoders from the GENERIC backbone instead.")
            print("  To continue from your model (the intended setup), pass:")
            print("    --cbam-checkpoint checkpoints/rsna/best_model_attention.pth")
            print("=" * 70)
            if args.use_pretrained:
                print("  Loading pretrained backbone into both T2/T1 encoders...")
                weights_dir = os.path.expanduser("~/.spinenet/weights")
                try:
                    model.load_pretrained_backbones(weights_dir, verbose=False)
                    print("  Pretrained backbones loaded")
                except Exception as e:
                    print(f"  Warning: Could not load pretrained weights: {e}")
                    print("  -> Training from scratch")

    if args.unfreeze_backbone:
        print("  Unfreezing backbones (fine-tuning mode)")
        model.freeze_backbones(freeze=False)
    else:
        print("  Freezing backbones (training heads/gates only)")
        model.freeze_backbones(freeze=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")

    model = model.to(device)

    # --- Training setup ---
    print("\n[4/6] Setting up training...")
    if args.class_weight_mode != "none":
        print(f"  Computing class weights (mode={args.class_weight_mode}) from train set...")
        class_alpha = compute_class_weights(
            train_dataset, num_classes=3, mode=args.class_weight_mode
        )
        print(f"  Class weights: Normal={class_alpha[0]:.3f} "
              f"Moderate={class_alpha[1]:.3f} Severe={class_alpha[2]:.3f}")
        # .to(device): the weight is a buffer on the loss module, and nothing
        # ever moves the criterion, so a CPU weight against CUDA logits is a
        # hard RuntimeError on the first batch.
        criterion = nn.CrossEntropyLoss(weight=class_alpha.to(device), ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    print(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Loss: CrossEntropyLoss (class_weight_mode={args.class_weight_mode})")
    print("Scheduler: ReduceLROnPlateau (patience=5)")

    # --- Training loop ---
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
        for t2_vol, t1_left_vol, t1_right_vol, labels in train_pbar:
            t2_vol = t2_vol.unsqueeze(1).to(device)
            t1_left_vol = t1_left_vol.unsqueeze(1).to(device)
            t1_right_vol = t1_right_vol.unsqueeze(1).to(device)

            labels_device = {cond: labels[cond].to(device) for cond in CONDITIONS}

            optimizer.zero_grad()
            outputs = model(t2_vol, t1_left_vol, t1_right_vol)

            loss = 0.0
            for cond in CONDITIONS:
                loss += criterion(outputs[cond], labels_device[cond])
            loss /= len(CONDITIONS)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / num_batches if num_batches > 0 else float("nan")

        val_start = time.time()
        (val_loss, val_accuracies, val_weighted_logloss, all_labels, all_preds,
         val_per_class_metrics, val_auc_auprc_metrics, val_auc_auprc_overall) = evaluate(
            model, val_loader, device
        )
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
        print(f"Val Weighted Log Loss: {val_weighted_logloss:.4f}")
        print("\nValidation Accuracies:")
        for cond in CONDITIONS:
            print(f"  {cond}: {val_accuracies[cond]:.2%}")

        avg_severe_f1 = float("nan")
        if val_per_class_metrics:
            severe_f1s = [
                val_per_class_metrics[c]["f1"][2]
                for c in val_per_class_metrics
                if val_per_class_metrics[c] is not None and len(val_per_class_metrics[c]["f1"]) >= 3
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
                "val_weighted_logloss": val_weighted_logloss,
                "val_accuracies": val_accuracies,
                "best_val_loss": best_val_loss,
                "best_severe_f1": best_severe_f1,
                "select_by": args.select_by,
                "fusion": args.fusion,
            }, best_path)
            print(f"\nSaved best model to {best_path} "
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
                    "val_weighted_logloss": float(val_weighted_logloss),
                    "best_path": str(best_path),
                    "total_train_seconds": float(elapsed_total),
                    "avg_epoch_seconds": float(elapsed_total / (epoch + 1)),
                    "eval_seconds": float(val_elapsed),
                    "eval_samples": int(n_val_samples),
                    "eval_throughput_samples_per_sec": val_throughput,
                    "eval_ms_per_sample": val_ms_per_sample,
                    "fusion": args.fusion,
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
                "val_weighted_logloss": float(val_weighted_logloss),
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
                "fusion": args.fusion,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        print(f"\nBest Val Loss: {best_val_loss:.4f} | "
              f"Best Severe F1: {best_severe_f1:.4f} "
              f"(saved epoch {best_epoch}, by {args.select_by})")
        print(f"Epochs without improvement: {epochs_without_improvement}")

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\nEarly stopping triggered after {args.early_stop_patience} epochs without improvement")
            break

        print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print("[6/6] Training Complete!")
    print("=" * 70)
    print(f"\nBest model: epoch {best_epoch} (by {args.select_by}), "
          f"val_loss={best_val_loss:.4f}, severe_f1={best_severe_f1:.4f}")
    print(f"Saved at: {save_dir / f'best_model_{prefix}.pth'}")
    print(f"Metrics JSON: {metrics_logger.best_json}")


if __name__ == "__main__":
    main()
