#!/usr/bin/env python3
"""
Experiment #0 (post-hoc threshold/calibration) — Step 1: dump per-sample
softmax probabilities + true labels for an ALREADY-TRAINED RSNA grading model
on its validation split. No training happens here — inference only.

Purpose: feed `threshold_sweep.py` so we can search for a better decision
rule (per-class thresholds / logit adjustment / tau-normalization) on top of
a frozen checkpoint, instead of retraining.

Reconstructs the EXACT patient-level split used by train_rsna_hybrid.py /
train_rsna_attention.py:

    full_dataset = RSNAPreprocessedDataset(data_dir, split='train')
    unique_patients = full_dataset.metadata['study_id'].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_split, random_state=seed)
    val_indices = full_dataset.metadata[
        full_dataset.metadata['study_id'].isin(val_patients)].index.tolist()
    val_dataset = Subset(full_dataset, val_indices)   # NO augmentation, NO oversampling

Supports two model families (CLI-selectable so this is reusable for future
checkpoints, not just today's seed-42 Hybrid):

  --model hybrid   SpineNetHybrid (CBAM backbone (frozen) + BiomedCLIP (frozen)
                   + trainable fusion/cosine head). Needs BOTH a CBAM
                   checkpoint (backbone the Hybrid was built on) AND the
                   Hybrid checkpoint (trainable fusion weights) AND BiomedCLIP
                   weights available (via HF cache or network).
  --model cbam     GradingModelWithCBAM only. No BiomedCLIP dependency — useful
                   as an end-to-end pipeline sanity check when BiomedCLIP is
                   not available locally (SSL/proxy blocked, not cached).

If BiomedCLIP cannot be loaded (network blocked and not cached), this script
does NOT hang: it catches the failure, prints exactly what is missing and the
command to run this on a GPU box where BiomedCLIP is reachable, and exits
with a non-zero code.

Usage:
    # Hybrid, seed 42 (paper's canonical run)
    python3 experiments/f1_improvement/dump_logits.py --model hybrid \\
        --cbam-checkpoint checkpoints/v3_20260503/cbam_best.pth \\
        --hybrid-checkpoint checkpoints/v3_20260503/hybrid_best.pth \\
        --seed 42 \\
        --output experiments/f1_improvement/logits/hybrid_seed42_val.npz

    # CBAM-only fallback (no BiomedCLIP needed)
    python3 experiments/f1_improvement/dump_logits.py --model cbam \\
        --cbam-checkpoint checkpoints/v3_20260503/cbam_best.pth \\
        --seed 42 \\
        --output experiments/f1_improvement/logits/cbam_seed42_val.npz

On a 4090 box (or any box with internet / a network-reachable HF cache):
    python3 experiments/f1_improvement/dump_logits.py --model hybrid \\
        --cbam-checkpoint checkpoints/v3_20260503/cbam_best.pth \\
        --hybrid-checkpoint checkpoints/v3_20260503/hybrid_best.pth \\
        --seed 42 --device cuda \\
        --output experiments/f1_improvement/logits/hybrid_seed42_val.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# Make repo root importable regardless of cwd (this file lives 2 levels deep).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset  # noqa: E402

CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]

# Same prompts as train_rsna_hybrid.py — must match so cosine-sim classification
# reproduces the trained model's decision (only relevant for --model hybrid).
RSNA_PROMPTS = {
    "spinal_canal": {
        0: "normal or mild spinal canal stenosis",
        1: "moderate spinal canal stenosis",
        2: "severe spinal canal stenosis",
    },
    "left_foraminal": {
        0: "normal or mild left neural foraminal narrowing",
        1: "moderate left neural foraminal narrowing",
        2: "severe left neural foraminal narrowing",
    },
    "right_foraminal": {
        0: "normal or mild right neural foraminal narrowing",
        1: "moderate right neural foraminal narrowing",
        2: "severe right neural foraminal narrowing",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["hybrid", "cbam"], required=True)
    p.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "rsna_preprocessed"))
    p.add_argument("--cbam-checkpoint", type=str, required=True,
                   help="CBAM checkpoint. Required for both --model cbam (loaded "
                        "directly) and --model hybrid (frozen backbone the Hybrid "
                        "head was trained on top of).")
    p.add_argument("--hybrid-checkpoint", type=str, default=None,
                   help="Hybrid trainable-weights checkpoint. Required if --model hybrid.")
    p.add_argument("--slice-strategy", type=str, default="static", choices=["static", "dynamic"])
    p.add_argument("--fusion", type=str, default="concat_mlp", choices=["concat_mlp", "gated"])
    p.add_argument("--seed", type=int, default=42, help="Must match the seed used at training time (patient split).")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--output", type=str, required=True)
    return p.parse_args()


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_val_split(data_dir: str, seed: int, val_split: float):
    """Reproduce the exact patient-level split from train_rsna_hybrid.py."""
    full_dataset = RSNAPreprocessedDataset(data_dir=data_dir, split="train", transform=None)
    unique_patients = full_dataset.metadata["study_id"].unique()
    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_split, random_state=seed,
    )
    val_indices = full_dataset.metadata[
        full_dataset.metadata["study_id"].isin(val_patients)
    ].index.tolist()
    val_dataset = Subset(full_dataset, val_indices)
    val_study_ids = full_dataset.metadata.loc[val_indices, "study_id"].to_numpy()
    return val_dataset, val_study_ids, len(unique_patients), len(train_patients), len(val_patients)


def try_build_hybrid_model(cbam_checkpoint, device, slice_strategy, fusion):
    """Build SpineNetHybrid, catching BiomedCLIP load failures cleanly."""
    try:
        from spinenet.models.grading_hybrid import SpineNetHybrid
    except Exception as e:  # pragma: no cover - import-time failure
        print("BLOCKED: could not import SpineNetHybrid / its deps "
              f"(likely missing 'open_clip_torch'): {e}")
        print_gpu_box_hint(cbam_checkpoint)
        sys.exit(2)

    try:
        model = SpineNetHybrid(
            cbam_checkpoint_path=cbam_checkpoint,
            biomedclip_device=str(device),
            slice_strategy=slice_strategy,
            fusion_mode=fusion,
        ).to(device)
    except Exception as e:
        print("BLOCKED: BiomedCLIP weights are not available locally "
              "(download blocked by SSL/proxy, or not cached in "
              "~/.cache/huggingface/hub).")
        print(f"Underlying error: {type(e).__name__}: {e}")
        print_gpu_box_hint(cbam_checkpoint)
        sys.exit(2)
    return model


def print_gpu_box_hint(cbam_checkpoint):
    print("\nRun this on the 4090 box instead (has internet access to HuggingFace):")
    print("  git pull  # or scp this experiments/f1_improvement/ dir over")
    print("  source spinenet-venv/bin/activate")
    print("  export PYTHONPATH=$PYTHONPATH:$(pwd)")
    print("  python3 experiments/f1_improvement/dump_logits.py --model hybrid \\")
    print(f"      --cbam-checkpoint {cbam_checkpoint} \\")
    print("      --hybrid-checkpoint checkpoints/v3_20260503/hybrid_best.pth \\")
    print("      --seed 42 --device cuda \\")
    print("      --output experiments/f1_improvement/logits/hybrid_seed42_val.npz")
    print("\nAs a local fallback (no BiomedCLIP needed), the CBAM-only checkpoint "
          "can still be dumped to validate the rest of the pipeline:")
    print("  python3 experiments/f1_improvement/dump_logits.py --model cbam \\")
    print(f"      --cbam-checkpoint {cbam_checkpoint} \\")
    print("      --seed 42 \\")
    print("      --output experiments/f1_improvement/logits/cbam_seed42_val.npz")


def build_text_database(model, device):
    db = {}
    for condition, prompts in RSNA_PROMPTS.items():
        texts = [prompts[i] for i in sorted(prompts.keys())]
        embs = model.encode_text(texts).to(device)
        db[condition] = embs
    return db


def build_cbam_model(cbam_checkpoint, device):
    from spinenet.models.grading_attention import GradingModelWithCBAM
    model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    ckpt = torch.load(cbam_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  (load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys — "
              "expected if checkpoint was saved with a different wrapper)")
    model = model.to(device)
    model.eval()
    return model


def load_hybrid_trainable_weights(model, hybrid_checkpoint, device):
    ckpt = torch.load(hybrid_checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Missing should only be frozen cbam.*/biomedclip.* params (trainable-only
    # checkpoint by design — see train_rsna_hybrid.py save logic).
    non_frozen_missing = [k for k in missing if not (k.startswith("cbam.") or k.startswith("biomedclip."))]
    if non_frozen_missing:
        print(f"  WARNING: missing non-frozen keys when loading Hybrid checkpoint: {non_frozen_missing}")
    if unexpected:
        print(f"  WARNING: unexpected keys when loading Hybrid checkpoint: {unexpected}")
    model.eval()
    return model


@torch.no_grad()
def run_inference_cbam(model, loader, device):
    all_probs = {c: [] for c in CONDITIONS}
    all_logits = {c: [] for c in CONDITIONS}
    all_labels = {c: [] for c in CONDITIONS}
    for volumes, labels in loader:
        volumes = volumes.unsqueeze(1).to(device)
        outputs = model(volumes)
        for c in CONDITIONS:
            logits_np = outputs[c].cpu().numpy()
            probs = torch.softmax(outputs[c], dim=1).cpu().numpy()
            all_probs[c].append(probs)
            all_logits[c].append(logits_np)
            all_labels[c].append(labels[c].numpy())
    return all_probs, all_logits, all_labels


@torch.no_grad()
def run_inference_hybrid(model, text_db, loader, device):
    all_probs = {c: [] for c in CONDITIONS}
    all_logits = {c: [] for c in CONDITIONS}
    all_labels = {c: [] for c in CONDITIONS}
    for volumes, labels in loader:
        volumes = volumes.unsqueeze(1).to(device)
        image_emb = model.encode_image(volumes)
        for c in CONDITIONS:
            text_embs = text_db[c]
            logits = model.logit_scale.exp().clamp(max=100.0) * (image_emb @ text_embs.T)
            logits_np = logits.cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs[c].append(probs)
            all_logits[c].append(logits_np)
            all_labels[c].append(labels[c].numpy())
    return all_probs, all_logits, all_labels


def main():
    args = parse_args()
    device = pick_device(args.device)
    print("=" * 70)
    print("Experiment #0 — dump_logits.py (inference only, no training)")
    print("=" * 70)
    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(f"Seed:       {args.seed}")
    print(f"Val split:  {args.val_split}")

    if args.model == "hybrid" and args.hybrid_checkpoint is None:
        print("ERROR: --hybrid-checkpoint is required when --model hybrid")
        sys.exit(1)

    print("\n[1/4] Reconstructing seed-{} patient split...".format(args.seed))
    val_dataset, val_study_ids, n_patients, n_train_p, n_val_p = build_val_split(
        args.data_dir, args.seed, args.val_split
    )
    print(f"  Total patients: {n_patients}  train: {n_train_p}  val: {n_val_p}")
    print(f"  Val samples (IVDs): {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    print("\n[2/4] Building model...")
    t0 = time.time()
    if args.model == "cbam":
        model = build_cbam_model(args.cbam_checkpoint, device)
        text_db = None
    else:
        model = try_build_hybrid_model(args.cbam_checkpoint, device, args.slice_strategy, args.fusion)
        print(f"  Loading Hybrid trainable weights from {args.hybrid_checkpoint}")
        model = load_hybrid_trainable_weights(model, args.hybrid_checkpoint, device)
        print("  Pre-computing text embeddings (frozen BiomedCLIP text encoder)...")
        text_db = build_text_database(model, device)
    print(f"  ✓ Model ready ({time.time() - t0:.1f}s)")

    print("\n[3/4] Running inference on val split...")
    t0 = time.time()
    if args.model == "cbam":
        all_probs, all_logits, all_labels = run_inference_cbam(model, val_loader, device)
    else:
        all_probs, all_logits, all_labels = run_inference_hybrid(model, text_db, val_loader, device)
    elapsed = time.time() - t0
    print(f"  ✓ Inference done in {elapsed:.1f}s ({len(val_dataset)/elapsed:.1f} samples/s)")

    print("\n[4/4] Saving probabilities + logits + labels...")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for c in CONDITIONS:
        save_dict[f"probs_{c}"] = np.concatenate(all_probs[c], axis=0).astype(np.float32)
        save_dict[f"logits_{c}"] = np.concatenate(all_logits[c], axis=0).astype(np.float32)
        save_dict[f"labels_{c}"] = np.concatenate(all_labels[c], axis=0).astype(np.int64)
    save_dict["study_id"] = np.asarray(val_study_ids)
    meta = {
        "model": args.model,
        "seed": args.seed,
        "val_split": args.val_split,
        "cbam_checkpoint": str(args.cbam_checkpoint),
        "hybrid_checkpoint": str(args.hybrid_checkpoint) if args.hybrid_checkpoint else None,
        "slice_strategy": args.slice_strategy,
        "fusion": args.fusion,
        "n_val_samples": len(val_dataset),
        "n_val_patients": n_val_p,
        "device": str(device),
    }
    save_dict["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(out_path, **save_dict)
    print(f"  ✓ Saved to {out_path}")
    print(f"  Shapes: " + ", ".join(
        f"{c}={save_dict[f'probs_{c}'].shape}" for c in CONDITIONS
    ))
    print("\nDone.")


if __name__ == "__main__":
    main()
