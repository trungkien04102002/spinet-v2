"""
Dump per-sample zero-shot predictions for SPIDER, plus a few input visualizations.

Run AFTER eval_zeroshot_spider.py has produced
experiments/paper_results/spider_zeroshot/results.csv. This script re-encodes the
test volumes with the same Hybrid checkpoint and writes:

  experiments/paper_results/spider_zeroshot/predictions.csv
      one row per (patient_id, ivd_label) with true and predicted class for every
      disease, plus per-disease cosine similarity scores.

  experiments/paper_results/spider_zeroshot/sample_inputs.png
      9-slice grid for 6 example IVD volumes (2 from each tier) so the advisor
      can visually inspect what the model sees.

Usage:
    python3 dump_spider_predictions.py \
        --hybrid-checkpoint checkpoints/hybrid/best_model_hybrid_fixed_e7.pth \
        --cbam-checkpoint   checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from spinenet.models.grading_hybrid import SpineNetHybrid
from prepare_spider_zeroshot import SPIDER_DISEASES, DISEASE_TIER


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid-checkpoint", type=str, required=True)
    p.add_argument("--cbam-checkpoint", type=str, required=True)
    p.add_argument("--spider-test", type=str,
                   default="rsna_preprocessed_spider/spider_zeroshot_test.csv")
    p.add_argument("--volumes-dir", type=str, default="rsna_preprocessed_spider")
    p.add_argument(
        "--output-dir", type=str,
        default="experiments/paper_results/spider_zeroshot",
    )
    p.add_argument("--prompt-template", type=str, default="med")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--device", type=str, default=None,
        help="Force device (cpu/mps/cuda). Default: auto.",
    )
    return p.parse_args()


def pick_device(forced: str | None) -> torch.device:
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


PROMPT_TEMPLATES = {
    "bare": "{label}",
    "med": "a magnetic resonance image of {label}",
    "full": "a sagittal MRI of the lumbar spine showing {label}",
}


def encode_disease_prompts(model: SpineNetHybrid, template: str):
    db = {}
    for disease, info in SPIDER_DISEASES.items():
        keys = sorted(info["prompts"].keys())
        texts = [
            PROMPT_TEMPLATES[template].format(label=info["prompts"][k]) for k in keys
        ]
        db[disease] = (keys, model.encode_text(texts))
    return db


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Hybrid checkpoint: {args.hybrid_checkpoint}")
    print(f"CBAM checkpoint:   {args.cbam_checkpoint}")

    # Load model
    model = SpineNetHybrid(
        cbam_checkpoint_path=args.cbam_checkpoint,
        biomedclip_device=str(device),
        slice_strategy="static",
    ).to(device)
    ckpt = torch.load(args.hybrid_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Encode prompts
    print("Encoding disease prompts...")
    text_db = encode_disease_prompts(model, args.prompt_template)

    # Load test set
    test_df = pd.read_csv(args.spider_test)
    volumes_root = Path(args.volumes_dir)
    print(f"Test samples: {len(test_df)}")

    # Encode all volumes
    print("Encoding volumes...")
    image_embs = []
    valid_mask = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_df), args.batch_size)):
            batch_df = test_df.iloc[i:i + args.batch_size]
            volumes = []
            for _, row in batch_df.iterrows():
                npy_path = volumes_root / row["npy_path"]
                if not npy_path.exists():
                    volumes.append(None)
                    continue
                volumes.append(torch.from_numpy(np.load(npy_path)))

            ok_mask = [v is not None for v in volumes]
            valid_mask.extend(ok_mask)
            valid_volumes = [v for v in volumes if v is not None]
            if not valid_volumes:
                continue
            batch = torch.stack(valid_volumes).unsqueeze(1).to(device)
            embs = model.encode_image(batch).cpu().numpy()
            j = 0
            for ok in ok_mask:
                if ok:
                    image_embs.append(embs[j])
                    j += 1
                else:
                    image_embs.append(np.zeros(512, dtype=np.float32))

    image_embs = np.stack(image_embs)
    print(f"Image embeddings: {image_embs.shape}")

    # Build per-sample predictions
    print("Building per-sample predictions...")
    rows = []
    img_t = torch.from_numpy(image_embs).float()
    for i, sample in test_df.iterrows():
        rec = {
            "patient_id": int(sample["patient_id"]),
            "ivd_label": int(sample["ivd_label"]),
            "valid": bool(valid_mask[i]),
        }
        for disease, info in SPIDER_DISEASES.items():
            keys, text_embs = text_db[disease]
            sims = (img_t[i] @ text_embs.cpu().T).numpy()
            pred_idx = int(np.argmax(sims))
            pred_class = keys[pred_idx]
            rec[f"{disease}_true"] = int(sample[disease])
            rec[f"{disease}_pred"] = int(pred_class)
            rec[f"{disease}_correct"] = int(rec[f"{disease}_true"] == pred_class)
            for k, sim in zip(keys, sims):
                rec[f"{disease}_sim_class{k}"] = float(sim)
            rec[f"{disease}_tier"] = DISEASE_TIER[disease]
        rows.append(rec)

    pred_df = pd.DataFrame(rows)
    pred_csv = out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved: {pred_csv}  ({len(pred_df)} rows, {len(pred_df.columns)} cols)")

    # Save sample input visualizations: 6 IVDs spanning easy/medium/hard tiers
    print("Saving sample input visualizations...")
    examples = []
    # Pick 2 samples per tier (one with positive label, one with negative for binary; for multiclass, two distinct classes)
    picks = [
        ("Disc_narrowing", 1, "easy / disc_narrowing positive"),
        ("Disc_narrowing", 0, "easy / disc_narrowing negative"),
        ("Disc_herniation", 1, "medium / disc_herniation positive"),
        ("Disc_bulging", 1, "medium / disc_bulging positive"),
        ("Pfirrman_grade", 5, "hard / pfirrman grade 5"),
        ("Modic", 2, "hard / modic type 2"),
    ]
    for col, val, title in picks:
        candidates = test_df[test_df[col] == val]
        if len(candidates) == 0:
            continue
        sample = candidates.iloc[0]
        examples.append((sample, title))

    fig, axes = plt.subplots(len(examples), 9, figsize=(18, 2 * len(examples)))
    for row, (sample, title) in enumerate(examples):
        v = np.load(volumes_root / sample["npy_path"])
        for s in range(9):
            ax = axes[row, s]
            ax.imshow(v[s], cmap="gray", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if s == 0:
                ax.set_ylabel(
                    f"pid={sample['patient_id']} ivd={sample['ivd_label']}\n{title}",
                    fontsize=8,
                )
            if row == 0:
                ax.set_title(f"slice {s}", fontsize=8)

    plt.tight_layout()
    img_path = out_dir / "sample_inputs.png"
    plt.savefig(img_path, dpi=80, bbox_inches="tight")
    print(f"Saved: {img_path}")

    # Quick summary printed to terminal
    print("\n=== Per-disease accuracy (predictions.csv) ===")
    for disease in SPIDER_DISEASES:
        col = f"{disease}_correct"
        acc = pred_df[col].mean()
        print(f"  {disease:24s}  acc={acc:.3f}  (tier={DISEASE_TIER[disease]})")


if __name__ == "__main__":
    main()
