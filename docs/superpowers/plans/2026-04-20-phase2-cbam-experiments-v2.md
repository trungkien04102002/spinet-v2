# Phase 2 v2 — CBAM Experiments (Staged, Minimal-First)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce metrics for Table 2 (baseline vs CBAM+Focal) using a staged approach — each stage produces a working artifact, and later stages are optional.

**Architecture:** Each stage is independently executable and produces a committable artifact. Stop after Stage A if time is short (paper still works). Stage B and C add polish and ablation strength.

**Replaces:** `2026-04-20-phase2-cbam-experiments.md` (original, too large — kept for reference).

---

## Stage overview

| Stage | What it produces | Time | Required for paper? |
|---|---|---|---|
| **A** | `02_cbam_focal.txt` — IVD-level metrics for existing CBAM+Focal checkpoint, matching `01_baseline.txt` format | ~1 hour | ✅ YES (minimum Table 2) |
| **B** | `03_cbam_only.txt`, `04_focal_only.txt` — 2 ablation variants trained | ~2 hours (GPU) | ⚠️ Recommended (strengthens Table 2) |
| **C** | CBAM+Focal **patient-level** predictions in Phase-1 format → add to Table 1 | ~1 hour | ⚠️ Optional (adds CBAM column to Table 1) |
| **D** | 3-seed runs for variance estimates | ~6 hours (GPU) | ❌ Nice to have (only for Rank-B targets) |

**Recommended path for Rank-C timeline:** Stage A only, then move to Phase 3.  
**Recommended path for thesis defense:** Stage A + Stage B.  
**Full paper story:** Stage A + Stage B + Stage C.

---

## Stage A — Extract CBAM+Focal IVD metrics (MINIMUM VIABLE)

This is the smallest unit that unblocks Table 2. No training required.

**Files:**
- Create: `evaluate_checkpoint_ivd.py` (script)
- Output: `experiments/paper_results/table2_ablation/ivd_metrics/02_cbam_focal.txt`

### Step-by-step

- [ ] **A.1: Verify checkpoint exists**

```bash
cd /Users/kienha/spinet-v2
ls -la checkpoints/best_model_attention.pth
```

Expected: file exists, size ~100MB. If missing, Stage A cannot run — go retrain first.

- [ ] **A.2: Inspect `base_line.txt` to confirm output format**

```bash
cat experiments/paper_results/table2_ablation/ivd_metrics/01_baseline.txt
```

Expected format (per condition):
```
<Condition Name>:
  Overall Accuracy: 0.XXXX (XX.XX%)
  Total Samples: XXXX

  Per-Class Metrics:
  -----
  Class           Precision    Recall       F1-Score     Support
  Normal/Mild     ...          ...          ...          ...
  Moderate        ...          ...          ...          ...
  Severe          ...          ...          ...          ...
  -----
  Macro Avg       ...          ...          ...          ...
```

This is the exact format Stage A must reproduce for the CBAM+Focal checkpoint.

- [ ] **A.3: Find whatever script generated `base_line.txt`**

```bash
grep -r "Overall Accuracy" /Users/kienha/spinet-v2 --include="*.py" -l 2>/dev/null
grep -r "Per-Class Metrics" /Users/kienha/spinet-v2 --include="*.py" -l 2>/dev/null
```

If found: reuse that script, only changing the checkpoint path + model class. If not found, the script was ad-hoc — implement it fresh (Step A.4).

- [ ] **A.4: Write `evaluate_checkpoint_ivd.py` (only if no existing script)**

Create `/Users/kienha/spinet-v2/evaluate_checkpoint_ivd.py`:

```python
"""Evaluate a trained checkpoint on held-out validation split and print per-IVD
per-class precision/recall/F1 in the format used by experiments/rsna/base_line.txt.

Usage:
    python3 evaluate_checkpoint_ivd.py \
        --checkpoint checkpoints/best_model_attention.pth \
        --model-type cbam \
        --output experiments/paper_results/table2_ablation/ivd_metrics/02_cbam_focal.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_attention import GradingModelWithCBAM
from spinenet.models.grading_baseline import GradingModelBaseline


CONDITIONS = [
    ("spinal_canal", "Spinal Canal Stenosis"),
    ("left_foraminal", "Left Foraminal Narrowing"),
    ("right_foraminal", "Right Foraminal Narrowing"),
]
CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


def build_model(model_type: str, checkpoint_path: Path, device: torch.device):
    if model_type == "cbam":
        model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    elif model_type == "baseline":
        model = GradingModelBaseline(format="rsna")
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def collect_predictions(model, loader, device):
    all_preds = {c: [] for c, _ in CONDITIONS}
    all_labels = {c: [] for c, _ in CONDITIONS}
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Evaluating"):
            volumes = volumes.unsqueeze(1).to(device)
            outputs = model(volumes)
            for cond, _ in CONDITIONS:
                preds = torch.argmax(outputs[cond], dim=1).cpu().numpy()
                all_preds[cond].extend(preds.tolist())
                all_labels[cond].extend(labels[cond].numpy().tolist())
    return all_preds, all_labels


def format_report(preds, labels) -> str:
    out = []
    out.append("EVALUATION METRICS")
    out.append("=" * 70)
    for cond, display_name in CONDITIONS:
        y_true = np.array(labels[cond])
        y_pred = np.array(preds[cond])
        valid = y_true != -1
        y_true = y_true[valid]
        y_pred = y_pred[valid]
        if len(y_true) == 0:
            continue
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )
        out.append(f"\n{display_name}:")
        out.append(f"  Overall Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        out.append(f"  Total Samples: {len(y_true)}")
        out.append("")
        out.append("  Per-Class Metrics:")
        out.append("  " + "-" * 60)
        out.append(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        out.append("  " + "-" * 60)
        for i, name in enumerate(CLASS_NAMES):
            out.append(f"  {name:<15} {prec[i]:<12.3f} {rec[i]:<12.3f} {f1[i]:<12.3f} {int(support[i])}")
        out.append("  " + "-" * 60)
        out.append(f"  {'Macro Avg':<15} {prec.mean():<12.3f} {rec.mean():<12.3f} {f1.mean():<12.3f} {int(support.sum())}")
        out.append("  " + "-" * 60)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["baseline", "cbam"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("rsna_preprocessed"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = RSNAPreprocessedDataset(data_dir=str(args.data_dir), split="train", transform=None)
    patients = full.metadata["study_id"].unique()
    _, val_patients = train_test_split(patients, test_size=args.val_split, random_state=args.seed)
    val_indices = full.metadata[full.metadata["study_id"].isin(val_patients)].index.tolist()
    val_ds = Subset(full, val_indices)
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args.model_type, args.checkpoint, device)
    preds, labels = collect_predictions(model, loader, device)
    report = format_report(preds, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(report)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **A.5: Smoke-test on CPU with a tiny subset first**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
# Quick sanity: run on baseline checkpoint, verify output matches 01_baseline.txt approximately.
python3 evaluate_checkpoint_ivd.py \
    --checkpoint checkpoints/best_model.pth \
    --model-type baseline \
    --output /tmp/sanity_baseline.txt
diff /tmp/sanity_baseline.txt experiments/paper_results/table2_ablation/ivd_metrics/01_baseline.txt || echo "(differences may be due to non-determinism; numbers should be close)"
```

If numbers don't match `01_baseline.txt` closely → the original script used different split/seed/protocol. Investigate before trusting Stage A output.

- [ ] **A.6: Run for CBAM+Focal checkpoint**

```bash
cd /Users/kienha/spinet-v2
python3 evaluate_checkpoint_ivd.py \
    --checkpoint checkpoints/best_model_attention.pth \
    --model-type cbam \
    --output experiments/paper_results/table2_ablation/ivd_metrics/02_cbam_focal.txt
```

Expected: text file with same structure as `01_baseline.txt`, numbers should show Severe Recall much higher (33% → 83.5% for Spinal Canal per earlier reports).

- [ ] **A.7: Commit**

```bash
cd /Users/kienha/spinet-v2
git add evaluate_checkpoint_ivd.py experiments/paper_results/table2_ablation/ivd_metrics/02_cbam_focal.txt
git commit -m "Add IVD-level evaluation script; generate CBAM+Focal metrics for Table 2"
```

### Stage A Done when

- [ ] `02_cbam_focal.txt` exists with same structure as `01_baseline.txt`.
- [ ] Severe Recall in `02_cbam_focal.txt` is clearly higher than in `01_baseline.txt` (>0% for foraminal, >31% for spinal canal).
- [ ] Script committed.

---

## Stage B — Ablation: +CBAM only, +Focal only (RECOMMENDED, NOT REQUIRED)

Adds 2 rows to Table 2 showing that each component contributes — without both, Severe recall does not recover. Needed if a reviewer asks "is Focal alone enough?".

### Step-by-step

- [ ] **B.1: Verify `train_rsna_attention.py` supports the relevant flags**

The sweep needs:
- `--no-cbam` (exists, line 67-68)
- `--use-focal=True/False` — check, if only `--use-focal` exists without True/False toggle, add one.
- `--no-uncertainty` — may not exist; add if needed.
- `--augmentation none --oversample-factor 1` (to isolate CBAM/Focal from augmentation/oversampling effects)

- [ ] **B.2: Run +CBAM only variant (no Focal)**

```bash
cd /Users/kienha/spinet-v2
python3 train_rsna_attention.py \
    --use-cbam --augmentation none --oversample-factor 1 \
    --use-focal False --use-uncertainty=False \
    --epochs 30 --batch-size 32 --lr 1e-3 \
    --save-dir experiments/rsna/cbam_only/ 2>&1 | tee experiments/paper_results/logs/cbam_only.log
```

Note: the above assumes `--save-dir` and `--use-focal True/False` exist. If not, add them first in a small prep commit.

- [ ] **B.3: Run +Focal only variant (no CBAM)**

```bash
python3 train_rsna_attention.py \
    --no-cbam --augmentation none --oversample-factor 1 \
    --use-focal --use-uncertainty=False \
    --epochs 30 --batch-size 32 --lr 1e-3 \
    --save-dir experiments/rsna/focal_only/ 2>&1 | tee experiments/paper_results/logs/focal_only.log
```

- [ ] **B.4: Evaluate both with Stage A's script**

```bash
python3 evaluate_checkpoint_ivd.py \
    --checkpoint experiments/rsna/cbam_only/best_model_attention.pth \
    --model-type cbam \
    --output experiments/paper_results/table2_ablation/ivd_metrics/03_cbam_only.txt

python3 evaluate_checkpoint_ivd.py \
    --checkpoint experiments/rsna/focal_only/best_model_attention.pth \
    --model-type cbam \
    --output experiments/paper_results/table2_ablation/ivd_metrics/04_focal_only.txt
```

(Note: Variant 04 uses `--model-type cbam` but with `--no-cbam` at training time. This is because `train_rsna_attention.py` always builds `GradingModelWithCBAM` — which has `use_cbam=False` behavior internally. Verify by reading model source if uncertain.)

- [ ] **B.5: Commit**

```bash
git add experiments/paper_results/table2_ablation/ivd_metrics/03_cbam_only.txt experiments/paper_results/table2_ablation/ivd_metrics/04_focal_only.txt experiments/paper_results/logs/*.log
git commit -m "Add ablation variants: +CBAM only and +Focal only on RSNA"
```

### Stage B Done when

- [ ] `03_cbam_only.txt` and `04_focal_only.txt` exist.
- [ ] Severe Recall pattern visible: baseline ≈ 0%, CBAM-only > baseline, Focal-only > baseline, full > either individual — demonstrates complementarity.
- [ ] If CBAM-only alone already recovers Severe → the "Focal is necessary" claim weakens; update paper narrative accordingly (don't hide it).

---

## Stage C — Patient-level predictions for Table 1 integration (OPTIONAL)

Adds a "+CBAM+Focal (ours)" column to Table 1. Requires handling the data-leakage issue.

### Step-by-step

- [ ] **C.1: Check for data leakage between your train split and Table 1's 150 patients**

```bash
cd /Users/kienha/spinet-v2
python3 - <<'EOF'
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# User's training split
meta = pd.read_csv("rsna_preprocessed/train_metadata.csv")
patients = meta["study_id"].unique()
train_p, val_p = train_test_split(patients, test_size=0.2, random_state=42)
train_p, val_p = set(train_p.astype(int)), set(val_p.astype(int))

# Table 1's patients (from merged_binary_predictions.csv)
merged = pd.read_csv("experiments/paper_results/table1_baselines/metrics/merged_binary_predictions.csv")
t1 = set(merged["study_id"].astype(int))

print(f"Table 1 patients: {len(t1)}")
print(f"  In user train (LEAKAGE):  {len(t1 & train_p)} ({100 * len(t1 & train_p) / len(t1):.1f}%)")
print(f"  In user val (OK to use):  {len(t1 & val_p)} ({100 * len(t1 & val_p) / len(t1):.1f}%)")
print(f"  Neither:                  {len(t1 - train_p - val_p)}")
EOF
```

- [ ] **C.2: If leak > 10%, restrict CBAM evaluation to non-train patients**

Write `export_rsna_patient_predictions.py` (similar to `evaluate_checkpoint_ivd.py` but aggregates to patient-level with ANY-rule, AND filters to patients outside user train set).

- [ ] **C.3: If leak < 10%**, the Table 1 N=47 inner-join subset is effectively a fair test — proceed with direct export.

- [ ] **C.4: Export to CSV format expected by `evaluate_detection_abnormal.py`**

Target format:
```
study_id,spinal_canal_pred,spinal_canal_prob,left_foraminal_pred,left_foraminal_prob,right_foraminal_pred,right_foraminal_prob
```

Output path:
```
experiments/paper_results/table1_baselines/predictions/spinenetv2_cbam_focal_ours.csv
```

- [ ] **C.5: Re-run Table 1 evaluator with CBAM added**

Edit `/Users/kienha/thesis-experiments/detect-abnormal/evaluate_detection_abnormal.py` to register `SpineNetV2-CBAM-ours` in `MODELS` + appropriate loader. Re-run, re-copy outputs back to `table1_baselines/metrics/`.

- [ ] **C.6: Commit**

```bash
git add experiments/paper_results/table1_baselines/predictions/spinenetv2_cbam_focal_ours.csv
git commit -m "Add CBAM+Focal patient-level predictions to Table 1 comparison"
```

### Stage C Done when

- [ ] Table 1 has 4 model rows instead of 3, with CBAM+Focal included.
- [ ] Data leakage audit is documented in the paper's Limitations section.

---

## Stage D — 3-seed variance runs (OPTIONAL, FOR RANK-B)

Skip if Rank-C. Only run if Ablation Table 2 results are too close to call and you need statistical significance.

```bash
# Run each Stage B variant 2 more times with seeds 123 and 2024.
# Aggregate with mean +/- std across 3 seeds.
# Write helper: aggregate_3seeds.py that produces `ablation_mean_std.csv`.
```

Time: ~6 GPU-hours for 3 seeds × 4 variants.

---

## Definition of Done for Phase 2

Minimum (Rank-C paper + thesis):
- [ ] Stage A complete: `02_cbam_focal.txt` exists with correct format.
- [ ] Table 2 in paper shows 2 rows minimum: baseline vs +CBAM+Focal.

Recommended:
- [ ] Stages A + B complete: 4 ablation files, 4-row Table 2.

Full:
- [ ] Stages A + B + C complete: Table 1 includes CBAM column, Table 2 has 4 variants.

## Known risks

- **`--seed` / `--save-dir` flags don't exist in training scripts** → add them in a prep commit at the start of Stage B. Without them, runs overwrite each other's checkpoints.
- **`01_baseline.txt` was generated by a script that no longer exists** → if Stage A smoke test (A.5) fails to match numbers, investigate carefully. May need to accept slight differences (non-determinism in DataLoader workers).
- **CBAM-only (Stage B variant 03) actually matches full model in Severe recall** → would weaken "Focal is necessary" claim. Accept and report honestly; do not re-run with different settings to manufacture a gap.
