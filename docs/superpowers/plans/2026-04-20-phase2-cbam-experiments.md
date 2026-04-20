# Phase 2 — CBAM + Focal Loss Experiments on RSNA

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce (1) RSNA binary predictions from the CBAM+Focal checkpoint formatted for the Phase 1 evaluator, and (2) a 4-variant ablation study with 3 seeds each demonstrating that each component (CBAM, Focal Loss) contributes to Severe-class recall improvement.

**Architecture:** The CBAM+Focal model and training scripts already exist in this repo. Three sub-phases: (A) export predictions from the existing `best_model_attention.pth` in the format Phase 1 expects; (B) run the ablation matrix of 4 variants × 3 seeds; (C) consolidate results into a paper-ready markdown report.

**Tech Stack:** PyTorch, existing `train_rsna_attention.py` / `train_rsna_baseline.py` / `test_rsna_preprocessed.py`, pandas, numpy, sklearn.

---

## Pre-flight checks

- [ ] **Step 0.1: Confirm working directory and branch**

```bash
cd /Users/kienha/spinet-v2 && pwd && git branch --show-current
```

Expected: `/Users/kienha/spinet-v2` and branch `train-attention` (or create a new feature branch if preferred).

- [ ] **Step 0.2: Verify checkpoints**

```bash
ls -la checkpoints/*.pth 2>/dev/null
```

Required at minimum: `best_model_attention.pth`. For ablation, you also need `best_model.pth` (baseline). If `best_model.pth` is missing, the ablation study in Part B will need to generate it.

- [ ] **Step 0.3: Verify preprocessed dataset**

```bash
ls rsna_preprocessed/volumes/ | head -5 && wc -l rsna_preprocessed/train_metadata.csv
```

Expected: multiple study_id directories present, metadata CSV has > 1000 rows.

- [ ] **Step 0.4: Confirm Python env**

```bash
source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected: CUDA True on the GPU machine. If running on laptop (CPU), skip Part B (training) and do only Part A + C.

---

## Part A — Export Phase-1-compatible predictions from existing CBAM checkpoint

This is fast and unblocks Phase 1's Table 4 without new training.

### Task A1: Write a prediction exporter

**Files:**
- Create: `export_rsna_predictions.py` (in repo root)
- Test: `tests/test_export_rsna_predictions.py`

- [ ] **Step A1.1: Write a smoke test**

```bash
mkdir -p /Users/kienha/spinet-v2/tests
```

Create `/Users/kienha/spinet-v2/tests/test_export_rsna_predictions.py`:

```python
"""Smoke test that export_rsna_predictions.py writes a csv with expected columns."""
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def test_export_produces_expected_columns(tmp_path):
    out_csv = tmp_path / "pred.csv"
    cmd = [
        sys.executable, str(REPO / "export_rsna_predictions.py"),
        "--checkpoint", str(REPO / "checkpoints" / "best_model_attention.pth"),
        "--model-type", "cbam",
        "--metadata", str(REPO / "rsna_preprocessed" / "train_metadata.csv"),
        "--data-dir", str(REPO / "rsna_preprocessed"),
        "--output", str(out_csv),
        "--dry-run",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    for col in ["study_id", "spinal_canal_pred", "left_foraminal_pred", "right_foraminal_pred"]:
        assert col in df.columns, f"missing column {col}"
```

- [ ] **Step A1.2: Run test, confirm it fails**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_export_rsna_predictions.py -v
```

Expected: ERROR (script doesn't exist yet).

- [ ] **Step A1.3: Implement `export_rsna_predictions.py`**

Create `/Users/kienha/spinet-v2/export_rsna_predictions.py`:

```python
"""Export RSNA predictions from a trained checkpoint in Phase-1 evaluator format.

Writes a CSV with columns:
  study_id, spinal_canal_pred, left_foraminal_pred, right_foraminal_pred
  (+ matching *_prob columns for AUC-ROC analysis)

Per-patient aggregation uses the ANY-rule across all IVD levels: a patient is
marked positive for a condition if ANY level is predicted as Moderate or Severe.
This matches the binary aggregation in
  /Users/kienha/thesis-experiments/detect-abnormal/evaluate_detection_abnormal.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_attention import GradingModelWithCBAM
from spinenet.models.grading_baseline import GradingModelBaseline


CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]


def build_model(model_type: str, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if model_type == "cbam":
        model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    elif model_type == "baseline":
        model = GradingModelBaseline(format="rsna")
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    rows = []
    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Predicting"):
            volumes = volumes.unsqueeze(1).to(device)
            outputs = model(volumes)
            for i in range(volumes.shape[0]):
                row = {"study_id": int(labels["study_id"][i].item()) if "study_id" in labels else -1}
                for cond in CONDITIONS:
                    logits = outputs[cond][i].cpu().numpy()
                    probs = np.exp(logits) / np.exp(logits).sum()
                    # class 0 = Normal/Mild, class 1 = Moderate, class 2 = Severe.
                    row[f"{cond}_class"] = int(np.argmax(probs))
                    row[f"{cond}_abnormal_prob"] = float(probs[1] + probs[2])
                rows.append(row)
    return pd.DataFrame(rows)


def aggregate_any_rule(df: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """ANY-rule aggregation per study_id: patient positive if any level is Moderate/Severe."""
    # Attach study_id from metadata if not already present.
    if "study_id" not in df.columns or (df["study_id"] == -1).all():
        df = df.copy()
        df["study_id"] = metadata["study_id"].values[: len(df)]

    result = []
    for study_id, group in df.groupby("study_id"):
        row = {"study_id": int(study_id)}
        for cond in CONDITIONS:
            any_positive = (group[f"{cond}_class"] >= 1).any()
            max_prob = float(group[f"{cond}_abnormal_prob"].max())
            row[f"{cond}_pred"] = int(any_positive)
            row[f"{cond}_prob"] = max_prob
        result.append(row)
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", choices=["baseline", "cbam"], required=True)
    parser.add_argument("--metadata", type=Path, default=Path("rsna_preprocessed/train_metadata.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("rsna_preprocessed"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="Use first 4 samples only (smoke-test mode).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RSNAPreprocessedDataset(data_dir=str(args.data_dir), split="train", transform=None)
    if args.dry_run:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(4, len(dataset)))))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = build_model(args.model_type, args.checkpoint, device)

    df = predict_all(model, loader, device)
    metadata = pd.read_csv(args.metadata)
    aggregated = aggregate_any_rule(df, metadata)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(args.output, index=False)
    print(f"Wrote {len(aggregated)} patient-level predictions to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step A1.4: Run the smoke test**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_export_rsna_predictions.py -v
```

Expected: 1 passed. If it fails with dataloader errors, inspect `rsna_preprocessed_dataloader.py` — the `__getitem__` tuple signature may need adjusting in `predict_all()`.

- [ ] **Step A1.5: Full export for CBAM checkpoint**

```bash
cd /Users/kienha/spinet-v2
python3 export_rsna_predictions.py \
    --checkpoint checkpoints/best_model_attention.pth \
    --model-type cbam \
    --output /Users/kienha/thesis-experiments/detect-abnormal/prediction/spinenetv2_cbam_result.csv
```

Expected: CSV written with ~1000 patient rows (exact number depends on preprocessing).

- [ ] **Step A1.6: Full export for baseline checkpoint (if available)**

```bash
ls checkpoints/best_model.pth && python3 export_rsna_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --model-type baseline \
    --output /Users/kienha/thesis-experiments/detect-abnormal/prediction/spinenetv2_baseline_v2_result.csv
```

If `best_model.pth` does not exist, skip this step and note it in the report as a gap to fill via Part B.

- [ ] **Step A1.7: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add export_rsna_predictions.py tests/test_export_rsna_predictions.py
git commit -m "Add RSNA prediction exporter for Phase 1 evaluator compatibility"
```

### Task A2: Extend Phase 1 evaluator to include CBAM

**Files:**
- Modify: `/Users/kienha/thesis-experiments/detect-abnormal/evaluate_detection_abnormal.py` (MODELS constant)

- [ ] **Step A2.1: Add CBAM entry to MODELS**

Open `/Users/kienha/thesis-experiments/detect-abnormal/evaluate_detection_abnormal.py`. Find the `MODELS` constant and the corresponding PATH constant (likely `SPINENET_CBAM_PATH` or add a new one). Register the new file following the existing SpineNetV2 pattern. The exact lines to edit depend on the current state of the file — read it first.

Concretely:
1. Add `SPINENET_CBAM_PATH = os.path.join(BASE_DIR, "prediction", "spinenetv2_cbam_result.csv")`.
2. Append `"SpineNetV2-CBAM"` to the `MODELS` list.
3. Add `"SpineNetV2-CBAM": "spinenet_cbam"` to `MODEL_PREFIXES`.
4. Add a loader function (or a branch in the existing parser) that reads the CBAM CSV using columns `spinal_canal_pred`, `left_foraminal_pred`, `right_foraminal_pred`. These are already binary and already ANY-rule-aggregated by the exporter, so no further transformation is needed beyond matching on study_id.

- [ ] **Step A2.2: Re-run the evaluator**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 evaluate_detection_abnormal.py
```

Expected: Output now includes `SpineNetV2-CBAM` rows in all metric tables.

- [ ] **Step A2.3: Re-run McNemar and confusion matrices**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal
python3 mcnemar_test.py --models SpineNetV2 MedGemma NingShen SpineNetV2-CBAM
python3 plot_confusion_matrices.py  # update MODELS constant inside first if needed
```

- [ ] **Step A2.4: Commit**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/evaluate_detection_abnormal.py detect-abnormal/outputs/
git commit -m "Integrate SpineNetV2-CBAM predictions into Phase 1 comparison"
```

---

## Part B — Ablation matrix: 4 variants × 3 seeds

Only required if you don't already have checkpoints for each variant. If you already have all 4 checkpoints × 3 seeds from earlier experiments, skip to Part C.

### Ablation matrix

| Variant | CBAM | Focal Loss | Augmentation | Oversampling | Purpose |
|---|---|---|---|---|---|
| A: baseline | No | No (CE) | None | No | Reference point |
| B: +CBAM only | Yes | No (CE) | None | No | CBAM contribution alone |
| C: +Focal only | No | Yes | None | No | Focal contribution alone |
| D: full (+CBAM+Focal+Aug+Oversample) | Yes | Yes | medium | 5x | Main proposed method |

Each variant × 3 seeds (42, 123, 2024) = 12 runs. Estimated time on one 3090: ~40 min/run = ~8 GPU-hours.

### Task B1: Automate the training sweep

**Files:**
- Create: `scripts/run_ablation_sweep.sh` (shell script)

- [ ] **Step B1.1: Write the sweep script**

Create `/Users/kienha/spinet-v2/scripts/run_ablation_sweep.sh`:

```bash
#!/usr/bin/env bash
# Run 4-variant x 3-seed ablation on RSNA.
# Usage: bash scripts/run_ablation_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source spinenet-venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p experiments/rsna_ablation

SEEDS=(42 123 2024)

for seed in "${SEEDS[@]}"; do
    echo "=== Seed $seed / Variant A: baseline ==="
    python3 train_rsna_baseline.py \
        --epochs 30 --batch-size 32 --lr 1e-3 \
        --save-dir experiments/rsna_ablation/A_baseline_seed${seed} \
        --seed ${seed} || echo "WARN: A seed ${seed} failed"

    echo "=== Seed $seed / Variant B: +CBAM only ==="
    python3 train_rsna_attention.py \
        --use-cbam --no-focal=False --use-uncertainty=False --augmentation none --oversample-factor 1 \
        --epochs 30 --batch-size 32 --lr 1e-3 \
        --save-dir experiments/rsna_ablation/B_cbam_only_seed${seed} || echo "WARN: B seed ${seed} failed"

    echo "=== Seed $seed / Variant C: +Focal only ==="
    python3 train_rsna_attention.py \
        --no-cbam --use-focal --use-uncertainty=False --augmentation none --oversample-factor 1 \
        --epochs 30 --batch-size 32 --lr 1e-3 \
        --save-dir experiments/rsna_ablation/C_focal_only_seed${seed} || echo "WARN: C seed ${seed} failed"

    echo "=== Seed $seed / Variant D: full ==="
    python3 train_rsna_attention.py \
        --use-cbam --use-focal --use-uncertainty=True --augmentation medium --oversample-factor 5 \
        --epochs 30 --batch-size 32 --lr 1e-3 \
        --save-dir experiments/rsna_ablation/D_full_seed${seed} || echo "WARN: D seed ${seed} failed"
done
echo "Ablation sweep complete. See experiments/rsna_ablation/ for checkpoints."
```

**Note on ablation script assumptions:** This script assumes `--save-dir` and `--seed` flags exist on the training scripts. If not, either: (a) add those flags in a separate preparatory commit to `train_rsna_baseline.py` and `train_rsna_attention.py`; or (b) manually move `checkpoints/best_model*.pth` after each run. Option (a) is cleaner.

- [ ] **Step B1.2: If `--seed` / `--save-dir` flags don't exist, add them**

Edit `train_rsna_baseline.py` and `train_rsna_attention.py` to accept:
```python
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save-dir', type=str, default='checkpoints')
```
And at start of `main()`:
```python
import torch, random, numpy as np
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
```
Replace hardcoded `Path('checkpoints')` with `Path(args.save_dir)`.

Commit this as a prep change:
```bash
git commit -am "feat: add --seed and --save-dir flags for ablation reproducibility"
```

- [ ] **Step B1.3: Make executable and dry-run first 2 minutes**

```bash
chmod +x /Users/kienha/spinet-v2/scripts/run_ablation_sweep.sh
# Dry-run: start it, confirm the first variant begins, then Ctrl+C.
bash /Users/kienha/spinet-v2/scripts/run_ablation_sweep.sh
```

Expected: Training output scrolls; stops cleanly on Ctrl+C.

- [ ] **Step B1.4: Launch the full sweep**

```bash
cd /Users/kienha/spinet-v2
nohup bash scripts/run_ablation_sweep.sh > experiments/rsna_ablation/sweep.log 2>&1 &
echo "PID: $!"
```

Monitor with `tail -f experiments/rsna_ablation/sweep.log`.

- [ ] **Step B1.5: After sweep completes, export predictions for each checkpoint**

```bash
cd /Users/kienha/spinet-v2
for dir in experiments/rsna_ablation/*/; do
    variant=$(basename "$dir")
    # Detect model type by directory name prefix.
    if [[ "$variant" == A_* ]]; then model_type="baseline"; else model_type="cbam"; fi
    ckpt="$dir/best_model_attention.pth"
    [[ "$variant" == A_* ]] && ckpt="$dir/best_model.pth"
    python3 export_rsna_predictions.py \
        --checkpoint "$ckpt" --model-type "$model_type" \
        --output "experiments/rsna_ablation/predictions/${variant}.csv"
done
```

- [ ] **Step B1.6: Commit sweep script and log**

```bash
git add scripts/run_ablation_sweep.sh experiments/rsna_ablation/sweep.log experiments/rsna_ablation/predictions/
# Do NOT commit *.pth — they are gitignored. Confirm with: git status.
git commit -m "Complete 4x3 ablation sweep on RSNA"
```

### Task B2: Aggregate ablation metrics

**Files:**
- Create: `experiments/rsna_ablation/aggregate_ablation.py`

- [ ] **Step B2.1: Write aggregation script**

Create `/Users/kienha/spinet-v2/experiments/rsna_ablation/aggregate_ablation.py`:

```python
"""Aggregate metrics across variants and seeds.

Reads each variant_seed.csv of binary patient-level predictions, joins against
RSNA ground truth, computes per-condition precision/recall/F1, then reports
mean +/- std across seeds per variant.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


REPO = Path(__file__).resolve().parents[2]
GT_CSV = REPO.parent / "thesis-experiments" / "detect-abnormal" / "ground_truth" / "train.csv"
PRED_DIR = REPO / "experiments" / "rsna_ablation" / "predictions"
CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]


def gt_to_patient_binary(gt: pd.DataFrame) -> pd.DataFrame:
    """Collapse GT to patient-level binary per condition using ANY-rule."""
    rows = []
    for study_id, group in gt.groupby("study_id"):
        row = {"study_id": int(study_id)}
        for cond_long, cond_short in [
            ("spinal_canal_stenosis", "spinal_canal"),
            ("left_neural_foraminal_narrowing", "left_foraminal"),
            ("right_neural_foraminal_narrowing", "right_foraminal"),
        ]:
            any_abnormal = False
            for level in LEVELS:
                col = f"{cond_long}_{level}"
                if col not in group.columns:
                    continue
                values = group[col].astype(str).str.lower()
                if values.isin(["moderate", "severe"]).any():
                    any_abnormal = True
                    break
            row[f"{cond_short}_true"] = int(any_abnormal)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_for_one_run(pred_csv: Path, gt_binary: pd.DataFrame) -> dict:
    pred = pd.read_csv(pred_csv).merge(gt_binary, on="study_id", how="inner")
    out = {}
    for cond in CONDITIONS:
        y_true = pred[f"{cond}_true"].to_numpy()
        y_pred = pred[f"{cond}_pred"].to_numpy()
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        out[f"{cond}_precision"] = p
        out[f"{cond}_recall"] = r
        out[f"{cond}_f1"] = f1
    return out


def main() -> None:
    gt = pd.read_csv(GT_CSV)
    gt_binary = gt_to_patient_binary(gt)

    rows = []
    for pred_csv in sorted(PRED_DIR.glob("*.csv")):
        match = re.match(r"([A-D]_[A-Za-z_]+)_seed(\d+)\.csv", pred_csv.name)
        if not match:
            continue
        variant, seed = match.group(1), int(match.group(2))
        metrics = metrics_for_one_run(pred_csv, gt_binary)
        rows.append({"variant": variant, "seed": seed, **metrics})

    df = pd.DataFrame(rows)
    summary = df.drop(columns=["seed"]).groupby("variant").agg(["mean", "std"]).round(3)
    summary.to_csv(PRED_DIR.parent / "ablation_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step B2.2: Run aggregation**

```bash
cd /Users/kienha/spinet-v2 && python3 experiments/rsna_ablation/aggregate_ablation.py
```

Expected: Printed table with 4 rows (variants) × columns precision/recall/F1 mean±std per condition. Saved to `experiments/rsna_ablation/ablation_summary.csv`.

- [ ] **Step B2.3: Commit**

```bash
git add experiments/rsna_ablation/aggregate_ablation.py experiments/rsna_ablation/ablation_summary.csv
git commit -m "Add ablation metrics aggregation across seeds"
```

---

## Part C — Assemble CBAM results report

**Files:**
- Create: `experiments/cbam_rsna_results.md`

- [ ] **Step C1: Draft the results markdown**

Create `/Users/kienha/spinet-v2/experiments/cbam_rsna_results.md`:

```markdown
# CBAM + Focal Loss on RSNA 2024 — Results

**Evaluation setup**
- Dataset: RSNA 2024 Lumbar Spine, 80/20 patient split (same split as baseline experiments).
- Unit: patient, ANY-rule across 5 vertebral levels.
- Binary task: Moderate|Severe -> 1.
- Conditions: spinal_canal, left_foraminal, right_foraminal.
- Seeds: 42, 123, 2024.

## Ablation summary

Fill from `experiments/rsna_ablation/ablation_summary.csv`:

| Variant | SC Precision | SC Recall | SC F1 | LF F1 | RF F1 |
|---|---|---|---|---|---|
| A: baseline | mean ± std | ... | ... | ... | ... |
| B: +CBAM | ... | ... | ... | ... | ... |
| C: +Focal | ... | ... | ... | ... | ... |
| D: full | ... | ... | ... | ... | ... |

(SC = Spinal Canal, LF = Left Foraminal, RF = Right Foraminal.)

## Per-class metrics (Variant D, best seed)

Report only the 3-class precision/recall/F1 for variant D on the best-performing seed (or best epoch). Fill from the training log's `print_per_class_metrics` output.

### Spinal Canal Stenosis
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal/Mild | ... | ... | ... | ... |
| Moderate | ... | ... | ... | ... |
| Severe | ... | ... | ... | ... |

(Repeat for Left/Right Foraminal.)

## Key Findings

1. **Severe-class recall**: Baseline (Variant A) has recall = 0 on Severe. Variant D achieves recall X%. The combination of CBAM + Focal is necessary — either alone (B, C) underperforms D.
2. **Accuracy trade-off**: Variant D has accuracy Y% vs baseline Z%. Small drop in accuracy is expected and acceptable for clinical screening use cases where missing Severe cases is more costly than false alarms.
3. **McNemar significance**: Variant D vs Variant A is statistically significant (p < 0.05) on all 3 conditions.

## Comparison with other baselines

See `/Users/kienha/thesis-experiments/detect-abnormal/outputs/baseline_comparison_report.md` for the unified table comparing Variant D with MedGemma, Ning Shen, and vanilla SpineNetV2.

## Reproduction

From `/Users/kienha/spinet-v2/`:
```bash
bash scripts/run_ablation_sweep.sh          # trains all 12 runs (~8 GPU-hours)
python3 experiments/rsna_ablation/aggregate_ablation.py    # builds summary CSV
```

## Limitations

- 3 seeds only — variance estimate is rough; ideally 5+ seeds for Rank-B venues.
- Fixed 80/20 split by study_id — no k-fold cross-validation (out of time/compute budget).
- Focal γ = 2.0 is not tuned via grid search.
```

- [ ] **Step C2: Fill in concrete numbers**

```bash
cd /Users/kienha/spinet-v2
python3 -c "import pandas as pd; print(pd.read_csv('experiments/rsna_ablation/ablation_summary.csv').to_markdown())"
```

Paste into the report, replacing placeholders.

- [ ] **Step C3: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add experiments/cbam_rsna_results.md
git commit -m "Write CBAM + Focal RSNA results report for paper"
```

---

## Definition of Done

Phase 2 is complete when:

- [ ] `experiments/cbam_rsna_results.md` exists with all 4 variants × 3 seeds in the summary table.
- [ ] `/Users/kienha/thesis-experiments/detect-abnormal/prediction/spinenetv2_cbam_result.csv` exists and matches Phase-1 schema.
- [ ] Phase 1 evaluator re-run includes SpineNetV2-CBAM in all tables.
- [ ] Per-class metrics (including Severe) are documented for Variant D.
- [ ] McNemar test shows Variant D > Variant A with p < 0.05 (if not, re-run and debug).
- [ ] No training checkpoints committed (they are gitignored).

## Known risks and escalation

- **GPU OOM on batch 32** → reduce to batch 16 in sweep script; adjust lr proportionally.
- **Variant B (CBAM only, no Focal) overfits severely on majority class** → expected. That's the point of the ablation — to show Focal is necessary.
- **Seed-to-seed variance > 5% on Severe F1** → the metric is noisy because Severe is rare. Report raw per-seed values in an appendix; don't hide variability.
- **Predictions file column mismatch with Phase 1 evaluator** → inspect `_parse_spinenet_cbam()` (if you added one) vs the CSV header; fix whichever is wrong.
- **If `best_model.pth` (baseline) does not exist before Part B**, Part A step A1.6 cannot run for baseline. Add baseline training (Variant A) at the start of Part B and run it first.
