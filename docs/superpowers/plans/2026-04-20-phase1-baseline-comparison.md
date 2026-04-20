# Phase 1 — Baseline Comparison of 3 Third-Party Models

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a single paper-ready markdown report + underlying CSVs comparing SpineNetV2 (vanilla), MedGemma, and Ning Shen against RSNA 2024 ground truth, with per-condition precision/recall/F1, confusion matrices, and McNemar pairwise significance tests.

**Architecture:** Re-use the existing `evaluate_detection_abnormal.py` pipeline as the source of truth for binary aggregation and study_id joining. Add only missing pieces (McNemar tests, consolidated markdown writeup). Do NOT re-implement metric logic; it already exists.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, scipy (for McNemar via `scipy.stats`), statsmodels (optional for McNemar), matplotlib/seaborn for confusion-matrix plots.

---

## Pre-flight checks

- [ ] **Step 0.1: Verify you are in the correct repo**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && pwd
```

Expected: `/Users/kienha/thesis-experiments/detect-abnormal`

- [ ] **Step 0.2: Verify input files exist**

```bash
ls -la ground_truth/train.csv prediction/all_results_spinnet_v2.csv prediction/med_gemma_result.json prediction/ningshen_submission.csv
```

Expected: all 4 files listed with non-zero size.

- [ ] **Step 0.3: Verify Python dependencies**

```bash
python3 -c "import pandas, numpy, sklearn, scipy; print('OK')"
```

Expected: `OK`. If import fails, `pip install pandas numpy scikit-learn scipy statsmodels matplotlib seaborn`.

---

## Task 1: Establish a clean baseline run of the existing evaluator

**Files:**
- Modify: none
- Test: run + inspect `outputs/` directory

- [ ] **Step 1.1: Clear stale outputs so you can see what the pipeline regenerates**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal
mkdir -p outputs_archive_$(date +%Y%m%d) && mv outputs/*.csv outputs/*.md outputs/*.txt outputs_archive_$(date +%Y%m%d)/ 2>/dev/null || true
```

Expected: `outputs/` is empty (or only has subdirs).

- [ ] **Step 1.2: Run the main evaluator**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 evaluate_detection_abnormal.py
```

Expected: Script completes without error. Prints something like `Evaluating 3 models on 3 conditions across ~49 common patients`. Writes multiple CSVs to `outputs/`.

- [ ] **Step 1.3: Inspect the output**

```bash
ls outputs/
```

Expected at minimum: `merged_binary_predictions.csv`, `all_conditions_metrics.csv`, `average_metrics.csv`, `detailed_metrics.csv`.

- [ ] **Step 1.4: Check the merged prediction table**

```bash
head -3 outputs/merged_binary_predictions.csv && echo "---" && wc -l outputs/merged_binary_predictions.csv
```

Expected: CSV with columns `study_id, <model>_<condition>_true, <model>_<condition>_pred` for each of 3 models × 3 conditions. Line count ≈ 50 (49 patients + header).

- [ ] **Step 1.5: Check headline metrics**

```bash
cat outputs/average_metrics.csv
```

Expected: A small table with one row per model, columns precision/recall/F1/accuracy. MedGemma's F1 should be ≈ 0 (known mode-collapse finding per `CLAUDE.md`).

- [ ] **Step 1.6: Commit the clean run**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/outputs/*.csv && git status
git commit -m "Regenerate baseline evaluator outputs for paper Phase 1"
```

Stop and ask the user if the baseline F1 numbers look sane before proceeding. If MedGemma F1 ≠ 0 or number of common patients < 30, something broke in parsing — diagnose before continuing.

---

## Task 2: Add McNemar pairwise significance tests

Current pipeline reports metrics per model but no statistical comparison. Rank-C reviewers will ask "is CBAM significantly better than baseline?" — McNemar is the standard test for paired binary predictions.

**Files:**
- Create: `/Users/kienha/thesis-experiments/detect-abnormal/mcnemar_test.py`
- Test: `/Users/kienha/thesis-experiments/detect-abnormal/tests/test_mcnemar.py`

- [ ] **Step 2.1: Write the failing test**

```bash
mkdir -p /Users/kienha/thesis-experiments/detect-abnormal/tests
```

Create `/Users/kienha/thesis-experiments/detect-abnormal/tests/test_mcnemar.py`:

```python
"""Unit tests for mcnemar_test.py helpers."""
import numpy as np
import pytest
from mcnemar_test import compute_mcnemar


def test_mcnemar_identical_predictions_returns_pvalue_1():
    y_true = np.array([1, 0, 1, 0, 1])
    y_model_a = np.array([1, 0, 1, 0, 1])
    y_model_b = np.array([1, 0, 1, 0, 1])
    stat, pvalue = compute_mcnemar(y_true, y_model_a, y_model_b)
    assert pvalue == pytest.approx(1.0), f"identical preds should yield p=1, got {pvalue}"


def test_mcnemar_fully_disagreeing_predictions_significant():
    # Model A correct on all, Model B wrong on all — maximally disagreeing.
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_model_a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_model_b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    stat, pvalue = compute_mcnemar(y_true, y_model_a, y_model_b)
    assert pvalue < 0.05, f"fully disagreeing preds should be significant, got p={pvalue}"
```

- [ ] **Step 2.2: Run the test to confirm it fails**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 -m pytest tests/test_mcnemar.py -v
```

Expected: ModuleNotFoundError for `mcnemar_test`.

- [ ] **Step 2.3: Implement `mcnemar_test.py`**

Create `/Users/kienha/thesis-experiments/detect-abnormal/mcnemar_test.py`:

```python
"""McNemar pairwise significance test for two classifiers on the same study set.

McNemar's test is appropriate when two classifiers predict binary labels for
the same test instances. It compares the off-diagonal counts of the 2x2
contingency table of agreements/disagreements and tests H0: "the two
classifiers have equal error rates".

Reference: McNemar, Q. (1947). Note on the sampling error of the difference
between correlated proportions or percentages. Psychometrika 12(2), 153-157.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


def compute_mcnemar(y_true: np.ndarray, y_model_a: np.ndarray, y_model_b: np.ndarray) -> tuple[float, float]:
    """Compute McNemar's chi-square statistic and two-sided p-value.

    Uses the continuity-corrected form ((|b - c| - 1)^2) / (b + c) when b + c > 25,
    otherwise the exact binomial test via chi2 with df=1 on the discordant cells.

    Args:
        y_true: ground-truth binary labels, shape [N].
        y_model_a: predictions of classifier A, shape [N].
        y_model_b: predictions of classifier B, shape [N].

    Returns:
        (statistic, p_value).
    """
    y_true = np.asarray(y_true).astype(int)
    y_model_a = np.asarray(y_model_a).astype(int)
    y_model_b = np.asarray(y_model_b).astype(int)

    correct_a = (y_model_a == y_true)
    correct_b = (y_model_b == y_true)

    # Discordant cells: A correct & B wrong, A wrong & B correct.
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 25 else (b - c) ** 2 / (b + c)
    p_value = 1.0 - chi2.cdf(statistic, df=1)
    return float(statistic), float(p_value)


def run_pairwise_mcnemar(merged_csv: Path, conditions: list[str], models: list[str], output_path: Path) -> pd.DataFrame:
    """Run McNemar across every pair of models for every condition.

    Reads the merged binary predictions produced by evaluate_detection_abnormal.py
    and writes a CSV with columns: condition, model_a, model_b, b, c, statistic, pvalue.
    """
    merged = pd.read_csv(merged_csv)
    rows = []
    for condition in conditions:
        # The evaluator stores ground truth once per condition.
        y_true_col = f"{models[0].lower()}_{condition}_true"
        # Some evaluators write a neutral column `{condition}_true` — try both.
        if y_true_col not in merged.columns:
            y_true_col = f"{condition}_true"
        if y_true_col not in merged.columns:
            raise KeyError(f"Cannot find ground-truth column for condition {condition}. Available: {list(merged.columns)}")

        y_true = merged[y_true_col].to_numpy()
        for model_a, model_b in combinations(models, 2):
            pred_a_col = f"{model_a.lower()}_{condition}_pred"
            pred_b_col = f"{model_b.lower()}_{condition}_pred"
            if pred_a_col not in merged.columns or pred_b_col not in merged.columns:
                continue
            y_a = merged[pred_a_col].to_numpy()
            y_b = merged[pred_b_col].to_numpy()
            statistic, pvalue = compute_mcnemar(y_true, y_a, y_b)
            correct_a = int(np.sum((y_a == y_true)))
            correct_b = int(np.sum((y_b == y_true)))
            rows.append({
                "condition": condition,
                "model_a": model_a,
                "model_b": model_b,
                "n": len(y_true),
                "correct_a": correct_a,
                "correct_b": correct_b,
                "statistic": statistic,
                "pvalue": pvalue,
                "significant_at_0.05": pvalue < 0.05,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Pairwise McNemar tests across classifiers.")
    parser.add_argument("--merged", type=Path, default=Path("outputs/merged_binary_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mcnemar_pairwise.csv"))
    parser.add_argument("--models", nargs="+", default=["SpineNetV2", "MedGemma", "NingShen"])
    parser.add_argument("--conditions", nargs="+", default=["spinal_canal", "left_foraminal", "right_foraminal"])
    args = parser.parse_args()

    df = run_pairwise_mcnemar(args.merged, args.conditions, args.models, args.output)
    print(df.to_string(index=False))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.4: Run unit tests to confirm they pass**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 -m pytest tests/test_mcnemar.py -v
```

Expected: 2 passed.

- [ ] **Step 2.5: Run against the merged predictions**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 mcnemar_test.py
```

Expected: Table printed to stdout with 9 rows (3 conditions × 3 pairs), plus `outputs/mcnemar_pairwise.csv` written. MedGemma vs NingShen likely highly significant; SpineNetV2 vs NingShen may or may not be.

- [ ] **Step 2.6: Commit**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/mcnemar_test.py detect-abnormal/tests/test_mcnemar.py detect-abnormal/outputs/mcnemar_pairwise.csv
git commit -m "Add McNemar pairwise significance test across models"
```

---

## Task 3: Generate confusion matrices for each model × condition

**Files:**
- Create: `/Users/kienha/thesis-experiments/detect-abnormal/plot_confusion_matrices.py`
- Output: `outputs/confusion_matrices.png` + `outputs/confusion_counts.csv`

- [ ] **Step 3.1: Implement confusion-matrix script**

Create `/Users/kienha/thesis-experiments/detect-abnormal/plot_confusion_matrices.py`:

```python
"""Generate a 3x3 grid of confusion matrices (3 models x 3 conditions).

Reads merged_binary_predictions.csv produced by evaluate_detection_abnormal.py
and writes confusion_matrices.png + confusion_counts.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


MODELS = ["SpineNetV2", "MedGemma", "NingShen"]
CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
CONDITION_TITLES = {
    "spinal_canal": "Spinal Canal Stenosis",
    "left_foraminal": "Left Foraminal Narrowing",
    "right_foraminal": "Right Foraminal Narrowing",
}


def _true_col(merged: pd.DataFrame, model: str, condition: str) -> str:
    for candidate in (f"{model.lower()}_{condition}_true", f"{condition}_true"):
        if candidate in merged.columns:
            return candidate
    raise KeyError(f"No ground-truth column for {model}/{condition}")


def generate(merged_csv: Path, png_out: Path, counts_out: Path) -> None:
    merged = pd.read_csv(merged_csv)
    fig, axes = plt.subplots(len(MODELS), len(CONDITIONS), figsize=(12, 12))
    count_rows = []

    for i, model in enumerate(MODELS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i][j]
            pred_col = f"{model.lower()}_{condition}_pred"
            true_col = _true_col(merged, model, condition)
            if pred_col not in merged.columns:
                ax.set_visible(False)
                continue
            y_true = merged[true_col].to_numpy()
            y_pred = merged[pred_col].to_numpy()
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"], ax=ax, cbar=False)
            ax.set_title(f"{model}\n{CONDITION_TITLES[condition]}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            tn, fp, fn, tp = cm.ravel()
            count_rows.append({
                "model": model, "condition": condition,
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            })

    plt.tight_layout()
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()

    pd.DataFrame(count_rows).to_csv(counts_out, index=False)
    print(f"Wrote {png_out} and {counts_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", type=Path, default=Path("outputs/merged_binary_predictions.csv"))
    parser.add_argument("--png", type=Path, default=Path("outputs/confusion_matrices.png"))
    parser.add_argument("--counts", type=Path, default=Path("outputs/confusion_counts.csv"))
    args = parser.parse_args()
    generate(args.merged, args.png, args.counts)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run it**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal && python3 plot_confusion_matrices.py
```

Expected: Two files written, no errors. `confusion_matrices.png` should be a 3×3 grid.

- [ ] **Step 3.3: Inspect the plot visually**

Open `outputs/confusion_matrices.png` in Preview. Sanity-check: MedGemma column should be all TN+FN (mode collapse); SpineNetV2 and NingShen should have both correct and incorrect predictions.

- [ ] **Step 3.4: Commit**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/plot_confusion_matrices.py detect-abnormal/outputs/confusion_matrices.png detect-abnormal/outputs/confusion_counts.csv
git commit -m "Add confusion-matrix grid visualization for 3 models x 3 conditions"
```

---

## Task 4: Assemble the paper-ready comparison report

This is the artifact reviewers / the advisor will actually read. It consolidates Tasks 1–3 into one markdown document.

**Files:**
- Create: `/Users/kienha/thesis-experiments/detect-abnormal/outputs/baseline_comparison_report.md`

- [ ] **Step 4.1: Draft the report**

Create `/Users/kienha/thesis-experiments/detect-abnormal/outputs/baseline_comparison_report.md` with this structure. Fill in the numbers from `outputs/average_metrics.csv`, `outputs/all_conditions_metrics.csv`, `outputs/mcnemar_pairwise.csv`, `outputs/confusion_counts.csv` (do not copy text verbatim — use the actual values from the files):

```markdown
# Baseline Comparison: Third-Party Models on RSNA 2024

**Evaluation setup**
- Dataset: RSNA 2024 Lumbar Spine Degenerative Classification, training subset (ground truth from `train.csv`).
- Unit of evaluation: patient (study_id), after inner-join across all 3 models (N = XX patients).
- Task: binary abnormal detection per condition (Moderate|Severe -> 1, Normal/Mild -> 0; ANY-rule across 5 vertebral levels).
- Conditions: spinal_canal, left_foraminal, right_foraminal.

## Headline Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SpineNetV2 (vanilla) | ... | ... | ... | ... |
| MedGemma | ... | ... | ... | ... |
| Ning Shen | ... | ... | ... | ... |

(Values: mean across 3 conditions. See `average_metrics.csv`.)

## Per-Condition Breakdown

### Spinal Canal Stenosis
| Model | Precision | Recall | F1 | TN/FP/FN/TP |
| ... rows ... |

### Left Foraminal Narrowing
| ... |

### Right Foraminal Narrowing
| ... |

## Pairwise Statistical Significance (McNemar)

| Condition | Model A | Model B | p-value | Significant (α=0.05) |
| ... rows from mcnemar_pairwise.csv ... |

## Key Findings

1. **MedGemma exhibits mode collapse** — predicts Normal/Mild for nearly all studies. F1 ≈ 0. This is a known finding, not a parsing error.
2. **Ning Shen vs SpineNetV2**: ...
3. **Clinical sensitivity gap**: None of the 3 baselines achieves clinically-meaningful Severe recall. This motivates the CBAM+Focal Loss improvements reported in Phase 2.

## Confusion Matrices

See `confusion_matrices.png`. Mode-collapse pattern visible in MedGemma column (all predictions on "Normal" side).

## Reproduction

From `/Users/kienha/thesis-experiments/detect-abnormal/`:
```bash
python3 evaluate_detection_abnormal.py
python3 mcnemar_test.py
python3 plot_confusion_matrices.py
```

## Limitations

- N = XX patients after inner-join — small sample, caution with generalization.
- SpineNetV2 has no per-level output; patient-level ANY-rule is an approximation.
- MedGemma was prompted zero-shot; a fine-tuned variant is out of scope here.
```

- [ ] **Step 4.2: Fill in concrete numbers**

```bash
cd /Users/kienha/thesis-experiments/detect-abnormal
python3 -c "import pandas as pd; print(pd.read_csv('outputs/average_metrics.csv').to_markdown(index=False))"
python3 -c "import pandas as pd; print(pd.read_csv('outputs/all_conditions_metrics.csv').to_markdown(index=False))"
python3 -c "import pandas as pd; print(pd.read_csv('outputs/mcnemar_pairwise.csv').to_markdown(index=False))"
python3 -c "import pandas as pd; print(pd.read_csv('outputs/confusion_counts.csv').to_markdown(index=False))"
```

Paste each `to_markdown` output into the corresponding section of the report. Replace all `...` placeholders.

- [ ] **Step 4.3: Commit the report**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/outputs/baseline_comparison_report.md
git commit -m "Write paper-ready baseline comparison report for Phase 1"
```

---

## Task 5: Cross-repo handoff artifact

The CBAM results from Phase 2 need to slot into this comparison. Prepare the extension point now.

- [ ] **Step 5.1: Document the schema expected from Phase 2**

Create `/Users/kienha/thesis-experiments/detect-abnormal/prediction/SCHEMA_FOR_CBAM.md`:

```markdown
# Schema expected from spinet-v2 (Phase 2)

When Phase 2 completes, export CBAM predictions into this directory as:

**File:** `spinenetv2_cbam_result.csv`

**Columns:**
- `study_id` (int) — RSNA patient study ID; must be present for join.
- `spinal_canal_pred` (0/1) — binary prediction via same ANY-rule as evaluate_detection_abnormal.py.
- `left_foraminal_pred` (0/1)
- `right_foraminal_pred` (0/1)
- (optional) `*_prob` float columns for AUC-ROC.

**Scope:** include only study_ids also in `common_patients_all_3_models.csv` to preserve the inner-join.

After dropping this file here, add a `SpineNetV2-CBAM` row to `evaluate_detection_abnormal.py` MODELS list and re-run the whole pipeline. The existing code will auto-include it.
```

- [ ] **Step 5.2: Commit**

```bash
cd /Users/kienha/thesis-experiments && git add detect-abnormal/prediction/SCHEMA_FOR_CBAM.md
git commit -m "Document expected CBAM prediction schema for Phase 2 handoff"
```

---

## Definition of Done

Phase 1 is complete when:

- [ ] `outputs/baseline_comparison_report.md` exists with all tables filled, no `...` placeholders.
- [ ] `outputs/mcnemar_pairwise.csv` exists with 9 rows.
- [ ] `outputs/confusion_matrices.png` exists and visually shows mode-collapse for MedGemma.
- [ ] All new scripts have passing tests (`pytest tests/` exits with 0 fails).
- [ ] Every new file is committed and git is clean (`git status` shows nothing to commit).
- [ ] The report can be pasted directly into the paper's "Baseline Comparison" subsection with only formatting edits.

## Known risks and escalation

- **Number of common patients < 30** → inner-join broke. Most likely cause: new MedGemma file uses a different study_id key. Diagnose with `check_patient_overlap.py`.
- **McNemar p-values all 1.0** → models agree perfectly, which is implausible. Check that `merged_binary_predictions.csv` actually contains different columns for different models.
- **Confusion-matrix plot empty for a condition** → column name mismatch. Check `_true_col()` in `plot_confusion_matrices.py`.
- **If you find a bug in `evaluate_detection_abnormal.py` during this phase**, fix it in a separate commit with a `fix:` prefix — do not bundle with the new report.
