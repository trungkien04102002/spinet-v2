# Phase 3 — SPIDER Linear Probe External Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that CBAM-enhanced features trained on RSNA transfer to the SPIDER dataset's different label space (pfirrmann 5-class, spondylolisthesis binary, disc_herniation binary), via Linear Probing (freeze backbone + CBAM, train only new heads).

**Architecture:** `train_spider.py` and `grading_spider.py` already implement linear probing via the `--freeze-backbone` flag and `load_pretrained_rsna_backbone()` method. This phase runs 2 required setups × 3 seeds, aggregates metrics, and produces a paper-ready report. Full Fine-tune is added as an optional comparison if the advisor explicitly requests it.

**Tech Stack:** PyTorch, existing `train_spider.py` / `test_spider.py` / `spider_dataloader.py`, pandas, sklearn.

---

## BLOCKER: Advisor confirmation required

**Do not start Tasks 1–3 until advisor Nhan Phan confirms the Linear Probing interpretation.** The message drafted in earlier brainstorming should be sent first. Once advisor replies with either "yes Linear Probe" or any variant confirming this protocol, proceed.

If advisor says "also do Full Fine-tune", execute Task 4 as well. If advisor proposes a different protocol, **revise this plan** — do not force the existing code path.

---

## Pre-flight checks

- [ ] **Step 0.1: Verify SPIDER dataset is on disk**

```bash
ls /Users/kienha/spinet-v2/spider/ 2>/dev/null | head -5
```

Expected: Directories / `.mha` files. If empty or missing, **stop** — download SPIDER from https://spider.grand-challenge.org/ (~5GB) before proceeding.

- [ ] **Step 0.2: Verify both RSNA checkpoints**

```bash
ls -la /Users/kienha/spinet-v2/checkpoints/best_model.pth /Users/kienha/spinet-v2/checkpoints/best_model_attention.pth
```

Both files must exist. If `best_model.pth` (baseline) is missing, Phase 2 Part B must complete first to produce it — otherwise there is no baseline to compare against.

- [ ] **Step 0.3: Smoke-test the SPIDER dataloader**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 spider_dataloader.py
```

Expected: Self-test in `__main__` block runs successfully, prints number of samples + a sample volume shape of `(9, 112, 224)`. If it errors, fix the dataloader before any training.

- [ ] **Step 0.4: Smoke-test the SPIDER model**

```bash
cd /Users/kienha/spinet-v2 && python3 spinenet/models/grading_spider.py
```

Expected: Both `GradingModelSPIDERBaseline` and `GradingModelSPIDERCBAM` forward-pass successfully on dummy input.

---

## Task 1: Run Linear Probe × 2 setups × 3 seeds

### Setup A — Baseline (no CBAM) Linear Probe

Load RSNA baseline backbone, freeze, train SPIDER heads only.

### Setup B — CBAM Linear Probe

Load RSNA CBAM backbone + CBAM modules, freeze both, train SPIDER heads only.

- [ ] **Step 1.1: Add `--seed` and `--save-dir` flags to `train_spider.py` if missing**

Check `train_spider.py`. If those flags don't exist, add them analogously to Phase 2 Task B1.2. Commit the change separately:

```bash
cd /Users/kienha/spinet-v2 && git commit -am "feat: add --seed and --save-dir flags to train_spider.py"
```

- [ ] **Step 1.2: Create sweep script for linear probe**

Create `/Users/kienha/spinet-v2/scripts/run_spider_linear_probe.sh`:

```bash
#!/usr/bin/env bash
# Run SPIDER Linear Probe: 2 setups x 3 seeds = 6 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
source spinenet-venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p experiments/spider_linear_probe

SEEDS=(42 123 2024)

for seed in "${SEEDS[@]}"; do
    echo "=== Seed $seed / Setup A: Baseline Linear Probe ==="
    python3 train_spider.py --model baseline \
        --rsna-checkpoint checkpoints/best_model.pth \
        --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
        --seed ${seed} \
        --save-dir experiments/spider_linear_probe/A_baseline_seed${seed} \
        || echo "WARN: Setup A seed ${seed} failed"

    echo "=== Seed $seed / Setup B: CBAM Linear Probe ==="
    python3 train_spider.py --model cbam \
        --rsna-checkpoint checkpoints/best_model_attention.pth \
        --freeze-backbone --epochs 15 --batch-size 16 --lr 1e-3 \
        --seed ${seed} \
        --save-dir experiments/spider_linear_probe/B_cbam_seed${seed} \
        || echo "WARN: Setup B seed ${seed} failed"
done
echo "Linear probe sweep complete."
```

```bash
chmod +x /Users/kienha/spinet-v2/scripts/run_spider_linear_probe.sh
```

- [ ] **Step 1.3: Dry-run 2 minutes**

```bash
bash /Users/kienha/spinet-v2/scripts/run_spider_linear_probe.sh
# After first seed/setup begins training, Ctrl+C to confirm no crash.
```

- [ ] **Step 1.4: Full sweep**

```bash
cd /Users/kienha/spinet-v2
nohup bash scripts/run_spider_linear_probe.sh > experiments/spider_linear_probe/sweep.log 2>&1 &
echo "PID: $!"
```

Estimated time: ~15 min/run × 6 runs = ~1.5 GPU-hours.

- [ ] **Step 1.5: Commit script and log**

```bash
git add scripts/run_spider_linear_probe.sh experiments/spider_linear_probe/sweep.log
git commit -m "Run SPIDER Linear Probe sweep (2 setups x 3 seeds)"
```

---

## Task 2: Aggregate SPIDER metrics

**Files:**
- Create: `experiments/spider_linear_probe/aggregate_spider.py`

- [ ] **Step 2.1: Write aggregation script**

Create `/Users/kienha/spinet-v2/experiments/spider_linear_probe/aggregate_spider.py`:

```python
"""Aggregate per-task metrics across SPIDER linear probe runs.

Parses the validation metrics stored in each run's checkpoint (under key
'val_accuracies') and the final-epoch per-class metrics from the training log.
Produces a summary CSV with mean +/- std per setup per task.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "experiments" / "spider_linear_probe"

SETUPS = {"A_baseline": "Baseline", "B_cbam": "CBAM"}
TASKS = ["pfirrmann", "spondylolisthesis", "disc_herniation"]


def _parse_log_metrics(log_path: Path) -> dict:
    """Extract final-epoch per-task accuracy from train_spider.py stdout.

    train_spider.py prints a block ending with `Mean: XX.XX%` each epoch.
    We take the last block.
    """
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    # Last occurrence of each per-task line.
    found = {}
    for task in TASKS:
        matches = re.findall(rf"- {task.capitalize()}: ([0-9\.]+)%", text, re.IGNORECASE)
        if matches:
            found[task] = float(matches[-1])
    return found


def main() -> None:
    rows = []
    for run_dir in sorted(RUNS_DIR.glob("*_seed*")):
        match = re.match(r"([A-B]_[a-z]+)_seed(\d+)", run_dir.name)
        if not match:
            continue
        setup, seed = match.group(1), int(match.group(2))
        checkpoint = run_dir / "best_model_cbam.pth"
        if not checkpoint.exists():
            checkpoint = run_dir / "best_model_baseline.pth"
        metrics_row = {"setup": SETUPS.get(setup, setup), "seed": seed}
        if checkpoint.exists():
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            val_acc = ckpt.get("val_accuracies", {})
            for task in TASKS:
                if task in val_acc:
                    metrics_row[f"{task}_acc"] = float(val_acc[task])
        rows.append(metrics_row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found under", RUNS_DIR)
        return
    summary = df.drop(columns=["seed"]).groupby("setup").agg(["mean", "std"]).round(3)
    summary.to_csv(RUNS_DIR / "spider_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Run aggregation**

```bash
cd /Users/kienha/spinet-v2 && python3 experiments/spider_linear_probe/aggregate_spider.py
```

Expected: Printed 2-row table (Baseline vs CBAM) with per-task mean ± std accuracy.

- [ ] **Step 2.3: Per-class metrics from training logs**

The per-class F1 for SPIDER is printed every 5 epochs by `train_spider.py::print_per_class_metrics`. For each setup × seed, locate the last per-class block in the respective log and record it. Store the extracted blocks as:

```bash
grep -A 30 "Per-Class Metrics" experiments/spider_linear_probe/*/stdout.log 2>/dev/null > experiments/spider_linear_probe/per_class_raw.txt
```

(Or scroll the single `sweep.log` and pull out each per-class table.)

- [ ] **Step 2.4: Commit**

```bash
git add experiments/spider_linear_probe/aggregate_spider.py experiments/spider_linear_probe/spider_summary.csv experiments/spider_linear_probe/per_class_raw.txt
git commit -m "Aggregate SPIDER Linear Probe metrics across seeds"
```

---

## Task 3: Write paper-ready SPIDER report

**Files:**
- Create: `experiments/spider_linear_probe_results.md`

- [ ] **Step 3.1: Draft the report**

Create `/Users/kienha/spinet-v2/experiments/spider_linear_probe_results.md`:

```markdown
# SPIDER External Validation — Linear Probe Results

**Protocol**
- Load RSNA-pretrained checkpoint (baseline or CBAM).
- Freeze backbone (and CBAM, for CBAM variant) — only SPIDER-specific heads are trainable.
- Train 15 epochs on SPIDER train split, batch 16, AdamW lr 1e-3.
- Evaluate per-task (pfirrmann 5-class, spondylolisthesis binary, disc_herniation binary) on held-out SPIDER validation (80/20 patient split, seed-consistent across setups).
- Seeds: 42, 123, 2024.

## Headline results

Fill from `experiments/spider_linear_probe/spider_summary.csv`:

| Setup | Pfirrmann Acc | Spondylolisthesis Acc | Disc Herniation Acc |
|---|---|---|---|
| Baseline (no CBAM) Linear Probe | mean ± std | ... | ... |
| CBAM Linear Probe (proposed) | ... | ... | ... |

## Per-class F1 (best seed, Setup B / CBAM)

### Pfirrmann (5 classes)
| Grade | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | ... | ... | ... | ... |
| ... |

### Spondylolisthesis (binary)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| No | ... | ... | ... | ... |
| Yes | ... | ... | ... | ... |

### Disc Herniation (binary)
(same structure)

## Key Findings

1. **CBAM > Baseline on transfer:** CBAM Linear Probe outperforms Baseline Linear Probe by Δ% on [task], Δ% on [task]. This supports the claim that CBAM learns features that generalize to adjacent clinical tasks, not only the specific RSNA label space on which it was trained.
2. **No adaptation of backbone is necessary:** Since only the linear heads are trained, any performance improvement is attributable to the quality of the RSNA-learned representation — not to model capacity adapting to SPIDER.
3. **Statistical note:** We do not run McNemar here because the tasks are multi-class (pfirrmann) or evaluated on different label spaces than RSNA; paired tests across different classifications are not meaningful.

## Limitations

- Linear Probe is a conservative evaluation — it cannot capture cases where CBAM needs slight adaptation to new task. A Full Fine-tune variant would provide an upper bound (see appendix if run).
- SPIDER has fewer patients than RSNA (~250 vs ~2000). Small-N effects on per-class metrics for rare classes (e.g., Pfirrmann grade 5).
- 15 epochs is a compromise — longer runs may further improve the head but with diminishing returns since backbone is frozen.

## Reproduction

From `/Users/kienha/spinet-v2/`:
```bash
bash scripts/run_spider_linear_probe.sh
python3 experiments/spider_linear_probe/aggregate_spider.py
```
```

- [ ] **Step 3.2: Fill in real numbers**

```bash
cd /Users/kienha/spinet-v2
python3 -c "import pandas as pd; print(pd.read_csv('experiments/spider_linear_probe/spider_summary.csv').to_markdown())"
```

Replace placeholders in the report.

- [ ] **Step 3.3: Commit**

```bash
git add experiments/spider_linear_probe_results.md
git commit -m "Write SPIDER Linear Probe results report for paper"
```

---

## Task 4 (OPTIONAL) — Full Fine-tune comparison

**Only execute if the advisor explicitly asks for Full Fine-tune.**

- [ ] **Step 4.1: Create Full Fine-tune sweep script**

Create `/Users/kienha/spinet-v2/scripts/run_spider_finetune.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source spinenet-venv/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p experiments/spider_finetune

SEEDS=(42 123 2024)

for seed in "${SEEDS[@]}"; do
    python3 train_spider.py --model baseline \
        --rsna-checkpoint checkpoints/best_model.pth \
        --no-freeze --epochs 30 --batch-size 16 --lr 1e-4 \
        --seed ${seed} \
        --save-dir experiments/spider_finetune/A_baseline_seed${seed}

    python3 train_spider.py --model cbam \
        --rsna-checkpoint checkpoints/best_model_attention.pth \
        --no-freeze --epochs 30 --batch-size 16 --lr 1e-4 \
        --seed ${seed} \
        --save-dir experiments/spider_finetune/B_cbam_seed${seed}
done
```

- [ ] **Step 4.2: Run sweep, aggregate (re-using `aggregate_spider.py` with `--dir` flag if needed), append to report**

Follow the same pattern as Task 2 and Task 3, writing results into a new section `## Full Fine-tune (upper-bound comparison)` in `experiments/spider_linear_probe_results.md`.

Estimated time: ~1–2 GPU-hours × 6 runs = 6–12 GPU-hours.

- [ ] **Step 4.3: Commit**

```bash
git add scripts/run_spider_finetune.sh experiments/spider_finetune/
git commit -m "Add optional Full Fine-tune comparison on SPIDER"
```

---

## Definition of Done

Phase 3 is complete when:

- [ ] `experiments/spider_linear_probe_results.md` exists with all 2 setups × 3 seeds summarized.
- [ ] Per-class F1 for at least Setup B (CBAM) is documented for all 3 tasks.
- [ ] `spider_summary.csv` exists and shows mean ± std per setup.
- [ ] CBAM Linear Probe outperforms Baseline Linear Probe on at least one task — if it does NOT, this is a **negative result** and must be reported honestly in the paper (don't hide it).
- [ ] Full Fine-tune section (Task 4) is either completed or marked "Deferred to future work" with a clear reason.

## Known risks and escalation

- **SPIDER dataloader very slow on first epoch** (resampling from `.mha` each time) — acceptable, first epoch takes ~3× subsequent epochs. If it's > 10× slower, cache preprocessed `.npy` like the RSNA pipeline.
- **`load_pretrained_rsna_backbone()` drops many parameters** — expected (RSNA heads are skipped). Console should show e.g. "Loaded 120/130 parameters from RSNA". If it says "0/130", the state-dict keys don't match — investigate.
- **SPIDER pfirrmann accuracy ≤ 40% even for CBAM** → 5-class is hard with frozen features and small N. Report honestly. Consider this a motivation for Full Fine-tune if time allows.
- **Advisor reply never arrives** → send a follow-up message after 1 week. Do not silently run either variant without confirmation; misaligned experiments waste GPU time and invite redo.

## Escalation / next steps after Phase 3

Once all three phases complete:
1. Assemble unified Table 1 in the paper (RSNA comparison including CBAM) and Table 2 (SPIDER external validation).
2. Write Methodology section — reference existing docs: `spinenet/models/attention.py` (CBAM definition), `spinenet/losses.py` (Focal, Uncertainty), `spinenet/augmentation.py`.
3. Write Discussion focusing on: class-imbalance problem, clinical sensitivity priority, cross-dataset transferability of learned features.
4. Target venues: EMBC, SPIE Medical Imaging, MICCAI Workshops (per `project_scope_decision.md` memory).
