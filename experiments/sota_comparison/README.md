# External SOTA comparison (RSNA lumbar grading)

Compares our **Hybrid** grading model against recent external methods, retrained
on the **exact same patient split** as our paper (fair head-to-head), reporting
the same metrics as Table 1.

## One command (run on the RTX-4090)

```bash
cd /path/to/spinet-v2
git pull
bash experiments/sota_comparison/run_all.sh
```

This trains 2 baselines × 3 seeds (6 runs, ~2-4h total on a 4090), logs each run
to `logs/`, then auto-builds the comparison tables. Re-run `make_table.py` alone
anytime to rebuild the tables from existing checkpoints.

## What gets compared

| Config | Paradigm | Source |
|---|---|---|
| SpineNetV2 | 3D-ResNet34 | our baseline (numbers from `checkpoints/v3_20260503/`) |
| **Hybrid (Ours)** | CBAM-3D + BiomedCLIP, prompt head | ours (same checkpoints) |
| brendanartley | 2.5D CNN + BiLSTM + attention-pool | RSNA-2024 2nd place, adapted |
| transformer | 2.5D CNN + Transformer-encoder | 2024-style transformer aggregation |

**Protocol note (be honest in the report):** the two external baselines are
*re-implementations of the methods' core grading architectures* adapted to our
shared per-IVD crop representation `(9,112,224)`, trained on our exact
patient-level split (seeds 42/123/456, 80/20 by `study_id`), evaluated on the 3
conditions we grade (spinal canal + left/right foraminal). They are NOT the
authors' full end-to-end DICOM pipelines. Both external baselines train from
scratch (no compatible pretrained backbone).

## Files

```
run_all.sh          one-shot GPU runner -> trains all + builds tables
train_sota.py       training harness (--model {brendanartley,transformer});
                    patient split is byte-identical to train_rsna_baseline.py
make_table.py       builds comparison_table.md + .tex (quantitative + capability)
export_split.py     regenerates the fixed split under splits/ (already run)
models/             grading_brendanartley.py, grading_transformer.py
splits/             seed{42,123,456}_{train,val}_studies.txt (committed);
                    *_manifest.csv are gitignored (derivable via export_split.py)
external/           reference clone of brendanartley's repo (gitignored)
checkpoints/, logs/ training outputs (gitignored)
comparison_table.*  generated output tables
```

## Outputs

- `comparison_table.md` / `comparison_table.tex` — quantitative table
  (Mean Accuracy, Mean F1 macro, Severe F1, Severe Recall; mean ± std over 3 seeds)
  plus a qualitative **capability table** ("takes a new label set as input",
  cross-dataset transfer, zero-shot) — the "why they can't do what we do" table.
- The `.tex` capability table uses `\ding{55}`/`\ding{51}`, so the including
  document needs `\usepackage{pifont}`.

## Manual single run (debug)

```bash
export PYTHONPATH=/path/to/spinet-v2
python experiments/sota_comparison/train_sota.py --model brendanartley --seed 42 \
    --epochs 30 --batch-size 32 --lr 1e-3
python experiments/sota_comparison/make_table.py
```
