# RSNA 3-seed + C2 — SOURCE OF TRUTH (MIWAI paper)

Last updated: 2026-06-03. This file is the canonical pointer for the RSNA
multi-seed experiment that closes reviewer concerns **C1 (single-seed)** and
**C2 (trivial-ensemble)** for the MIWAI paper (`paper/lncs_hk252/main.tex`).

If a future session is unsure where the numbers in Table 1 come from, START HERE.

---

## 1. Where the data lives

| What | Path |
|---|---|
| Trained checkpoints (4 configs x 3 seeds) | `checkpoints/v3_20260503/{baseline,cbam,hybrid,bmc_only}/` |
| Per-run metrics (the numbers in Table 1) | `*_best_metrics.json` inside those dirs |
| Training logs | `experiments/v3_rsna_multiseed/run_*_seed*.log` |
| C2 trivial-ensemble result | `experiments/v3_rsna_multiseed/trivial_ensemble_seed123.json` |
| Local backup tar (slim, ~700MB-1GB) | repo root `rsna_seed_backup_*.tar.gz` |

Checkpoint filename convention (seed tag = `_seed<S>`, empty for seed 42):
- baseline: `baseline/best_model_seed<S>.pth` + `baseline_seed<S>_best_metrics.json`
- cbam:     `cbam/best_model_attention_seed<S>.pth` + `attention_seed<S>_best_metrics.json`
- hybrid:   `hybrid/best_model_hybrid_seed<S>.pth` + `hybrid_seed<S>_best_metrics.json`  (14M = trainable-only, frozen backbones loaded from cbam ckpt + BiomedCLIP)
- bmc:      `bmc_only/best_model_hybrid_seed<S>_biomedclip_only.pth` + `hybrid_seed<S>_biomedclip_only_best_metrics.json`

Seeds used: **{42, 123, 456}** (same set as the SPIDER 3-seed runs).

---

## 2. How to regenerate the numbers (no GPU needed)

```bash
# Table 1 (3-seed mean +/- std + LaTeX rows). Reads the *_best_metrics.json above.
python3 scripts/aggregate_rsna_seeds.py --seeds 42 123 456

# Severe Recall (not in the aggregator's 9 metrics) — compute from per_class recall[2]:
#   see the one-off snippet in git history / ask; baseline 12.3%, hybrid 48.6% (3-seed)
```

C2 ensemble (needs GPU + data; already run, result saved):
```bash
bash scripts/run_ensemble_c2.sh 123   # -> experiments/v3_rsna_multiseed/trivial_ensemble_seed123.json
```

Scripts that produced all this (all on branch `biomedclip-integration`):
`run_rsna_multiseed.sh`, `scripts/{aggregate_rsna_seeds,trivial_ensemble_rsna,run_ensemble_c2,check_rsna_runs,backup_rsna_runs,preflight_rsna}.{sh,py}`.

---

## 3. Canonical numbers now in the paper (Table 1, 3-seed mean +/- std)

| Metric | SpineNetV2 | CBAM-only | BMC-only | Hybrid |
|---|---|---|---|---|
| Mean Accuracy | **81.0 ± 0.6** | 69.5 ± 1.0 | 71.2 ± 1.7 | 72.4 ± 0.3 |
| Mean F1 macro | 0.420 ± 0.010 | 0.477 ± 0.057 | 0.482 ± 0.019 | **0.527 ± 0.027** |
| Mean Recall macro | 0.412 ± 0.005 | 0.531 ± 0.068 | 0.532 ± 0.004 | **0.592 ± 0.026** |
| Mean Precision macro | 0.493 ± 0.028 | 0.499 ± 0.037 | 0.508 ± 0.037 | **0.526 ± 0.013** |
| Mean AUC macro | 0.835 ± 0.010 | 0.803 ± 0.050 | 0.826 ± 0.015 | **0.838 ± 0.018** |
| Mean AUPRC macro | 0.530 ± 0.012 | 0.498 ± 0.051 | 0.513 ± 0.030 | **0.533 ± 0.024** |
| Severe F1 | 0.152 ± 0.008 | 0.262 ± 0.082 | 0.268 ± 0.034 | **0.356 ± 0.017** |
| Severe AUC | 0.872 ± 0.011 | 0.865 ± 0.050 | 0.885 ± 0.010 | **0.899 ± 0.011** |
| Severe AUPRC | 0.284 ± 0.019 | 0.278 ± 0.099 | 0.275 ± 0.052 | **0.333 ± 0.042** |

**Severe Recall** (headline, not a Table 1 row): SpineNetV2 **12.3% ± 0.6** -> Hybrid **48.6% ± 1.9** (per-seed 46.4/49.7/49.7) ≈ 4x.

Hybrid wins every metric except Mean Accuracy (SpineNetV2 highest — expected, it predicts the majority class). The 3-seed std is small, so the original single-seed-42 numbers were NOT a lucky draw.

### C2 trivial-ensemble (seed 123) — answers "is hybrid just an ensemble?"
| | Ensemble(Base+BMC) | Hybrid(learned) |
|---|---|---|
| Mean F1 | 0.441 | **0.499** |
| Severe F1 | 0.181 | **0.348** |
| Mean Recall | 0.437 | **0.567** |
| Mean AUC | 0.851 | 0.822 |

Verdict: learned concat-MLP fusion beats trivial averaging on F1/Severe-F1/Recall (the deployment metrics), so it is NOT reducible to a logit average. (Ensemble has higher AUC/AUPRC = better ranking but worse thresholded Severe decisions — reported honestly in the paper.)

---

## 4. What changed in the paper (`paper/lncs_hk252/main.tex`) on 2026-06-03

- Table `tab:main` -> 3-seed mean ± std (`\footnotesize`), caption says "3 seeds {42,123,456}".
- Abstract / contribution-4 / "Severe class is the headline" / Summary -> 3-seed numbers (recall 12.3->48.6, Severe F1 0.152->0.356, AUC 0.899); removed all "single-seed" caveats.
- §4.5 "Why CBAM" -> in-domain Severe now: both branches contribute similarly (CBAM 0.262 ~ BMC 0.268), hybrid combines to 0.356 (the old "CBAM is the single largest source" was a seed-42 artifact, corrected).
- §4.5 -> C2 written as a real result (was a "hypothesis / future work").
- Limitations -> dropped the single-seed sentence; Summary -> dropped "trivial-ensemble control" + "three-seed RSNA sweep" from future work (both done).
- Build: 12/12 pages, 0 undefined.

### Zero-shot Table 2 (tab:spider) — REBUILT 2026-06-03 from reproducible source
The OLD Table 2 per-label F1/AUC (bulging 0.709, AUC 0.806, Pfirrmann AUC 0.492...) did NOT reproduce from any artifact (only the mean F1 0.362 matched). Rebuilt from `experiments/paper_results/spider_zeroshot/predictions.csv` (raw cosine sims). Recompute snippet: per disease, sklearn macro-F1 on (true,pred) + one-vs-rest macro AUC from softmax(sims). Authoritative hybrid numbers now in paper:
| Label | Hybrid F1 | Hybrid AUC | Naked F1 |
|---|---|---|---|
| Disc bulging | 0.613 | 0.829 | 0.363 |
| Disc herniation | 0.563 | 0.772 | 0.532 |
| Disc narrowing | 0.586 | 0.834 | 0.403 |
| Upper endplate | 0.467 | 0.740 | 0.569 |
| Lower endplate | 0.468 | 0.738 | 0.558 |
| Pfirrmann | 0.155 | 0.687 | 0.152 |
| Modic | 0.018 | 0.454 | 0.071 |
| Spondylolisthesis | 0.028 | 0.807 | 0.500 |
| **Mean** | **0.362** | 0.733 | **0.394** |
Naked = `experiments/paper_results/spider_zeroshot_naked_biomedclip/results.csv` (f1_macro). KEY honest finding: hybrid beats naked on the 3 disc-morphology labels (RSNA-aligned) but naked mean F1 (0.394) > hybrid (0.362) overall — paper says so. The old "Pfirrmann below-chance 0.492" claim was WRONG (real AUC 0.687); removed.

**C3 + C4 CLOSED (2026-06-03):** C3 = prompt strings printed in §4.3 ("a magnetic resonance image of [label]", a-priori, no leakage); prompts defined in `prepare_spider_zeroshot.py`. C4 = Naked-BiomedCLIP F1 column added to Table 2. All four reviewer concerns (C1/C2/C3/C4) now addressed. Build 12/12 pages.

Still open (lower priority): bootstrap CIs / more seeds (future work). See `reviews/EDITORIAL_DECISION.md`.

---

## 5. Related
- 5-reviewer review + decision: `paper/lncs_hk252/reviews/`
- Formula explainer: `paper/lncs_hk252/GIAI_THICH_CONG_THUC_VI.md`
- Overleaf bundle: `paper/lncs_hk252/HK252_MIWAI_overleaf.zip`
