# SpineNetV2 + BiomedCLIP — Full Phase Report

> Date: 2026-04-27. Unified results across Phase 1+2 (RSNA), Phase 3 (SPIDER zero-shot), Phase 4 (SPIDER transfer).
> All numbers reproducible from `experiments/{fresh_baseline, fresh_cbam, hybrid, paper_results, spider_phase4}/`.

## Related docs

**Big picture & architecture:**
- [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) — high-level pipeline view, phase diagrams, status colors
- [`experiments/paper_results/HIGH_LEVEL_ARCHITECTURE.md`](paper_results/HIGH_LEVEL_ARCHITECTURE.md) — detailed Mermaid diagrams of the Hybrid dual-encoder
- [`experiments/hybrid_architecture.png`](hybrid_architecture.png) — visual diagram of CBAM + BiomedCLIP fusion
- [`STEPS.md`](../STEPS.md) — master execution plan and 3 contributions

**Phase-specific docs:**
- Phase 1+2 (RSNA): [`RSNA_PIPELINE.md`](../RSNA_PIPELINE.md) (preprocessing), [`HYBRID_TRAINING_GUIDE.md`](../HYBRID_TRAINING_GUIDE.md) (Hybrid model on RSNA), [`experiments/fresh_cbam/RESULTS_SUMMARY.md`](fresh_cbam/RESULTS_SUMMARY.md) (detailed RSNA results)
- Phase 3 (SPIDER zero-shot): [`SPIDER_ZEROSHOT_SETUP.md`](../SPIDER_ZEROSHOT_SETUP.md) (setup), [`experiments/paper_results/spider_zeroshot/RESULTS_REPORT.md`](paper_results/spider_zeroshot/RESULTS_REPORT.md) (full results), [`SUMMARY_FOR_ADVISOR.md`](paper_results/spider_zeroshot/SUMMARY_FOR_ADVISOR.md) (advisor-friendly version)
- Phase 4 (SPIDER transfer): [`SPIDER_TRAINING_GUIDE.md`](../SPIDER_TRAINING_GUIDE.md) (general workflow), this report (Phase 4 4-way comparison)
- Workflow & restore: [`NEXT_SESSION_WORKFLOW.md`](../NEXT_SESSION_WORKFLOW.md), [`COMMANDS.md`](../COMMANDS.md)

## TL;DR

| Phase | Task | Best model | Mean F1 macro | Mean Acc |
|---|---|---|---|---|
| **1+2** | RSNA grading (supervised) | Hybrid fixed (E7) | **0.516** | 70.0% |
| **3** | SPIDER zero-shot (no training) | Naked BiomedCLIP | **0.394** | — |
| **4** | SPIDER transfer (supervised) | Hybrid frozen (E11) | **0.623** | 79.7% |

**Headline takeaways**:
1. Hybrid (CBAM + BiomedCLIP) gives the best RSNA result on the hardest classes (Severe foraminal F1 = 0.276 / 0.283 vs CBAM 0.180 / 0.197).
2. Zero-shot SPIDER works — Hybrid achieves F1 = 0.59 / 0.40 / 0.28 across easy / medium / hard tiers without ever seeing SPIDER labels.
3. On supervised SPIDER, Hybrid frozen beats both Vanilla and CBAM transfer baselines (+0.013 F1 macro over Vanilla, +0.026 over CBAM).
4. Full fine-tuning (`--unfreeze-cbam`) does NOT improve mean F1 over linear probing — gains on 3 conditions cancelled by Pfirrmann regression.

---

## Standardized metrics (used in all tables)

- **Primary**: Mean F1 macro (per condition averaged → mean across conditions)
- **Primary**: Mean Accuracy
- **Secondary, domain-specific**:
  - RSNA: Avg Severe F1 (clinical detection focus)
  - SPIDER zero-shot: per-tier F1 (easy / medium / hard)
- **Detail**: per-condition F1 macro, per-class precision/recall/F1 (in appendix)

---

## Phase 1+2 — RSNA Lumbar Stenosis Classification

> 📖 Architecture: [`HIGH_LEVEL_ARCHITECTURE.md`](paper_results/HIGH_LEVEL_ARCHITECTURE.md) · 📊 Detailed results: [`fresh_cbam/RESULTS_SUMMARY.md`](fresh_cbam/RESULTS_SUMMARY.md) · 🛠 Pipeline: [`RSNA_PIPELINE.md`](../RSNA_PIPELINE.md), [`HYBRID_TRAINING_GUIDE.md`](../HYBRID_TRAINING_GUIDE.md)

**Setup**: 3 conditions (spinal_canal, left_foraminal, right_foraminal) × 3 classes (Normal / Moderate / Severe). 80/20 patient-level split, 1942 val samples, `random_state=42`.

| Method | Mean F1 macro | Mean Acc | Avg Severe F1 | Notes |
|---|---|---|---|---|
| Baseline (no CBAM, no CW), E25 | 0.420 | 81.4% | 0.152 | Foraminal Severe collapses to F1=0.000 |
| CBAM + sqrt CW, E20 | 0.509 | 68.6% | 0.333 | Recovers Severe but loses overall acc |
| **Hybrid fixed + sqrt CW, E7** ⭐ | **0.516** | 70.0% | **0.353** | Best balanced; best foraminal Severe |

### Per-condition F1 macro

| Method | Spinal Canal | L. Foraminal | R. Foraminal |
|---|---|---|---|
| Baseline | 0.511 | 0.380 | 0.371 |
| CBAM | 0.646 | 0.427 | 0.453 |
| Hybrid | 0.582 | 0.476 | 0.490 |

**Reading**: CBAM dominates spinal canal (0.646 vs 0.582), but Hybrid's BiomedCLIP path adds the most value on the harder lateral conditions where 3D-only attention struggles.

### Severe-class F1 (clinical relevance)

| Method | Spinal Severe | L-Fora Severe | R-Fora Severe |
|---|---|---|---|
| Baseline | 0.457 | 0.000 | 0.000 |
| CBAM | 0.623 | 0.180 | 0.197 |
| Hybrid | 0.500 | **0.276** | **0.283** |

Hybrid Severe Recall: 0.75 / 0.38 / 0.36 — catches more Severe across all conditions (clinically the most important miss to avoid).

---

## Phase 3 — SPIDER Zero-shot Evaluation

> 📖 Architecture: [`HIGH_LEVEL_ARCHITECTURE.md`](paper_results/HIGH_LEVEL_ARCHITECTURE.md) (cosine-sim head) · 📊 Detailed results: [`spider_zeroshot/RESULTS_REPORT.md`](paper_results/spider_zeroshot/RESULTS_REPORT.md), [`SUMMARY_FOR_ADVISOR.md`](paper_results/spider_zeroshot/SUMMARY_FOR_ADVISOR.md) · 🛠 Setup: [`SPIDER_ZEROSHOT_SETUP.md`](../SPIDER_ZEROSHOT_SETUP.md) · 📈 Visuals: [`ablation_hybrid_vs_naked.png`](paper_results/spider_zeroshot/ablation_hybrid_vs_naked.png), [`accuracy_by_ivd_level.png`](paper_results/spider_zeroshot/accuracy_by_ivd_level.png)

**Setup**: SPIDER test set (1439 IVDs), 8 disease labels never seen during RSNA training. Prediction via cosine similarity between image embedding and BiomedCLIP-encoded text prompts.

### Per-tier comparison

| Tier | Diseases | Hybrid | Naked BiomedCLIP |
|---|---|---|---|
| Easy   | Disc_narrowing | **0.586** | 0.403 |
| Medium | Spondylolisthesis, Disc_herniation, Disc_bulging | 0.401 | **0.465** |
| Hard   | Modic, UP_endplate, LOW_endplate, Pfirrman | 0.277 | **0.338** |
| **Mean F1 macro (8 diseases)** | | 0.362 | **0.394** |

### Per-disease F1 macro

| Disease | Tier | Hybrid | Naked BiomedCLIP | Δ |
|---|---|---|---|---|
| Disc_narrowing | easy | **0.586** | 0.403 | +0.183 |
| Disc_bulging | medium | **0.613** | 0.363 | +0.250 |
| Disc_herniation | medium | **0.563** | 0.532 | +0.031 |
| Pfirrman_grade | hard | **0.155** | 0.152 | +0.003 |
| UP_endplate | hard | 0.467 | **0.569** | -0.102 |
| LOW_endplate | hard | 0.468 | **0.558** | -0.090 |
| Spondylolisthesis | medium | 0.028 | **0.500** | -0.472 |
| Modic | hard | 0.018 | **0.071** | -0.053 |

**Reading — selective transfer pattern**:
- **Hybrid wins** on disc-related diseases (narrowing, bulging, herniation, Pfirrman) — these align with the spinal canal stenosis features the CBAM 3D backbone learned on RSNA.
- **Naked BiomedCLIP wins** on vertebra/endplate diseases (Modic, UP/LOW endplate, Spondylolisthesis) — these are out-of-distribution for the RSNA-trained backbone, and the projection MLP trained on RSNA actively pushes embeddings *away* from useful regions.

This is a meaningful finding for the paper: the RSNA-trained image projection does not always help — it specifically helps when the target disease shares anatomical correlates with stenosis.

---

## Phase 4 — SPIDER Transfer Learning (supervised)

> 📖 Architecture: [`HIGH_LEVEL_ARCHITECTURE.md`](paper_results/HIGH_LEVEL_ARCHITECTURE.md) (Hybrid + 4 SPIDER heads) · 🛠 Setup: [`SPIDER_TRAINING_GUIDE.md`](../SPIDER_TRAINING_GUIDE.md) · 📂 Raw metrics: [`spider_phase4/`](spider_phase4/) (4 best_metrics.txt + training_log.csv per run) · 📈 Cross-phase plot: [`cross_phase_comparison.png`](cross_phase_comparison.png)

**Setup**: 4-way comparison on 4 SPIDER conditions (Pfirrmann 5-class, Modic 4-class, Disc_narrowing binary, Spondylolisthesis binary). Same train/val split, same hyperparameters where applicable.

| Model | Epochs | Mean F1 macro | Mean Acc | Trainable params |
|---|---|---|---|---|
| Vanilla SPIDER (E15) | 15 | 0.610 | 77.13% | 22M (full backbone) |
| CBAM SPIDER (E16) | 16 | 0.597 | 75.21% | 22M (full backbone + CBAM) |
| **Hybrid frozen (E11)** ⭐ | 11 | **0.623** | 79.68% | ~5M (proj + attn pool) |
| Hybrid unfreeze (E20) | 20 | 0.622 | 80.43% | ~95M (CBAM + proj + attn) |

### Per-condition F1 macro

| Condition | Vanilla | CBAM | Hybrid frozen | Hybrid unfreeze |
|---|---|---|---|---|
| Pfirrmann (5-cls) | 0.594 | 0.535 | **0.626** | 0.580 |
| Modic (4-cls) | 0.363 | 0.358 | 0.373 | **0.387** |
| Disc Narrowing (2-cls) | 0.851 | 0.862 | 0.865 | **0.878** |
| Spondylolisthesis (2-cls) | 0.633 | 0.634 | 0.627 | **0.643** |

### Key findings

1. **Hybrid frozen beats Vanilla and CBAM** — using only ~5M trainable params (linear probing on a frozen backbone), Hybrid matches the Vanilla model's full 22M-param fine-tune and beats it on F1 macro.
2. **CBAM transfer disappoints** — CBAM (frozen RSNA backbone + retrained heads) is *worse* than Vanilla, suggesting the RSNA-stenosis CBAM modules don't generalize cleanly to SPIDER conditions. This is consistent with Phase 3 showing CBAM features help only stenosis-aligned tasks.
3. **Full fine-tune doesn't help mean F1** — `--unfreeze-cbam` improves 3/4 conditions (Modic +0.014, Disc Narrowing +0.013, Spondylolisthesis +0.016) but Pfirrmann drops -0.046, netting ~0 change. Frozen Hybrid is the better default.
4. **Cheapest method wins** — Hybrid frozen has the lowest training cost and the best mean F1, which is the cleanest possible argument for the architecture in the paper.

---

## Cross-phase summary (for paper Table 1 / Table 3)

| Method | Phase | Trainable | Mean F1 macro | Mean Acc | Domain-specific |
|---|---|---|---|---|---|
| Baseline RSNA | 1 | full backbone | 0.420 | 81.4% | Severe F1 = 0.152 |
| CBAM RSNA | 2 | CBAM + heads | 0.509 | 68.6% | Severe F1 = 0.333 |
| **Hybrid RSNA** | 2 | ~5M | **0.516** | 70.0% | Severe F1 = **0.353** |
| Hybrid zero-shot SPIDER | 3 | none | 0.362 | — | Easy 0.586 / Med 0.401 / Hard 0.277 |
| Naked BiomedCLIP zero-shot | 3 | none | **0.394** | — | Easy 0.403 / Med 0.465 / Hard 0.338 |
| Vanilla SPIDER transfer | 4 | full backbone | 0.610 | 77.1% | — |
| CBAM SPIDER transfer | 4 | CBAM + heads | 0.597 | 75.2% | — |
| **Hybrid frozen SPIDER** | 4 | ~5M | **0.623** | **79.7%** | — |
| Hybrid unfreeze SPIDER | 4 | ~95M | 0.622 | 80.4% | — |

---

## Caveats / honest limitations

1. **Single-seed results** — best epoch + 1 seed for all runs. Multi-seed mean ± std is desirable but out of scope for the current iteration.
2. **Val set is patient-level held-out**, not a true blind test. SPIDER zero-shot (Phase 3) is the closest to a true held-out evaluation.
3. **Different loss recipes across phases** — RSNA uses focal + uncertainty + sqrt class weights; SPIDER uses cross-entropy. Direct architectural comparisons within a phase are clean; cross-phase F1 macro comparisons are illustrative, not statistical claims.
4. **Modic Type 1/3 F1 = 0** across all SPIDER models — only 2-3 val samples per minority class. Data limit, not model failure.
5. **Naked BiomedCLIP wins Phase 3 mean** — interpretable: the RSNA-trained projection over-fits to stenosis-related features, hurting generalization to vertebra/endplate diseases. This is a real finding the paper should engage with rather than smooth over.

---

## File pointers

| What | Path |
|---|---|
| RSNA Baseline | `experiments/fresh_baseline/best_metrics_baseline_full_e25.txt` |
| RSNA CBAM | `experiments/fresh_cbam/best_metrics_sqrt_cw_e20.txt` |
| RSNA Hybrid | `experiments/hybrid/best_metrics_hybrid_fixed_e7.txt` |
| SPIDER zero-shot (Hybrid) | `experiments/paper_results/spider_zeroshot/best_metrics.txt` |
| SPIDER zero-shot (Naked) | `experiments/paper_results/spider_zeroshot_naked_biomedclip/best_metrics.txt` |
| SPIDER Vanilla | `experiments/spider_phase4/best_metrics_baseline.txt` |
| SPIDER CBAM | `experiments/spider_phase4/best_metrics_cbam.txt` |
| SPIDER Hybrid frozen | `experiments/spider_phase4/best_metrics_hybrid_spider.txt` |
| SPIDER Hybrid unfreeze | `experiments/spider_phase4/best_metrics_hybrid_spider_unfreeze.txt` |
| Cross-phase bar chart | `experiments/cross_phase_comparison.png` |
| Architecture diagram | `experiments/hybrid_architecture.png` |
