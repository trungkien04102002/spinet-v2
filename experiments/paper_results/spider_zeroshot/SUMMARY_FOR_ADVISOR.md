# Phase 3 — Zero-Shot SPIDER Evaluation: Summary for Advisor Review

**Date**: 2026-04-27
**Student**: Trung Kien (advisor: Nhan Phan)
**Paper target**: EMBC 2027 (Rank-C)

## What this experiment shows

The Hybrid SpineNet (CBAM + BiomedCLIP) trained on RSNA 2024 Lumbar Spine (3 stenosis conditions) is asked to predict 8 different disease labels on the SPIDER dataset (1439 IVDs from 208 patients) **without any fine-tuning on SPIDER**. Predictions are made by encoding disease text prompts via the frozen BiomedCLIP text encoder and running cosine similarity argmax against image embeddings.

This isolates the contribution of the BiomedCLIP integration: the only mechanism by which the model can recognize unseen labels is through alignment with biomedical text embeddings learned during RSNA training.

## Headline result — all three difficulty tiers meet the pre-registered targets

| Tier | Diseases | Avg F1 macro | Target (`STEPS.md`) | Status |
|---|---|---|---|---|
| Easy   | Disc_narrowing | **0.586** | ≥ 0.50 | ✅ +0.09 |
| Medium | Disc_bulging, Disc_herniation, Spondylolisthesis | **0.401** | ≥ 0.35 | ✅ +0.05 |
| Hard   | Modic, Pfirrmann, UP_endplate, LOW_endplate | **0.277** | ≥ 0.25 | ✅ +0.03 |

Per-disease zero-shot results on SPIDER (1439 IVDs):

| Disease | Tier | Type | F1 macro | Bal acc | AUC |
|---|---|---|---|---|---|
| Disc_narrowing    | easy   | binary  | **0.586** | 0.599 | 0.834 |
| Disc_bulging      | medium | binary  | **0.613** | 0.645 | 0.829 |
| Disc_herniation   | medium | binary  | **0.563** | 0.640 | 0.772 |
| UP_endplate       | hard   | binary  | 0.467 | 0.536 | 0.740 |
| LOW_endplate      | hard   | binary  | 0.468 | 0.539 | 0.738 |
| Pfirrmann_grade   | hard   | 5-class | 0.155 | 0.271 | —     |
| Spondylolisthesis | medium | binary  | 0.028 | 0.500 | **0.807** |
| Modic             | hard   | 4-class | 0.018 | 0.211 | —     |

Six of eight diseases produce usable zero-shot predictions (F1 ≥ 0.155 on multiclass, ≥ 0.46 on binary). Two diseases fail the argmax decision; for Spondylolisthesis the AUC = 0.807 still shows the underlying representation is informative — the failure is in the prompt-cosine threshold, not in the visual features.

There is no SpineNet-baseline comparison because the original SpineNet (Baseline / CBAM-only) has fixed prediction heads for the 3 RSNA conditions and cannot output predictions for unseen SPIDER labels. A meaningful zero-shot comparison is, however, possible against off-the-shelf BiomedCLIP — see the next section.

## Ablation — Hybrid vs Naked BiomedCLIP

To check whether the Hybrid's zero-shot performance comes from RSNA training or from BiomedCLIP alone, we re-ran the same 1439-sample test with **Naked BiomedCLIP**: pretrained BiomedCLIP image+text encoders, mean-pool over 9 slices, no SpineNet training, no CBAM, no projection MLP.

Result: **selective transfer**, not universal improvement.

| Tier | Hybrid F1 | Naked F1 | Winner |
|---|---|---|---|
| Easy   | **0.586** | 0.403 | Hybrid (+0.18) |
| Medium | 0.401 | **0.465** | Naked (+0.06) |
| Hard   | 0.277 | **0.338** | Naked (+0.06) |

Per disease: Hybrid wins on disc-related (Disc_narrowing +0.18, Disc_bulging +0.25, Disc_herniation +0.03) and ties on Pfirrmann. Naked BiomedCLIP wins on vertebra/endplate-related (UP_endplate −0.10, LOW_endplate −0.09, Modic −0.05) and on Spondylolisthesis (−0.47, due to the Hybrid argmax catastrophe documented above).

**Interpretation**: RSNA stenosis training transfers preferentially to disc-related zero-shot tasks (where the source semantics align), but degrades performance on tasks where RSNA is silent (endplate signal, spinal alignment, marrow changes). Hybrid is not a strict improvement over off-the-shelf BiomedCLIP across all 8 diseases — its value is concentrated on the semantically-related tasks.

This finding directly motivates Phase 4 (SPIDER transfer learning): for the diseases where zero-shot fails, fine-tuning on SPIDER's training subset should recover performance. Phase 3 and Phase 4 therefore tell complementary stories.

What we can / cannot claim from this evidence:

- ✅ "Hybrid generalizes to disc-related zero-shot tasks vs off-the-shelf BiomedCLIP"
- ✅ "RSNA training transfers selectively to semantically-related diseases"
- ❌ "Hybrid universally improves zero-shot over BiomedCLIP" (counter-example: Naked wins 2/3 tier averages)
- ❌ "CBAM specifically drives the improvement" (cannot isolate CBAM from the trained projection MLP without a third ablation that retrains projection without CBAM)

## Failure modes — argmax calibration, not feature collapse

We documented two failure modes via confusion matrices on the per-sample predictions (`predictions.csv`):

### Spondylolisthesis (AUC 0.81, F1 0.03)
The model predicts "spondylolisthesis with vertebral slippage" as positive for **all 1439 samples**, including the 1397 true negatives. Inspecting the cosine-similarity scores: the positive prompt is closer than the negative prompt for nearly every image, but the *order* of similarities (positives ranked higher than negatives) is correct — hence AUC = 0.807. The argmax decision is broken because the prompt geometry has no clean threshold; AUC shows the underlying representation is informative.

### Modic (4-class)
The text prompts route the argmax catastrophically toward class 3 (`"modic type 3 sclerotic endplate changes"`) — 1266 / 1439 samples are predicted as class 3, even though only 7 / 1439 truly are. Almost every truly healthy disc (885 / 930) gets misclassified as Modic type 3.

```
Modic confusion matrix (true rows × predicted cols)
            pred=0  pred=1  pred=2  pred=3
true=0          20      24       1     885   ← 885 / 930 healthy → predicted "type 3 sclerotic"
true=1           2       1       0       1
true=2          64      57       1     376
true=3           3       0       0       4
```

This is a known calibration failure mode of zero-shot CLIP on rare labels with similar-sounding prompts. The text embedding for "modic type 3 sclerotic endplate changes" sits in a region of text-space that overlaps heavily with most spine MRI images, dominating the argmax.

### Pfirrmann (5-class, F1 0.155)
The model effectively performs *binary* classification when given 5 prompts: it predicts class 1 ("normal disc") or class 3 ("moderate disc degeneration") for almost everyone. Classes 2, 4, 5 are essentially never predicted. F1 still beats the majority baseline (0.155 vs 0.089) because it correctly identifies many true class-1 cases (199 / 218 correct).

## Implications for the paper

1. **The main story is intact**: 6 / 8 diseases generalize zero-shot, all three tiers pass targets. The narrative — "BiomedCLIP integration enables label-space extension without retraining" — is supported by the easy and medium results.
2. **Modic and Spondylolisthesis should be honestly framed as failure cases** with explanation (prompt calibration, rare-label argmax issue), not hidden. AUC for Spondylolisthesis is the legitimate metric to report.
3. **Pfirrmann is a partial success** (5-class collapses to 2 buckets) — the model has the visual signal for "normal vs degenerated" but cannot resolve the 5-grade clinical scale without fine-tuning.

## Anatomical insight

We also observed an accuracy gradient by IVD level (`accuracy_by_ivd_level.png`):

```
ivd=1 (lowest, ~L5/S1)  -> 48% accuracy on Disc_narrowing
ivd=6 (upper lumbar)    -> 87% accuracy
```

Lower lumbar discs (L4/L5, L5/S1) carry more pathology variation and are clinically harder; the model is conservative and predicts more accurately in upper lumbar where most discs are healthy. This pattern is consistent with the clinical literature and suggests the failure mode at L5/S1 is *not* random — it concentrates on the harder cases.

## Verification we already performed

To rule out the result being an artifact:

1. **Checkpoint integrity**: the Hybrid `.pth` contains exactly 7 trainable tensors (logit_scale, slice_pool, image_projection); CBAM and BiomedCLIP are loaded separately from RSNA-trained / pretrained sources.
2. **No SPIDER training**: `train_rsna_hybrid.py` reads only RSNA preprocessed data; no SPIDER ingestion in the training graph.
3. **Determinism**: same image encoded twice → identical embedding.
4. **Embedding signal**: two different volumes → cosine sim 0.90 (similar-but-distinct, expected for spine MRI).
5. **Pattern**: zero-shot expectations (easy > medium > hard) match observed pattern. If labels had leaked, the pattern would be uniformly high.

## What is in this directory

| File | Content |
|---|---|
| `RESULTS_REPORT.md`                  | Full report: methodology, all metrics, ablation, failure modes |
| `SUMMARY_FOR_ADVISOR.md`             | This file — single-page overview for thầy |
| `best_metrics.json`                  | Machine-readable summary (Hybrid) |
| `best_metrics.txt`                   | Human-readable per-disease table (Hybrid) |
| `results.csv`                        | Long-format per-disease metrics (Hybrid) |
| `results_plot.png`                   | Hybrid F1 macro per disease, AUC overlay, tier targets |
| `ablation_hybrid_vs_naked.png`       | Side-by-side Hybrid vs Naked BiomedCLIP, plus Δ chart |
| `confusion_multiclass.png`           | Confusion matrices for Modic and Pfirrmann (argmax failure visualized) |
| `accuracy_by_ivd_level.png`          | Per-IVD-level accuracy curves (binary diseases) |
| `predictions.csv`                    | Per-sample predictions (1439 × 56) — true / predicted / cosine sims for all diseases |
| `sample_inputs.png`                  | 9-slice visualizations of 6 example IVD volumes (2 per tier) |
| `../spider_zeroshot_naked_biomedclip/` | Naked BiomedCLIP ablation outputs (results.csv, best_metrics.{txt,json}) |

The associated technical write-up is `SPIDER_ZEROSHOT_SETUP.md` at the repo root, which documents dataset preparation, orientation handling, crop strategy, train/val/test split rationale, and the train-vs-frozen component breakdown.

## Recommended next steps

1. **Phase 4 — SPIDER transfer learning** (Vast.ai GPU, ~5 hours): fine-tune Vanilla and CBAM SpineNet on SPIDER's official `training` subset, evaluate on `validation` subset. This directly addresses the selective-transfer finding: where zero-shot fails (endplate, Modic, Spondylolisthesis), can supervised fine-tuning on SPIDER recover performance? Phase 4 is the natural complement to Phase 3 — zero-shot for related concepts, fine-tune for distant concepts.
2. **Prompt engineering ablation** (optional): test if longer / more clinical prompt phrasings improve Modic and Spondylolisthesis without changing model weights.
3. **Cleaner CBAM-isolation ablation** (optional, ~3 hours GPU): retrain only the projection MLP without CBAM features (input dim 512 instead of 1024) to isolate CBAM contribution from the projection MLP. Required to claim CBAM specifically (rather than the integration as a whole) drives the improvement.
4. **Multi-seed validation** (optional, ~6 hours GPU): retrain Hybrid 2-3 times with different random seeds; report mean ± std on the disc-related diseases where Hybrid wins.
