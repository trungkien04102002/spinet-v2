# Paper Research Index — Master Dictionary

Compiled 2026-05-06. Click any link to jump to the file.

This is the **single entry point** for everything related to the
SpineNet+CBAM+BiomedCLIP paper (Rank-C target, EMBC 2027).

---

## 1. Paper itself (LaTeX source)

| File | What |
|---|---|
| [`main.tex`](main.tex) | Top-level compile entry. `latexmk -pdf main.tex` to build. |
| [`abstract.tex`](sections/abstract.tex) | Abstract + keywords. |
| [`01_introduction.tex`](sections/01_introduction.tex) | Introduction. |
| [`02_related_work.tex`](sections/02_related_work.tex) | Related work — **updated 2026-05-06 with all SOTA citations**. |
| [`03_methods.tex`](sections/03_methods.tex) | Methods — **added Sec.\ Fusion Design Rationale**. |
| [`04_experiments.tex`](sections/04_experiments.tex) | Experiments setup. |
| [`05_results.tex`](sections/05_results.tex) | Results — **added Sec.\ SOTA Table 1**. |
| [`06_discussion.tex`](sections/06_discussion.tex) | Discussion — **added Foundation Model future-work + foraminal limitation**. |
| [`07_conclusion.tex`](sections/07_conclusion.tex) | Conclusion. |
| [`references.bib`](references.bib) | BibTeX — **20+ new entries added**. |
| [`main.pdf`](main.pdf) | Built PDF (5 pages, last build 2026-05-06). |

## 2. Vietnamese advisor report (separate from main paper)

| File | What |
|---|---|
| [`summary_v3.tex`](summary_v3.tex) | Vietnamese 5-page advisor report. |
| [`summary_v3.pdf`](summary_v3.pdf) | Built PDF (5 pages). Latest commit `ed705ef`. |
| [`summary.pdf`](summary.pdf) | Original advisor-approved scope (reference). |

## 3. SOTA research files (research dump)

### Master aggregations (read these first)
| File | What |
|---|---|
| [`SOTA_DEEP_RESEARCH.md`](SOTA_DEEP_RESEARCH.md) | **Master Table 1 (22 rows) + drop-in defense + future-work + ranked action items** |
| [`PAPER_CITATIONS.md`](PAPER_CITATIONS.md) | **Flat lookup of all 60+ papers** — name + authors + link + key numbers, organised by 10 categories |
| [`SOTA_COMPARISON_RSNA.md`](SOTA_COMPARISON_RSNA.md) | First aggregation pass (Layer 1) — 14 papers, gap analysis |

### Layer 1 (broad surveys)
| File | What |
|---|---|
| [`research_kaggle_top.md`](research_kaggle_top.md) | RSNA 2024 Kaggle Top-10 (only 4 publicly documented; zero F1/AUC) |
| [`research_published_papers.md`](research_published_papers.md) | 14 peer-reviewed/arXiv papers (broad survey) |
| [`research_methods_survey.md`](research_methods_survey.md) | 17 candidate 3D backbones (CNN/Transformer/VLM) |

### Layer 2 (deep extractions)
| File | What |
|---|---|
| [`research_deep_5_papers.md`](research_deep_5_papers.md) | Full metric tables: Nigru, Lin, McSweeney (PMC), Hallinan, Aktan/Liu |
| [`research_papers_2025_2026.md`](research_papers_2025_2026.md) | 25+ newer 2025-2026 papers (Pfirrmann, foraminal, herniation, spond, multi-task) |
| [`research_foundation_models_spine.md`](research_foundation_models_spine.md) | 17 VLM survey + drop-in defense + future-work paragraphs |
| [`research_aktan_full_extraction.md`](research_aktan_full_extraction.md) | **Liu et al. 2025 IJCIS PDF extracted** — Table 6 per-class. Critical: paper is segmentation, not severity grading. |

### Dataset research
| File | What |
|---|---|
| [`research_mendeley_dataset.md`](research_mendeley_dataset.md) | **Mendeley Lumbar MRI family (5 sub-datasets)** — verdict GO with x6ggzp2ycn/1 (Pfirrmann 1-5, 1545 discs). PLOS ONE 2024 baseline 88.1% Acc. |

---

## 4. Code & experiments

### Source code
| Path | What |
|---|---|
| [`../spinenet/models/grading_attention.py`](../spinenet/models/grading_attention.py) | CBAM 3D ResNet34 |
| [`../spinenet/models/grading_hybrid.py`](../spinenet/models/grading_hybrid.py) | Hybrid CBAM + BiomedCLIP fusion |
| [`../spinenet/models/grading_baseline.py`](../spinenet/models/grading_baseline.py) | Baseline 3D ResNet34 (no attention) |
| [`../spinenet/models/biomedclip_wrapper.py`](../spinenet/models/biomedclip_wrapper.py) | BiomedCLIP frozen encoder wrapper |
| [`../train_rsna_baseline.py`](../train_rsna_baseline.py) | RSNA baseline training |
| [`../train_rsna_attention.py`](../train_rsna_attention.py) | RSNA CBAM training |
| [`../train_rsna_hybrid.py`](../train_rsna_hybrid.py) | RSNA Hybrid training |
| [`../train_spider.py`](../train_spider.py) | SPIDER transfer training |
| [`../eval_rsna_auc.py`](../eval_rsna_auc.py) | RSNA AUC/AUPRC standalone evaluator |
| [`../eval_spider_auc.py`](../eval_spider_auc.py) | SPIDER AUC/AUPRC standalone evaluator |
| [`../eval_zeroshot_spider.py`](../eval_zeroshot_spider.py) | SPIDER zero-shot evaluator |

### Visualization (post-experiment)
| Path | What |
|---|---|
| [`../viz/grad_cam_rsna.py`](../viz/grad_cam_rsna.py) | Grad-CAM 3×3 panel (Severe cases × Base/CBAM/Hybrid) |
| [`../viz/confusion_matrix_rsna.py`](../viz/confusion_matrix_rsna.py) | Per-condition CMs |
| [`../viz/pr_curves_rsna.py`](../viz/pr_curves_rsna.py) | PR curves for Severe class |
| [`../experiments/v3_20260503/figures/grad_cam_severe.png`](../experiments/v3_20260503/figures/grad_cam_severe.png) | Grad-CAM output |
| [`../experiments/v3_20260503/figures/confusion_matrix_cbam.png`](../experiments/v3_20260503/figures/confusion_matrix_cbam.png) | CMs output |
| [`../experiments/v3_20260503/figures/pr_curves_severe.png`](../experiments/v3_20260503/figures/pr_curves_severe.png) | PR curves output |
| [`../experiments/v3_20260503/RESULTS_LOG.md`](../experiments/v3_20260503/RESULTS_LOG.md) | Raw cat outputs from Vast.ai runs |

### Scripts
| Path | What |
|---|---|
| [`../scripts/run_cbam_v2bug.sh`](../scripts/run_cbam_v2bug.sh) | RSNA CBAM v2bug reproduce |
| [`../scripts/run_hybrid_full.sh`](../scripts/run_hybrid_full.sh) | RSNA Hybrid full |
| [`../scripts/run_bmc_only.sh`](../scripts/run_bmc_only.sh) | RSNA BMC-only ablation |
| [`../scripts/run_spider_baseline.sh`](../scripts/run_spider_baseline.sh) | SPIDER baseline |
| [`../scripts/run_spider_cbam.sh`](../scripts/run_spider_cbam.sh) | SPIDER CBAM |
| [`../scripts/run_spider_hybrid.sh`](../scripts/run_spider_hybrid.sh) | SPIDER Hybrid |
| [`../scripts/run_spider_zeroshot.sh`](../scripts/run_spider_zeroshot.sh) | SPIDER zero-shot eval |
| [`../scripts/run_spider_eval_all.sh`](../scripts/run_spider_eval_all.sh) | SPIDER 3-config orchestration |

---

## 5. Project documentation

| File | What |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | Codebase guide for Claude Code |
| [`../README.md`](../README.md) | Upstream SpineNet README |
| [`../SETUP_GUIDE.md`](../SETUP_GUIDE.md) | Local + Vast.ai onboarding |
| [`../RSNA_PIPELINE.md`](../RSNA_PIPELINE.md) | RSNA preprocessing + training |
| [`../SPIDER_TRAINING_GUIDE.md`](../SPIDER_TRAINING_GUIDE.md) | SPIDER transfer guide |
| [`../COMMANDS.md`](../COMMANDS.md) | Exhaustive command reference |
| [`../SUMMARY_v3.md`](../SUMMARY_v3.md) | Long-form Vietnamese results report (v3 complete) |
| [`../THESIS_REPORT_DRAFT_v2.md`](../THESIS_REPORT_DRAFT_v2.md) | Earlier thesis draft (v2) |
| [`../MOTIVATION_AND_DESIGN.md`](../MOTIVATION_AND_DESIGN.md) | Architecture justification |
| [`../ANSWERS_THAY_20260428.md`](../ANSWERS_THAY_20260428.md) | Advisor 2026-04-28 Q&A answers |
| [`../MEETING_NOTES_20260428.md`](../MEETING_NOTES_20260428.md) | Advisor meeting notes |
| [`../THESIS_PAPER_SCOPE_VI.md`](../THESIS_PAPER_SCOPE_VI.md) | Vietnamese paper scope |
| [`../RANK_B_SOTA_RESEARCH_PLAN.md`](../RANK_B_SOTA_RESEARCH_PLAN.md) | Earlier Rank-B research plan |

---

## 6. Quick navigation by question

### "What's our model and how does it differ from prior work?"
→ [`02_related_work.tex`](sections/02_related_work.tex) (Spine MRI Grading subsection) → cites Nigru, Lin, Hallinan, Liu, M-SCAN, SpineNetV2 family.

### "Which papers do I cite?"
→ [`PAPER_CITATIONS.md`](PAPER_CITATIONS.md) — flat lookup with name + link + key numbers.

### "What's the SOTA comparison table?"
→ [`05_results.tex`](sections/05_results.tex) Table 1 (in paper).
→ [`SOTA_DEEP_RESEARCH.md`](SOTA_DEEP_RESEARCH.md) Section 2 (full 22-row analysis).

### "Why BiomedCLIP and not [other VLM]?"
→ [`research_foundation_models_spine.md`](research_foundation_models_spine.md) — full survey + defense + future-work paragraphs.
→ [`03_methods.tex`](sections/03_methods.tex) Sec.\ Fusion Design Rationale (drop-in defense paragraph).
→ [`06_discussion.tex`](sections/06_discussion.tex) Sec.\ Limitations and Future Work on Foundation Models (drop-in future-work paragraph).

### "How does our F1 compare to Kaggle Top-10?"
→ Short answer: incomparable (Kaggle = log-loss, no F1). 
→ [`research_kaggle_top.md`](research_kaggle_top.md) for full analysis.

### "What about Mendeley dataset the advisor mentioned?"
→ [`research_mendeley_dataset.md`](research_mendeley_dataset.md) — verdict GO with x6ggzp2ycn/1 Pfirrmann subset.

### "What experiments to run next?"
→ [`SOTA_DEEP_RESEARCH.md`](SOTA_DEEP_RESEARCH.md) Section 6 — ranked action items (3-seed retrain, nshen7 inference, Hallinan κ).

---

## 7. External resources

### Datasets
- **RSNA 2024 LumbarDISC**: [Kaggle](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) · [Dataset paper](https://arxiv.org/abs/2506.09162)
- **SPIDER**: [Mendeley/Zenodo via paper](https://www.nature.com/articles/s41597-024-03090-w)
- **Mendeley k57fr854j2**: [Raw](https://data.mendeley.com/datasets/k57fr854j2/2) · [Pfirrmann subset](https://data.mendeley.com/datasets/x6ggzp2ycn/1)

### GitHub repos worth bookmarking
- [Bartley 2nd-place Kaggle](https://github.com/brendanartley/RSNA-2024-Competition)
- [tattaka 4th-place Kaggle](https://github.com/tattaka/rsna-2024-lumbar-spine-degenerative-classification-public)
- [nshen7 independent](https://github.com/nshen7/rsna-2024)
- [Original SpineNet (Windsor)](https://github.com/rwindsor1/SpineNet)
- [BiomedCLIP HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

---

## 8. Status legend

- ⭐ = directly comparable (Table 1 candidate)
- ✅ = completed
- 🔄 = pending re-evaluation
- 📥 = manual download required (now done for Liu 2025)
- ⏳ = action item pending

---

## 9. Last update history

| Date | What |
|---|---|
| 2026-05-06 | INDEX.md created (this file). Liu PDF extracted. Mendeley research done. Paper Table 1 + Methods + Discussion updated. |
| 2026-05-06 | Layer 2 deep research: 5 papers extracted, 25+ 2025-26 papers, 17 VLMs surveyed |
| 2026-05-06 | Layer 1 SOTA research: Kaggle Top-10, broad survey, methods landscape |
| 2026-05-05 | viz/ scripts (Grad-CAM, CM, PR curves) + figures |
| 2026-05-05 | summary_v3.pdf committed (commit `ed705ef`) |
| 2026-05-04 | v3 Vast.ai retrain complete (4 RSNA configs + 3 SPIDER + zero-shot) |
