# Context for Codex review: MIWAI 2026 paper 122 camera-ready (round 5)

## 0. Read this first: the baseline was wrong in the previous round

A previous review round was given the wrong baseline file and reached two false conclusions because of it. Do not repeat them.

**The accepted, submitted manuscript is `MIWAI_2026_paper_122_submmited.pdf` at the repo root.** That PDF, and only that PDF, is the baseline. It was verified byte-identical in extracted text to `overleaf.pdf` and to the `main.tex` inside the author's Overleaf zip.

**`paper/camera_ready/main_backup.tex` is a stale earlier draft. Ignore it completely.** The Overleaf project happened to contain two `.tex` files, `main.tex` (the real one, 33,628 bytes) and `main_backup.tex` (an old draft the author had left there, 37,358 bytes). The previous context file wrongly labelled the backup as the accepted version.

Concrete consequences, all verified by grep against the submitted PDF:

- The abstract that the previous round asked us to "restore" (beginning "severity classes are heavily imbalanced and large labelled volumetric corpora are scarce. In this paper, we present a two-branch hybrid multimodal model") appears **zero** times in the submitted PDF. It exists only in the stale draft.
- The "Hybrid AUC" column in the zero-shot SPIDER table (values 0.829, 0.772, 0.834, ...) appears **zero** times in the submitted PDF. It was never in the accepted paper, so nothing was removed. It exists only in the stale draft.
- The accepted paper says the aggregate SPIDER F1 gap is `+0.007`, not `+0.006`.
- The string `two-seed` appears **zero** times in the accepted paper, so no such qualification was dropped.

## 1. Situation

**Paper 122**, "A Two-Branch Hybrid Model for Imbalanced Lumbar Disc Grading and Zero-Shot Label Extension", MIWAI 2026 (Springer LNAI). **Status: ACCEPTED, oral.** Camera-ready stage: no further peer review. The chairs and the Springer typesetters are the only remaining gate.

**Deadline: 14 August 2026, 23:59 UTC-12.** Three files go to EasyChair: final PDF, zip of all source files, scanned copyright form signed by hand.

Authors: Trung Kien Ha (student, presenting online, 15 minutes), Trong Nhan Phan (advisor, corresponding author, will sign the copyright form). HCMUT and VNU-HCM.

## 2. Files

| File | Role |
|---|---|
| `MIWAI_2026_paper_122_submmited.pdf` (repo root) | **The accepted submission. The only baseline.** |
| `paper/camera_ready/main.tex` | Current camera-ready source. This is what will be submitted. |
| `paper/camera_ready/main.pdf` | Its build. 12 pages, clean. |
| `paper/camera_ready/RESPONSE_TO_REVIEWERS_gdoc.html` | Point-by-point response the advisor will read. |
| `paper/camera_ready/references.bib` | Two entries added (`khosla2020supcon`, `kendall2018multitask`). |
| `paper/camera_ready/Springer_Instructions_for_Authors_of_Proceedings_CS.pdf` | Official format rules. |
| ~~`paper/camera_ready/main_backup.tex`~~ | **Stale draft. Do not read, do not diff against.** |

## 3. Hard constraints

1. **Do not create any contradiction with the accepted paper.** The author is worried a reader comparing the two versions will suspect a post-acceptance method change. Every change must be defensible as a correction, a clarification, or a direct response to a reviewer.
2. **No experimental number may change.** No run was repeated. Any recommendation requiring new numbers is out of scope.
3. **The abstract must match the accepted one word for word.** It was modified once by mistake and has now been restored exactly. Verify this rather than assuming it.
4. **No em-dashes or en-dashes in prose.** Page ranges in the bibliography are the only exception; they come from the bst.
5. **Page count 12 to 15, strict.** Currently 12. 13 is legal, so do not recommend cutting text purely to hold 12.
6. **No AI-sounding prose.** No stock phrasing, no heavy mid-sentence bolding, no over-parallel structure.

## 4. The three reviews (verbatim from EasyChair)

### Reviewer 1, SCORE 0 (borderline paper)

> This paper proposed a two branches model for IVD grading, focus on the minority severe class, to address the imbalance issue in the dataset. As pointed out by the author themselves, all metrics in their experimental result of the proposed model remain low, which suggest that the models predictive power can be equivalent to a mere "guessing". Perhaps the main contribution is the suggestion of the two branches architecture that can be further refine. In terms of methodology, some notation in section 3.1 is not define clearly. Suggest to rewrite equation with number labelling and define the corresponding notation in each equation instead of lump equations in text. Some questions and minor comment: Why use capital letter in s for the word severe? What is the "c" and "d" in the feature map of the CBAM block? The Hybrid's F1 (0.527) in table 3 only exceeds others by 0.045, not 4.5 points as claim. If the value should be report in percentage, please amend accordingly. Author claim this statement "Severe Recall about 4x (12.3% to 48.6%)", but no recall for severe class provided in table 3.

Tracked as C1 to C6: guessing-level metrics, Section 3.1 notation, capital S, `c` and `d` in CBAM, 0.045 not 4.5 points (with the option to report in percentage), Severe Recall missing from Table 3.

### Reviewer 2, SCORE 2 (accept)

> This paper presents a technically well-designed hybrid framework for lumbar MRI grading that combines a CBAM-enhanced 3D ResNet with a frozen BiomedCLIP branch to jointly address severe class imbalance and label-space generalization. The idea of exploiting a frozen vision-language model to enable prompt-based extension to unseen grading schemas without retraining constitutes a meaningful contribution beyond conventional medical image classifiers. The experimental evaluation is comprehensive, including ablation studies, cross-dataset validation, imbalance-oriented metrics, and statistical significance testing, providing convincing evidence of the proposed method's effectiveness. The manuscript is generally well organized, the methodology is clearly motivated, and the reported improvements are technically plausible. Some aspects could be strengthened, particularly a deeper discussion of computational cost, prompt engineering sensitivity, and broader comparisons with recent vision-language medical foundation models. Nevertheless, these limitations are relatively minor and do not diminish the overall contribution. The work demonstrates a good balance between methodological novelty, practical relevance, and experimental rigor, making it a valuable contribution to MIWAI 2026.

Three items: computational cost, prompt engineering sensitivity, broader VLM comparisons. R2 calls them "relatively minor".

### Reviewer 3, SCORE 1 (weak accept)

> The paper's main novelty is that the trained fusion model can replace its supervised classifier heads with new text prompts and thereby perform "zero-shot label extension."
> 1. The claim depends on the fused representation remaining aligned with the BiomedCLIP text space. However, the fusion MLP is trained through task-specific supervised heads, and no loss explicitly constrains its output to match text embeddings.
> 2. The authors need to explain why cosine similarity between the fused representation and unseen prompts should be semantically mean

Point 2 is truncated in the notification email itself; it ends "semantically meaningful".

R3's premise, "the fusion MLP is trained through task-specific supervised heads", is factually wrong about the implementation, but only because the submitted paper said so. See section 5.

## 4b. Instructions from the chairs

> LNAI has been very strict about maintaining the quality of papers. We would appreciate all the authors to kindly pay serious attention to improve the paper as per requested by reviews.

> The final paper and the signed copyright form are due on 14 August 2026, 11:59PM UTC-12. ... uploading three files: 1) The final pdf file of the paper which has accommodated the reviewers' comments. 2) A zip file of all the source files ... 3) The pdf file of the signed copyright form.
> The page limit is 12-15 pages and is strict.

A point-by-point response letter is **not** one of the three required uploads and there is no EasyChair slot for it. It exists because the advisor asked for one.

## 5. Ground truth from the code and logs

R3's comment forced a check of the implementation against what the paper claimed. The paper was wrong in several places. All of the following was verified by reading source and run logs.

| Fact | Evidence |
|---|---|
| Multimodal-only and Hybrid have **no linear classification head**. Logits are `logit_scale.exp().clamp(max=100) * (image_emb @ text_embs.T)`, cosine against frozen BiomedCLIP text anchors, **already during supervised RSNA training**. | `spinenet/models/grading_hybrid.py`, `forward()` |
| Focal loss is applied to those cosine logits. | `train_rsna_hybrid.py` L637 |
| `logit_scale` is a **trainable** `nn.Parameter` initialised to `log(1/0.07)`, not a pretrained constant. | `grading_hybrid.py` |
| Per-task **supervised-contrastive** term, weight 0.1, temperature 0.07, masking `labels != -1`. | `train_rsna_hybrid.py`, `supervised_contrastive_loss()` |
| **UncertaintyLoss** (Kendall homoscedastic, learned log-variance per task) is in the optimizer. | `train_rsna_hybrid.py` L556; `spinenet/losses.py` |
| The **SpineNetV2 baseline uses `nn.CrossEntropyLoss`**, not focal. | `train_rsna_baseline.py`; `run_baseline_seed123.log` |
| **CBAM-only used focal `gamma=1.8` and oversampling `5x`**; Multimodal-only and Hybrid used `gamma=2.0` and `3x`. | `run_cbam_seed123.log`, `run_hybrid_seed123.log` |
| The augmentation used was a mode that **does not swap left/right labels** on horizontal flip. Log line: `⚠ Augmentation: medium (LEGACY v2 buggy mode — HFlip does NOT swap labels)`. | same logs |
| SPIDER supervised transfer for Hybrid also uses **cosine scoring**, updating `image_projection`, `slice_pool`, `logit_scale`. No new linear head. | `train_spider_hybrid.py`, `cosine_logits()` |

**Why the Section 3 correction is not a post-acceptance method change:** the submitted Figure 1 caption already said the fused vector is "scored by cosine similarity against text prompts, so the label set, whether a trained set or a new one, becomes an input at inference", and the submitted abstract already said "predictions are made by matching this embedding to text prompts". Only Sections 3.1 and 3.2 said "linear head". The submitted paper contradicted itself; the fix removes that contradiction.

## 6. What changed, accepted version to current

### 6.1 Camera-ready housekeeping

- Author block de-anonymised, formatted to match the same advisor's published MIWAI 2025 paper (LNAI 16354, pp. 396 to 408): `Dien Hong Ward` not "District 10", `Linh Xuan Ward` not "Linh Trung Ward, Thu Duc City" (both old names were dissolved in the 2025 administrative reorganisation), no trailing period on affiliation lines, emails braced as `{htkien.sdh242,nhanpt}@hcmut.edu.vn` under institution 1, corresponding mark rendered `1,2(✉)`.
- Acknowledgments and disclosure-of-interests block restored.
- No ORCID; the author chose to skip it, which Springer 5.3 permits.

### 6.2 LNAI format compliance

- **All colour removed.** `\best` was `\textcolor{best}{\textbf{#1}}`; now plain `\textbf{#1}`. Hyperref link colours all black. Springer 4.5 forbids colour in text, tables, equations.
- Float placement relaxed to `[htbp]` plus float-fraction tuning, to remove whitespace gaps.

### 6.3 Reviewer 1

- **C2.** Section 3.1 rewritten: three inline expressions became numbered equations (1) to (3), every symbol defined at first use.
- **C4.** CBAM feature map defined as `F ∈ R^{C×D×H×W}` with `C` channels over depth `D`, height `H`, width `W`; `c` (head function) and `d` (512) explicitly distinguished from the channel count. The focal-loss class weight was also renamed from `w_{c(t)}` to `w_{y_t}` to remove a second reuse of `c`.
- **C5.** "a 4.5-point gain" became "by 0.045 to 0.050 in absolute macro-F1". Absolute was chosen over percentage because every other row of that table is on the 0 to 1 scale.
- **C6.** New Table 3 row: `Severe Recall  0.123 / 0.288 / 0.316 / 0.486`, computed from the existing three seeds, no rerun, reported on the 0 to 1 scale to match the neighbouring `Mean Recall macro` row.
- C1 and C3 are answered in the response letter only.

### 6.4 Reviewer 2

- Discussion states cost on both sides: parameter-efficient to train (1.18 M of 260 M) but **not lightweight to deploy**, both frozen backbones still run at inference, 66.4 ms per IVD, about 16 GB peak GPU memory.
- Prompt sensitivity named in the Discussion as future work; Section 4.3 says the prompt template was written a priori with no target-set tuning.
- Related work notes controlled comparisons with newer medical VLMs are future work.
- New `tab:config` consolidating the training configuration.

### 6.5 Reviewer 3, the substantive change

- **Section 3.1** splits the head: Eq. (2) linear head scoped to SpineNetV2 and CBAM-only; Eq. (3) cosine head for Multimodal-only and Hybrid **already during source training**. Head symbol renamed `c^{zs}` to `c^{cos}`.
- Section 3.1 records that **SpineNetV2 uses cross-entropy** while the other three use focal.
- Removed the over-strong claim that `Φ` "produces an embedding in the same space as `ψ`".
- **Section 3.2** restates the same split; `s` is the **learned** logit scale, not "BiomedCLIP-pretrained".
- **Section 3.3** discloses the full objective: focal on cosine logits, per-task supervised contrastive (0.1, 0.07), homoscedastic-uncertainty task weighting. Two new citations.
- **Trainable-parameter list** in Section 3.2, Figure 1 caption and Table 2 caption now include the scalar logit scale and the per-task uncertainty scalars. The 1.18 M count is unchanged.
- **Discussion** adds the honest limitation: alignment to text holds only through the RSNA source anchors, with no additional image-text contrastive objective, so compatibility with unseen prompts is partial.

### 6.6 Accuracy fixes applied after the previous review round

These four came out of the previous round and were each verified against the submitted PDF or the run logs before being applied.

1. **Abstract restored word for word** to the accepted text. Two substantive words had been lost in earlier trimming: `helps regularize` had become `regularizes`, which made the claim **stronger** than the accepted version, and `(paired t-test, p=0.011)` had lost the test name. Verified: `diff` of the abstract block against the submitted source is now empty.
2. **`h-flip + L/R swap` corrected to `h-flip`** in the Table 2 configuration row. The accepted paper claimed a left/right label swap that the logs show did not happen.
3. **Focal `gamma` split per configuration:** "We set γ = 1.8 for CBAM-only and γ = 2.0 for Multimodal-only and Hybrid". The accepted paper stated a flat γ = 2.0, which is wrong for CBAM-only.
4. **Section 4.4 restored the head distinction:** "SpineNetV2 and CBAM-only fit lightweight task-specific linear heads, while Multimodal-only and Hybrid keep cosine scoring against SPIDER text anchors and update only the trainable projection, slice-attention pool, and logit scale." Earlier prose compression had dropped this, which made Section 4.4 contradict the Section 3 correction.

Note that items 2 and 3 correct errors that were **already present in the accepted paper**. They are not regressions introduced during camera-ready editing.

### 6.7 Other honesty fixes

- An uncited "approximately 80% sensitivity" clinical claim was removed.
- The Modic result is attributed to "an input limitation" rather than implying model failure.
- Table 5 caption: "significant gain over CBAM" became "exploratory gain over CBAM (n=3 seeds)".
- `Inference (ms/sa)` typo fixed to `(ms/IVD)`.
- Prose compressed throughout. No claim, number, table, figure, or citation was dropped.

## 7. Deliberately left alone

- Every experimental number in Tables 1 to 5.
- The decision not to run the prompt-sensitivity experiment R2 asked about; answered as a stated limitation. The advisor approved this.
- The decision not to benchmark other medical VLMs; it would require retraining all four configurations.
- ORCID.

## 8. Format checks already performed

- 12 pages, within the strict 12 to 15 range.
- No colour in text, tables, or equations.
- Table captions above the tabular in all five tables; figure captions below.
- All figures and tables cross-referenced; no orphan floats.
- Equations numbered consecutively, no section counters, not in colour.
- `\titlerunning`, `\authorrunning`, acknowledgments, disclosure all present.
- Corresponding author marked; the marvosym `\Letter` glyph renders.
- Body 10pt, tables 7 to 8pt, in-figure text about 7 to 9pt, all above the Springer 6pt floor. All fonts embedded.
- Figure 1 is vector PDF; raster figures about 900 dpi (plots) and 460 dpi (Grad-CAM halftone).
- Clean build: no errors, no undefined references, no `??`, no em or en dashes in prose.
- `splncs04` bibliography, 22 references, 9 of 28 bib entries carry DOIs.

Known deviations, judged acceptable:

- Page size is US Letter, not A4. The accepted submission was also Letter.
- `\renewcommand{\figurename}{Figure}` overrides the LNCS default "Fig.". Also true of the accepted submission.
- The preamble redefines `\thebibliography` to tighten `itemsep`. Springer asks authors to avoid self-defined environments, so a typesetter may strip it. Judged not worth the layout risk to remove, but say if you disagree.

## 9. What we want checked

### 9.1 Consistency and honesty

1. Diff `main.tex` against `MIWAI_2026_paper_122_submmited.pdf` yourself. Does anything read as a method change made after acceptance rather than a correction or a response to a reviewer? Section 3 is where to look hardest.
2. Are Sections 3.1, 3.2, 3.3 and 4.4 mutually consistent, and consistent with the Figure 1 caption, the Table 2 caption and the abstract, on: which configurations use a linear head versus a cosine head; which loss each uses; and exactly which parameters are trained?
3. Is every claim in the response letter supported by the current manuscript, and does every "Where" cell point at a section that actually contains the change? One known gap to judge: the response says we "did not evaluate robustness to paraphrases, ensembles, or learned prompts", while the Discussion only says a "prompt-robustness study" is left to future work and Section 4.3 says the template was written a priori. Is the response overclaiming what the paper says?
4. Is anything in section 5 above still misdescribed in the paper?

### 9.2 Layout

The paper holds 12 pages tightly. `tab:config` must stay `[t]`, and `\FloatBarrier` must not be added; it was tried and pushed to 13 pages. If you propose text changes, say whether they add or remove lines. 13 pages is legal, so a change worth a page is acceptable; say so rather than silently trimming.

### 9.3 Format

Anything in the Springer instructions that section 8 missed or got wrong.

## 10. Previous rounds, resolved. Do not reopen.

- **Round 1** argued the submitted "linear head" description should be kept. **Reversed** after checking the code; see the end of section 5.
- **Round 2** proposed replacement wording that was code-inaccurate ("optimized through task-specific supervised heads", "pretrained fixed logit scale"). Not used.
- **Round 3** raised two real blockers, both fixed: the false claim that all four configurations use focal loss, and the omission of the uncertainty scalars from the trainable-parameter list.
- **Round 4** caught a wrong equation reference in Section 3.3. Fixed. Its optional `\FloatBarrier` suggestion was tried and reverted.
- **Round 5** (the round that used the wrong baseline) produced four blockers. Two were real and are now fixed: Section 4.4 contradicting the cosine-head correction, and the training configuration not matching the logs. One was half right: the abstract had indeed drifted, but the replacement text offered was from the stale draft; the true accepted abstract has now been restored. One was entirely false: the Table 4 AUC column, the `+0.006` figure, and the `two-seed` qualification, all artefacts of the stale draft.

Round 5 also suggested several cosmetic changes that were declined: unbolding the Table 3 header row, and deleting the `\thebibliography` override. Say if you think either is worth revisiting, but they are low priority.
