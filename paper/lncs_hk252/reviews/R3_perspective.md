# Perspective Review (VLM / Foundation Models)

*Reviewer 3 — cross-disciplinary perspective: CLIP-style models, zero-shot transfer, prompt engineering, multimodal fusion, foundation models for medical imaging. MIWAI 2026 (LNAI, double-blind).*

## Recommendation + 1-line justification

**Weak Accept (borderline, conditional on rebuttal).** The hybrid-fusion engineering and the honest "regularizer not booster" framing are credible and refreshingly un-oversold, but the zero-shot contribution — which is half the title — rests on an under-specified, non-reproducible prompt protocol and an unsubstantiated "Naked BiomedCLIP" comparison; both are fixable within the page budget and must be fixed.

## Scores (0–100)

| Axis | Score | Note |
|---|---|---|
| Technical Soundness (VLM) | 62 | 2.5D per-slice + attention pool is defensible; frozen-2D-on-3D-grayscale domain shift under-acknowledged. |
| Novelty Calibration | 58 | "Label space as input" is standard CLIP zero-shot repackaged; regularizer finding is the genuinely fresh part but slightly over-claimed. |
| Zero-shot Rigor | 38 | No prompt strings, no ensembling statement, single inherited temperature, missing baseline table. This is the weakest axis. |
| Cross-disciplinary Value | 70 | First BiomedCLIP-on-RSNA-2024 + RSNA→SPIDER zero-shot is a useful data point for the medical-VLM community. |
| Overall | 58 | Solid systems paper; the VLM half needs reproducibility repair. |

## Strengths

- **S1. The regularizer framing is honest and well-calibrated (Sec.~\ref{sec:disc}, Tab.~\ref{tab:spider_transfer_agg}).** The authors explicitly decline to claim the hybrid beats the baseline on aggregate SPIDER F1 ("$+0.006$ sits within seed noise"), and ground the one claim they do make (Hybrid–CBAM $+0.034 > 2\sigma$) in pooled-seed statistics. This level of restraint is rare and is the paper's strongest scientific feature from a foundation-model standpoint, where frozen-feature regularization claims are usually asserted, not measured.
- **S2. Unified supervised/zero-shot formalism (Eq.~\ref{eq:composition}, Eq.~\ref{eq:cosine}).** Casting both settings as $h = c \circ \Phi$ with only $c$ and $\mathcal{Y}$ changing is clean and makes the "label space as input" idea precise. The L2-normalized shared metric space ($\mathbb{S}^{d-1}$, $d=512$) is the correct construction for cosine zero-shot.
- **S3. Learned attention pool over slices (Eq.~\ref{eq:slicepool}) instead of mean pooling.** A single trainable vector $w_a$ is a minimal, well-motivated choice that keeps the frozen branch frozen while allowing lesion-salient slice up-weighting — appropriate restraint given only ~1.18M trainable params.
- **S4. Selective-transfer analysis (Sec. 4.3).** The observation that disc-semantic labels (bulging/herniation/narrowing) transfer while semantically distant labels (Modic, Pfirrmann, spondylolisthesis) collapse, and the attribution to "semantic proximity structure of the contrastive pretraining corpus," is exactly the right way to read a CLIP zero-shot result and shows genuine VLM literacy.
- **S5. AUPRC reported alongside AUC with prevalence baselines stated (Sec. 4.1).** Correct metric hygiene for imbalanced zero-shot; many CLIP-medical papers report only AUC and hide the ranking failure on rare positives.

## Weaknesses (issue / location / fix / severity)

**W1 — Prompt strings are never shown; protocol is not reproducible. [CRITICAL]**
*Location:* Eq.~\ref{eq:cosine}, Sec. 4.3, formalism uses abstract $\{q_k\}$ throughout; no actual prompt text appears anywhere in the paper. *Problem:* Prompt sensitivity is the single largest known confound in CLIP zero-shot — reported F1 can swing 10–20 points between "an MRI showing disc herniation" vs. "{label}" vs. a hand-tuned clinical phrasing. With mean zero-shot F1 = 0.362 and per-label values from 0.030 to 0.709, the reader cannot tell whether the spread reflects real semantic transfer or prompt luck. The entire zero-shot table (Tab.~\ref{tab:spider}) is currently irreproducible. *Fix:* Add a compact prompt-table (appendix or a 6-line listing) giving the exact $q_k$ for all 8 SPIDER labels; state whether negatives use a "no {pathology}" anti-prompt or a single-class threshold; and report whether the prompts were frozen a priori or selected on the SPIDER validation set (the latter would leak and must be disclosed). One table fits the page budget.

**W2 — No prompt-ensembling statement; single-prompt zero-shot is fragile and undisclosed. [MAJOR]**
*Location:* Eq.~\ref{eq:cosine}, Sec. 4.3. *Problem:* CLIP/BiomedCLIP zero-shot is standardly run with prompt ensembling (averaging text embeddings over many templates) precisely to damp single-prompt variance. The paper gives no indication whether $t_k$ is one embedding or an ensemble. If single-prompt, the numbers are upper/lower-bound-undefined; if ensembled, the templates matter (W1). *Fix:* State the ensembling policy explicitly; ideally report mean ± std of zero-shot F1 over a small template set (e.g., 5 templates) so a reader can see the prompt-variance band. Even one sentence + a std column materially raises rigor.

**W3 — "Naked BiomedCLIP" comparison is claimed but its numbers are absent. [MAJOR]**
*Location:* Sec. 4.3, final sentence ("the hybrid wins on disc-related labels"). *Problem:* This is the only experiment that isolates the contribution of the trainable fusion + CBAM to zero-shot transfer (i.e., what RSNA supervision actually buys over the off-the-shelf encoder). The claim is made qualitatively with no per-label or mean numbers, no table row, no delta. As written it is an unsubstantiated assertion. This is also the *fair* head-to-head the zero-shot story needs (see W4). *Fix:* Add a "Naked BiomedCLIP" column to Tab.~\ref{tab:spider} (8 labels + mean). This is cheap to run (frozen public weights, same prompts) and directly answers "is the supervised fusion doing anything zero-shot, or is BiomedCLIP doing all the work?"

**W4 — The zero-shot comparison vs. Baseline/CBAM is framed as a capability win, but it is a by-construction artifact. [MAJOR]**
*Location:* Sec. 4.3 ("the Baseline and CBAM-only configurations cannot do [zero-shot] because their fixed 3-class heads"); contribution bullet 3; Sec. 5 ("uniquely enables"). *Problem:* Saying the hybrid "uniquely enables" zero-shot because fixed-head models "cannot" is true but vacuous as a comparison — any model with a text-aligned embedding can do this; any model with a fixed head cannot. The honest baseline is *Naked BiomedCLIP* (W3), not Baseline/CBAM. As currently phrased the paper implies a competitive advantage where the difference is architectural definition. *Fix:* Reframe to: "By construction, only text-aligned branches (BMC-only, Hybrid) admit label-space extension; the meaningful zero-shot comparison is therefore against Naked BiomedCLIP (Tab. X), which isolates the value added by RSNA-supervised fusion." This costs nothing and removes a reviewer-bait overclaim.

**W5 — Single inherited temperature $s$ for an out-of-distribution label set. [MINOR–MAJOR]**
*Location:* Eq.~\ref{eq:cosine}, "the learned temperature (logit scale) inherited from BiomedCLIP pretraining." *Problem:* The pretrain logit scale is calibrated to the PMC-15M caption distribution. Reusing it unchanged for an OOD lumbar-MRI label set affects the softmax sharpness and therefore F1 at the argmax — and especially any thresholded binary decision (6 of 8 SPIDER labels are binary). Inheriting $s$ is a reasonable default, but its impact on the *binary* labels (where threshold = where the two-way softmax crosses 0.5) is non-trivial and unexamined. *Fix:* Note that for binary labels the decision is a single cosine threshold and report sensitivity to $s$ (or to the threshold) for at least one binary label; or justify why argmax over {positive-prompt, negative-prompt} is insensitive to $s$. Acknowledge in Limitations that $s$ was not re-tuned OOD.

**W6 — Frozen 2D encoder on 3D grayscale MRI: domain shift under-acknowledged. [MINOR]**
*Location:* Sec. 3.3 "BiomedCLIP branch (frozen, 2.5D)"; Related Work para on VLMs. *Problem:* The paper correctly notes radiology is a minority of PMC-15M, but does not engage with the deeper mismatch: BiomedCLIP was trained on *figure-caption* pairs — often multi-panel, annotated, captioned natural-resolution figures — whereas the input here is a single-channel grayscale IVD crop replicated to 3 channels and resized to 224. The per-slice 2.5D treatment (9 independent forward passes, no inter-slice 3D context inside the frozen branch) is defensible *because* the branch is frozen and the attention pool (Eq.~\ref{eq:slicepool}) reintroduces cross-slice weighting — but the paper should say *why* per-slice independence is acceptable here (it is: the CBAM-3D branch carries the volumetric context, the frozen branch only supplies a per-slice semantic prior). *Fix:* One sentence in Sec. 3.3 stating the division of labor (3D context ← CBAM branch; per-slice semantic prior ← frozen branch) turns an apparent weakness into a justified design choice. Also soften "2.5D" — true 2.5D usually implies 3-orthogonal-plane or adjacent-slice stacking; this is per-slice 2D + learned pooling, which is closer to a multi-instance attention pool.

**W7 — "Label space becomes an input" novelty is over-italicized. [MINOR]**
*Location:* Sec. 3.1 ("the label space becomes an \emph{input} to inference rather than a fixed architectural constant"). *Problem:* This is precisely the defining property of CLIP-style zero-shot classification (Radford et al. 2021) and open-vocabulary recognition broadly; it is not novel in itself. The paper does not currently cite the general CLIP zero-shot lineage at this point, which a VLM reviewer will read as either naivety or implicit over-claiming. *Fix:* Add a half-sentence + citation ("a standard property of contrastive image-text models~\cite{radford2021clip}, here applied to cross-*dataset* lumbar label-space extension") so the *application* is the claim, not the mechanism. The genuine novelty is the RSNA→SPIDER cross-dataset application and the regularizer finding — lean on those.

**W8 — Regularizer claim is plausible but the alternative (trivial ensemble) is only deferred, not bounded. [MINOR]**
*Location:* Sec.~\ref{sec:disc} ("A trivial-ensemble explanation ... cannot be fully ruled out and is left to future work."). *Problem:* From a foundation-model perspective, "frozen features resist overfitting and thus regularize a co-trained branch" is a *known and expected* effect (frozen backbones are a standard regularization/transfer technique), not a surprising discovery — so the novelty is in *demonstrating* it cross-dataset for this architecture, not in the mechanism. Leaving the logit-averaging ensemble unmeasured weakens the specific causal claim that fusion (not mere averaging) is responsible. *Fix:* Either run the cheap logit-average of Baseline+BMC-only (no training needed) and add one row, or calibrate the language: "consistent with the known regularizing effect of frozen foundation features" rather than presenting it as a novel finding.

## Opportunities (cross-disciplinary connections worth noting)

- **O1. Prompt ensembling + class-name engineering** (à la CLIP's 80-template ImageNet protocol, and CuPL/descriptor-based prompting) would likely lift the disc-semantic labels further and is a near-free win — worth at least a sentence pointing forward.
- **O2. The Modic/Pfirrmann zero-shot collapse is a publishable negative result.** It maps cleanly onto the medical-VLM finding that BiomedCLIP underperforms on fine-grained ordinal grading absent grade-specific captions in PMC-15M. Framing it that way (rather than as "domain gap") connects the paper to the medical-CLIP evaluation literature.
- **O3. RadCLIP slice-pooling adapter** is already cited as future work — the authors could note that their Eq.~\ref{eq:slicepool} attention pool is a lightweight precursor to it, strengthening positioning.
- **O4. Cross-attention / gated fusion** (FiLM, perceiver-style, or a small cross-attention between $f_\text{cbam}$ and the 9 $z_i$) is the obvious next fusion step; the concat-MLP is honestly scoped as minimal (W-none) and that scoping is fine.

## Questions to Authors

1. **What are the exact text prompts $q_k$ for each of the 8 SPIDER labels?** Were they fixed a priori or chosen on SPIDER validation data? (Reproducibility-critical — see W1.)
2. **Single prompt or prompt ensemble per label?** If single, what is the F1 variance across reasonable paraphrases? (W2)
3. **What are the Naked-BiomedCLIP per-label and mean zero-shot F1/AUC numbers?** Please add them to Tab.~\ref{tab:spider}. (W3)
4. **For the 6 binary SPIDER labels, how is the cosine decision made** — argmax over {positive-prompt, negative-prompt}, or a tuned threshold on the single positive-prompt similarity? How sensitive is binary F1 to the inherited temperature $s$? (W5)
5. **Is concat-MLP fusion applied to the L2-normalized $f_\text{bmc}$ and $f_\text{cbam}$, or raw embeddings?** Eq.~\ref{eq:fusion} concatenates before the final normalization — please confirm the normalization order, since it affects how the BiomedCLIP semantic geometry survives fusion into the text-aligned space.
6. **Does the trivial logit-average ensemble (Baseline + BMC-only) reproduce the hybrid's cross-dataset recovery?** This is cheap and would solidify the regularizer claim. (W8)

---

### Plain-text summary for synthesizer

**Recommendation: Weak Accept (borderline, conditional on rebuttal).** Honest, well-calibrated systems paper; the zero-shot / VLM half needs reproducibility repair.

**Top 3 issues by severity:**
1. **[CRITICAL]** Zero-shot prompt strings $\{q_k\}$ are never shown anywhere — protocol is not reproducible; prompt sensitivity is the #1 CLIP zero-shot confound and the entire SPIDER zero-shot table (Tab. 2) is currently un-replicable. Fix: add a prompt table + disclose a-priori-vs-validation selection.
2. **[MAJOR]** "Naked BiomedCLIP" baseline is claimed in prose but has no numbers/table row, even though it is the *only fair* zero-shot head-to-head (Baseline/CBAM "cannot do zero-shot" is by-construction, not a real comparison). Fix: add a Naked-BiomedCLIP column to Tab. 2.
3. **[MAJOR]** No prompt-ensembling statement and a single inherited pretraining temperature $s$ reused for an OOD label set (6 of 8 labels binary) — both standard CLIP rigor controls are missing/undisclosed.

**Calibration note:** "Label space becomes an input" is standard CLIP zero-shot, not novel (over-italicized); the *genuine* contributions are (a) the cross-dataset RSNA→SPIDER application and (b) the 3-seed-supported regularizer finding — though the latter is a *known* frozen-feature effect, demonstrated rather than discovered. The regularizer framing itself is the paper's best feature: appropriately restrained, statistically grounded ($>2\sigma$), and not over-sold.
