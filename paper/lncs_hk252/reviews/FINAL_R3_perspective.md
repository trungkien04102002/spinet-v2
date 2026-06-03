# Final Review — Reviewer 3 (Cross-Disciplinary: Vision-Language / Foundation Models)

**Paper:** A Frozen BiomedCLIP Branch as a Cross-Dataset Regularizer for Imbalanced Lumbar Disc Grading and Zero-Shot Label-Space Extension
**Venue:** MIWAI 2026
**Reviewer perspective:** vision-language models, CLIP-family zero-shot, foundation-model adaptation

---

## Verdict

**Weak Accept (lean accept), conditional on a small set of fixable issues.**

From a VLM standpoint this is a much healthier paper than the typical "we bolted CLIP onto X and it improved everything" submission. The central VLM claim has been re-calibrated from an accuracy/F1 winner into a *regularizer* finding, and the zero-shot section now honestly reports that off-the-shelf BiomedCLIP beats the proposed hybrid on overall zero-shot F1 (0.394 vs 0.362), with the hybrid winning only on the disc-morphology subset. That honesty is the paper's strongest asset and should be preserved through revision. My concerns are about citation lineage, zero-shot methodology rigor, and a couple of residual framing slips — not about the integrity of the core result.

---

## 1. Novelty calibration — is the VLM contribution honestly framed?

**Largely yes. This is well-calibrated and is the paper's biggest improvement.** Specific evidence:

- The abstract and contributions list now lead with the *regularization* finding, not a "CLIP makes us SOTA" claim. The contributions are explicitly "ordered by the strength of the multi-seed evidence" (line 68), which is the right epistemic posture.
- Section 4.3/Table 2 reports that **off-the-shelf BiomedCLIP edges the hybrid on mean F1 over all 8 labels (0.394 vs 0.362)** and explicitly states "RSNA supervision helps only where the label is semantically close" (line 268). Conceding that your trained model loses to the zero-shot baseline on aggregate is unusual and credible.
- The supervised-transfer section (4.4) does *not* claim the hybrid beats SpineNetV2 ("we do not claim the hybrid surpasses it here," line 298) and frames parity honestly.
- The trivial-ensemble control (line 314, averaging SpineNetV2 + BMC-only softmax → F1 0.45/Severe 0.18 vs learned-fusion 0.53/0.36) is exactly the ablation a skeptical VLM reviewer would demand, and it is present. Good.

**Residual overclaim concerns (MAJOR/MINOR):**

- **(MAJOR) The "regularizer" interpretation rests on a single n=3 comparison and is causally underdetermined.** The entire regularizer narrative hinges on: CBAM-only F1 0.619 < SpineNetV2 0.646, and Hybrid 0.653 recovers above CBAM by +0.034. With three seeds and pooled σ≈0.012–0.013, +0.034 is ~2.6σ — descriptive, as the paper correctly hedges. But the *mechanism* claim ("BiomedCLIP acts as a cross-dataset regularizer that offsets CBAM's RSNA overfitting") is stronger than the evidence. An equally consistent explanation is simple capacity/feature dilution: concatenating any reasonably general frozen feature stream and re-learning the fusion would dampen CBAM's in-domain bias. The paper does not rule out that a *non-BiomedCLIP* frozen feature (e.g., an ImageNet ViT, or even frozen RSNA-pretrained ResNet features) would produce the same "recovery." Without that control, "BiomedCLIP as regularizer" is one plausible reading among several. **Recommend** softening the mechanism claim further (frame as "a frozen general-purpose branch" effect that BiomedCLIP instantiates) OR adding one ablation with a non-biomedical frozen branch. At minimum, explicitly name this as an unverified mechanism in Limitations.

- **(MINOR) "Regularizer" is used loosely vs the technical sense.** In learning theory a regularizer constrains the hypothesis class / penalizes complexity. Here the effect is "a frozen branch supplies invariant features that resist domain-specific overfitting" — closer to *feature-level domain robustness / implicit ensembling* than to regularization in the Tikhonov/dropout sense. The paper's own ensemble ablation (line 314) shows the effect is partly but not fully an ensembling effect. The term is defensible as a metaphor and the paper does hedge it, but I'd add one sentence acknowledging it is "regularization in the loose sense of trading in-domain fit for cross-domain stability, not an explicit penalty term." This is a recognized frozen-feature phenomenon (frozen pretrained features generalize better OOD than fully fine-tuned ones — cf. the linear-probing-vs-fine-tuning OOD literature), so the framing is appropriate once hedged; the paper should *cite* that lineage rather than coining the effect fresh.

- **(MINOR) "To our knowledge, no prior peer-reviewed work has applied BiomedCLIP to RSNA 2024..." (line 76)** is a fine novelty statement but is a negative-existence claim resting only on the authors' search. Keep it, but it is weak novelty on its own — the paper is right not to lean on it as a primary contribution.

---

## 2. Zero-shot rigor

**Strengths:** prompts are disclosed verbatim, declared a priori, and the no-leakage argument is made explicitly (line 242). The single-template choice is stated. The temperature is now concretely specified as the BiomedCLIP-pretrained logit scale clamped at 100 (line 158), which removes the earlier ambiguity. The high-AUC/low-F1 divergence is named and attributed to threshold mis-calibration. All of this is well above the median for medical-VLM submissions.

**Issues:**

- **(MAJOR) Single template, no prompt ensembling — and this is not just a limitation, it likely *causes* the headline anomaly.** The paper reports spondylolisthesis AUC 0.807 / F1 0.03 (line 268) and attributes it to "the fixed cosine threshold mis-calibrates." That diagnosis is correct but incomplete from a CLIP-methodology standpoint. In standard CLIP zero-shot practice (Radford et al. 2021), the decision is **argmax over class-prompt cosine similarities**, not a fixed threshold on a single positive prompt. For binary SPIDER labels phrased as "lumbar disc herniation" vs "no disc herniation," the two text anchors are near-collinear in embedding space (negation is famously poorly encoded by CLIP text encoders), so the argmax/softmax decision boundary is essentially arbitrary — which is *exactly* the AUC≫F1 signature. The paper should (a) state explicitly whether the binary decision is argmax-over-two-prompts or a threshold on one prompt, and (b) acknowledge that the known CLIP negation weakness, not just generic "mis-calibration," is the likely driver. This matters because it bears on the headline disc-morphology win too.

- **(MAJOR/borderline-CRITICAL) The single-template + no-ensembling choice undercuts the fairness of the OTS-vs-Hybrid zero-shot comparison.** Prompt-ensembling (the standard CLIP "80-prompt" trick) typically moves zero-shot F1 by several points and can change *which* method wins. The paper draws a substantive conclusion — "off-the-shelf BiomedCLIP edges the hybrid (0.394 vs 0.362)" and "the hybrid wins only on disc-morphology" — from a comparison that both methods run under a single un-ensembled, un-tuned template. The relative ordering could plausibly flip under standard prompt-ensembling. The paper is honest about *not* tuning prompts (good for no-leakage), but it should explicitly flag that the OTS-vs-Hybrid ordering is template-dependent and may not survive prompt-ensembling, rather than presenting 0.394-vs-0.362 as a stable finding. As stated, a strict VLM reviewer could read this as an under-powered comparison dressed as a clean result.

- **(MINOR) Temperature s clamped at 100 affects only softmax sharpness, not argmax ranking.** The paper correctly uses the pretrained logit scale, but it is worth one clarifying sentence that s does not affect AUC (rank-invariant) and only sharpens the softmax — so it is irrelevant to the threshold mis-calibration story and the high-AUC/low-F1 gap is purely a decision-boundary issue, not a temperature issue. Otherwise a reader may wonder if re-temperaturing would fix spondylolisthesis (it would not).

- **(MINOR) Per-class operating-threshold disclosure.** For the binary zero-shot labels, the F1 in Table 2 depends entirely on the (undisclosed) decision rule. State it once. If thresholds were left at the cosine argmax default, say so; if Youden/0.5-on-softmax, say so. Without this the F1 column is not reproducible.

---

## 3. The 2.5D per-slice BiomedCLIP + attention-pool design

**Reasonable and honestly bounded.** Encoding 9 sagittal slices independently with frozen ViT-B/16 and aggregating via a single learned attention vector w_a ∈ R^512 (Eq. 4) is the obvious, defensible way to feed a 2D foundation model volumetric input, and it is the *only* trainable parameter in that branch (good for the frozen-branch story).

- **(MINOR) The attention pool is extremely low-capacity (one 512-vector) and its benefit is asserted, not measured.** Line 142 claims attention pooling "lets the branch up-weight the slices in which a lesion is most clearly visible," but there is no ablation vs mean-pooling on this branch. Given the regularizer thesis depends on what the frozen branch contributes, a one-line mean-pool-vs-attention-pool number would strengthen it. At minimum, present the claim as a design rationale, not a demonstrated effect.

- **(MINOR) "2.5D" is the right honest label and is used.** Good — the Limitations correctly note the branch is slice-independent before pooling (line 318) and that native-3D MRI-VLMs were unavailable. RadCLIP is cited as the closest alternative (line 90) and as future work. This is well-handled.

- **(MINOR) Distribution-shift of the input to BiomedCLIP is unaddressed.** BiomedCLIP was trained on PMC figure crops (often annotated, multi-panel, captioned radiographs), not on single grayscale-replicated-to-RGB sagittal MRI crops resized to 224. Replicating a 1-channel slice to 3 channels and resizing is standard but is a real domain gap for the *image* encoder, and it may partly explain weak far-label transfer. Worth one sentence.

---

## 4. "Label space as input" framing vs standard CLIP zero-shot — and the missing CLIP citation

- **(CRITICAL, but trivially fixable) The original CLIP paper (Radford et al., 2021) is never cited.** The references contain `zhang2023biomedclip`, `lu2024radclip` (RadCLIP), and `koleilat2025biomedcoop`, but **no Radford et al.** entry, and the text says "CLIP-style" (line 90) and "RadCLIP extends CLIP" without ever citing CLIP itself. For a paper whose title and contributions center on CLIP-style zero-shot label-space extension, omitting the foundational citation is a genuine scholarship defect that a VLM reviewer will flag immediately. **Add Radford et al. 2021** (and ideally the zero-shot transfer framing it introduced). This is a 5-minute fix but it is currently a real hole.

- **(MAJOR) "Label space as input" is presented as more novel than it is, and is not connected to its CLIP lineage.** The formulation in Section 3.1 (line 116) — "the label space becomes an *input* to inference rather than a fixed architectural constant" — is precisely the defining property of CLIP zero-shot classification: the classifier weights are synthesized from text prompts at inference, so the label set is an inference-time argument. This is genuinely the standard CLIP zero-shot mechanism, framed in fresh notation. The paper should explicitly say "this is the standard CLIP zero-shot construction (Radford et al.), here instantiated cross-dataset RSNA→SPIDER" so it does not read as if the open-vocabulary property is being newly proposed. The *novel* part is the cross-dataset RSNA→SPIDER label-schema transfer and the fused-encoder-into-BiomedCLIP-text-space alignment — that is a legitimate, modest novelty and should be stated as such, distinguished from the (non-novel) open-vocabulary mechanism.

- **(MINOR) One subtlety worth a sentence:** the fused embedding f_img comes from a *concat-MLP over CBAM-3D + BiomedCLIP-image* features and is then compared against BiomedCLIP *text* anchors. There is no contrastive training that aligns f_img to the text space — alignment is inherited only via the BiomedCLIP-image sub-component surviving the MLP. It is not obvious that a trained fusion MLP (optimized for RSNA cross-entropy, never for text alignment) preserves the image-text metric geometry that makes the cosine-to-text head valid. The fact that zero-shot still works on disc-morphology labels is mildly surprising and worth a sentence of mechanistic acknowledgment (the MLP could in principle have destroyed text-space alignment). This also partially explains why OTS BiomedCLIP — whose image embeddings are *natively* text-aligned — beats the hybrid on far labels.

---

## 5. Is "regularizer" a recognized frozen-feature effect, appropriately hedged?

Covered partly in §1. Short answer: **yes, the effect (frozen pretrained features generalizing better OOD than fully-tuned ones) is a recognized phenomenon, and the paper hedges the n=3 statistics appropriately** ("descriptive effect sizes, not significance tests," line 295; "with three seeds we read this as descriptive evidence," line 70). The hedging on statistics is exemplary for a venue at this tier. The remaining gap is (a) not citing the lineage of that phenomenon, and (b) the unverified causal attribution to BiomedCLIP specifically vs any frozen branch (the §1 MAJOR point).

---

## Strengths summary (for balance)

1. **Honest, falsifiable framing.** Conceding OTS > hybrid on aggregate zero-shot, parity (not superiority) vs SpineNetV2 on transfer, and the trivial-ensemble control all signal scientific integrity rare at this level.
2. **Statistical discipline.** 3 seeds, pooled σ, explicit 2σ bolding rule, refusal to claim significance — correctly calibrated.
3. **Temperature/prompt disclosure** now concrete and a priori.
4. **Clear problem-formulation unification** (Eq. 1) of supervised and zero-shot via a shared encoder Φ — clean and pedagogically strong.
5. **Threshold-vs-ranking diagnosis** (AUC≫F1) shows the authors understand imbalanced-eval subtleties.

## Required / recommended changes

- **CRITICAL:** Cite Radford et al. 2021 (CLIP). Connect "CLIP-style" and "label-space-as-input" to it.
- **MAJOR:** Disclose the binary zero-shot decision rule (argmax-over-2-prompts vs single-prompt threshold); name CLIP negation weakness as a likely driver of the spondylolisthesis AUC≫F1 gap.
- **MAJOR:** Flag that the OTS-vs-Hybrid zero-shot ordering (0.394 vs 0.362) is single-template and may not survive prompt-ensembling.
- **MAJOR:** Either add a non-BiomedCLIP frozen-branch control, or explicitly downgrade "BiomedCLIP acts as the regularizer" to "a frozen general-purpose branch acts as a regularizer (here BiomedCLIP)" and list the unverified mechanism in Limitations.
- **MAJOR:** State that "label space as input" is the standard CLIP zero-shot mechanism; locate the genuine novelty in the cross-dataset schema transfer.
- **MINOR:** Add attention-pool vs mean-pool note/ablation; note s only sharpens softmax (rank-invariant); note BiomedCLIP input domain-shift; note the fusion-MLP-vs-text-alignment subtlety; cite the frozen-feature-OOD lineage for "regularizer."

None of these threaten the core results; all are addressable in a revision without new large experiments (the non-BiomedCLIP control being the only optional compute ask).
