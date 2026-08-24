# Explainability as evidence, not illustration

Design document. 2026-08-24.

## 1. What this is for

The advisor asked for explainability work and, when pressed, defined the goal
precisely:

> "heat map chỉ là 1 method cho giải thích kết quả thôi"
> "Em nên survey nhiều hướng tiếp cận khác nhau và nhiều method khác nhau luôn"
> "**Mục tiêu là tạo ra evidence hỗ trợ cho kết quả đầu ra của mô hình**"

The last clause is the design constraint. A saliency map is an *illustration*:
the reader looks at it and decides for themselves whether it seems reasonable.
It cannot be wrong, so it cannot be evidence. Everything in this design is
built so that a claim can fail a measurement.

This also fixes a gap the project already admits in print. The MIWAI
camera-ready says of its existing Grad-CAM figure:

> "a saliency map only shows where the evidence sits and **is not by itself
> proof of a mechanism**"

## 2. What already exists, and why it is not enough

`viz/grad_cam_rsna.py` produces one figure: Grad-CAM, CBAM versus no-CBAM, on a
spinal-canal Severe case. `viz/confusion_matrix_rsna.py` and
`viz/pr_curves_rsna.py` also exist.

That work answers a Phase 2 question ("does CBAM make the model look in a
better place?") on the condition where the model is already strong (canal
AUPRC 0.63). The Phase 3 claim is a different one, on the condition where the
model is weak (foraminal AUPRC 0.47), and it has no visual or quantitative
support yet.

## 3. The trap this design is built to avoid

The obvious experiment is: *measure how far each model's saliency peak lands
from the radiologist's annotated point, and show the T1 model beats the T2
model.*

That experiment is invalid here. `prep_t1_crops.py:230-231` centres the crop on
the coordinate itself:

```python
x1 = max(0, x_int - crop_w // 2)
y1 = max(0, y_int - crop_h // 2)
```

For a T1 crop, the foraminal annotation **is** the centre of the image. A model
that does nothing but look at the middle of the frame would score well. The T2
crop is centred on the canal coordinate instead, so its foraminal point is
off-centre by construction. Any T1-versus-T2 distance comparison therefore
measures crop geometry, not model behaviour.

The design splits the claim into two questions that are each free of this
confound.

**Question 1 (no model involved).** How often is the foraminal finding
physically absent from the T2 crop?

CORRECTED 2026-08-24. The first version of this measurement was wrong and its
numbers (23.5% / 23.3%) must not be used. It subtracted a foraminal annotation's
pixel (x, y) from a canal annotation's pixel (x, y), but foraminal findings are
annotated on Sagittal T1 and canal findings on Sagittal T2, and those are
different series. Three of four sampled studies had mismatched grids:
384x384 at 0.78 mm against 640x640 at 0.47 mm, and 384x384 at 0.78 mm against
320x320 at 0.94 mm. A pixel offset across two such grids is not a physical
distance.

`geometry_evidence_mm.py` redoes it in the DICOM patient coordinate system,
using ImagePositionPatient and ImageOrientationPatient, which all 40 sampled
studies carry. Measured over 796 pairs in 80 studies:

| | left foraminal | right foraminal |
|---|---|---|
| in-plane displacement from canal, median | 9.7 mm | 9.9 mm |
| in-plane, 90th percentile | 15.2 mm | 15.0 mm |
| through-plane (slice axis), median | 16.0 mm = 4.0 slices | 17.6 mm = 4.0 slices |
| through-plane, 90th percentile | 5.0 slices | 5.0 slices |
| outside the in-plane window (70 x 35 mm half-extent) | 0.3% | 0.3% |
| **outside the 9-slice window (+/-4 slices)** | **32.2%** | **47.2%** |

The claim survives the correction but changes axis, and is sharper for it. The
in-plane window is generous, 70 x 35 mm half-extent against a 10 mm
displacement, so the foraminal region is essentially always present in plane.
The binding constraint is the **9-slice depth**: a third of left and nearly half
of right foraminal findings sit beyond the +/-4 slices the crop spans. The
model was asked to grade a structure outside its field of view along the
slice axis.

This is geometry, not inference: no model, no saliency method, no threshold, and
it cannot be dismissed with "the network might have learned a proxy". It also
points directly at the fix that was already built, per-side T1 crops centred on
each side's own annotated slice.

Worth investigating separately: the left/right asymmetry (32.2% against 47.2%)
is large and is not explained by anything in this measurement.

**Question 2 (model behaviour, confound removed).** Restricted to cases where
the foraminal point **is** inside the crop, in plane and in slices: when the T2 model
grades the foraminal label, does its evidence sit at that point?

This asks about relative position *within one image*, so it is unaffected by
where the crop was centred. If the answer is no, then even where the model
*could* look in the right place, it does not.

Together the two questions close the argument.

## 4. Which model gets explained

Both, and the difference between them is itself a result.

**CBAM branch** (`spinenet/models/grading_attention.py`): the head is
`layer4 → CBAM4 → AdaptiveAvgPool3d → flatten → Linear(512, 3)`. This is
exactly the "CAM architecture" of Draelos & Carin (arXiv:2011.08891), who prove
in their Sec. 3.6.3 that CAM, HiResCAM and Grad-CAM at the last convolutional
layer produce **identical** maps for such a head.

Two consequences. Computing HiResCAM as a separate visualisation would show
nothing new. But the thesis can state, with a citation and a proof rather than
a convention, that Grad-CAM on this branch is not a heuristic, it is an exact
decomposition of that layer's linear contribution to the class score. That
costs nothing and is worth a paragraph.

Boundary condition to state: the equivalence is proven for GAP followed by a
single FC. It would not survive a head with a hidden layer
(GAP → FC → ReLU → FC).

**Hybrid model** (`spinenet/models/grading_hybrid.py`): the score is a
temperature-scaled cosine similarity between an L2-normalised fused embedding
and a **frozen text-prompt embedding**. Grad-CAM's derivation assumes a linear
layer over pooled features. That assumption is violated three ways here: there
is no learned per-class weight vector (the prototypes are text embeddings), the
embedding is L2-normalised (cosine, not linear projection), and the embedding
fuses two branches.

So: the same method is provably faithful on one model in this project and
unjustified on the other. That contrast is a genuine finding for the chapter,
and it is the reason the two models need different treatment rather than one
method applied everywhere.

For the Hybrid, prefer perturbation-based attribution, which only requires that
the score function be evaluable and makes no linearity assumption.

## 5. The five layers

### Layer 0, Geometric evidence
Already computed (Section 3). Re-run with border clipping applied. Output: one
table, one histogram of offset distance with the crop window drawn on it.

**Proves:** the T2 crop cannot contain the graded region in ~23% of cases.
**Does not prove:** anything about what the model does with the other 77%.

### Layer 1, Method selection and a sanity-check gate
Three saliency methods, chosen for distinct mechanisms rather than variety:

| method | branch | why this one |
|---|---|---|
| Grad-CAM (Selvaraju et al. 2017) | CBAM | baseline, and provably faithful for this head (Sec. 4) |
| Layer-CAM (Jiang et al. 2021, IEEE TIP) | CBAM, shallow layer | the resolution answer, with evidence |
| Occlusion sensitivity (Zeiler & Fergus 2014), coarse grid | both | model-agnostic; measures actual score change rather than a gradient proxy |

Layer-CAM earns its place with numbers rather than novelty. On the VGG-16
WSOL benchmark (Jiang et al., Table I), moving plain Grad-CAM off the final
stage collapses localisation accuracy from 43.62% to 8.87% at stage 3, because
Grad-CAM averages the gradient over space into one scalar per channel.
Layer-CAM, which weights per spatial location, holds 46.62% to 41.83% across
the same range. Our `layer4` is only 7x14 for a structure as small as the
foramen, so this is directly on point.

Occlusion is the methodologically important inclusion: Zhang et al. 2024
(*Radiology: AI*) show gradient-based maps can stay visually stable while the
model's AUC drops 8.6% under perturbation, so a non-gradient cross-check is
needed rather than optional.

**Excluded, with reasons:** Grad-CAM++ and XGrad-CAM (same mechanism family as
Grad-CAM, differing only in how the channel weight is computed; Grad-CAM++'s
multi-instance benefit does not apply to a single-disc crop). Score-CAM and
Ablation-CAM (cost scales with the 512 channels of layer4, and no 3D medical
validation was found for either). RISE (4,000-8,000 forward passes per
explanation). LIME and KernelSHAP (need bespoke 3D supervoxel machinery;
Brioso et al. 2026 built special methods precisely because raw KernelSHAP is
impractical on volumes). Eigen-CAM was considered and dropped: it is
class-agnostic by construction, which is a poor fit for a 3-class ordinal task
where the question is specifically what drove *Severe*.

Integrated Gradients (Sundararajan et al. 2017) was the hardest exclusion and
is the first thing to add if time allows. It is mechanistically distinct from
every CAM variant, attributes in input space so it never inherits layer4's 7x14
bottleneck, and passed Adebayo's sanity checks. It is excluded on cost alone:
at 50 integration steps it is roughly 50x a Grad-CAM map, so 500 cases would be
about 6.6 hours against 8 minutes. If included, run it on a stratified subsample
of ~100 cases (~1.3 h) and report it as a subsample rather than silently mixing
sample sizes across methods in one table.

**Gate:** the model-parameter randomisation test (Adebayo et al. 2018, NeurIPS).
Cascade-randomise weights from the output backwards and measure rank
correlation between the original and post-randomisation maps. If a map does not
change when the model is destroyed, it is an edge detector and every later
number computed from it is void, so this runs first and its result decides
whether the rest is reported at all.

Precision required when citing this: the methods that *failed* Adebayo's checks
were Guided Backprop and Guided Grad-CAM. Plain Grad-CAM and Integrated
Gradients passed. Never write "Grad-CAM fails sanity checks". Also cite the
revisit (Yona & Greenfeld 2021, arXiv:2110.14297), which argues the original
conclusions are confounded by task choice.

### Layer 2, Localisation audit, with a chance baseline
On the in-window subset from Layer 0, compute for each case the distance from
the saliency peak to the annotated point, converted to millimetres using the
DICOM pixel spacing. Report Hit@τ across several τ rather than a single
threshold, following the TorchRay `PointingGame` convention, alongside the
continuous distance distribution, the pattern Saporta et al. 2022 (*Nature
Machine Intelligence*, CheXlocalize) use.

**Mandatory baselines**, without which the numbers mean nothing:
- a uniformly random peak inside the crop
- a fixed peak at the crop centre

The second matters most. If saliency does not beat "always guess the middle",
the method has added no information, and given how these crops are built that
is a live possibility rather than a formality.

**Proves:** spatial agreement between model evidence and radiologist
annotation, above chance.
**Does not prove:** causation, the model could coincide with the right region
for anatomical reasons without using it. Layer 3 is what addresses that.

### Layer 3, Exact branch attribution
The Hybrid fuses exactly two branches, so the Shapley value of each branch is
**exact with two extra forward passes**, not approximated. With branches A and
B, the marginal contributions are computed over the only two orderings, using
the existing `--ablate-branch` machinery (`cbam_only`, `biomedclip_only`) to
realise the "branch absent" condition.

This answers a question the thesis currently cannot: does the CBAM branch or
the BiomedCLIP branch drive the decision, and does that split differ between
canal and foraminal? If the foraminal decision leans on the CLIP branch, which
sees only 3 sagittal slices through a generic biomedical encoder, that is a
mechanistic explanation for why foraminal performance is poor, independent of
any saliency map.

There is medical precedent for the occlusion-based version of this analysis
(Gapp et al. 2025, *Int J CARS*), including a named failure mode, "unimodal
collapse", worth checking for.

**Caveat:** zeroing a branch is an out-of-distribution input for the fusion
head. Report the value with that stated, and cross-check against the model's
own `--modality-dropout` behaviour, which trained the model to tolerate exactly
this condition.

### Layer 4, Exhaustive faithfulness audit of the slice-attention pool
The model produces attention weights over its 9 slices. Whether attention
weights are a valid explanation is unsettled, Jain & Wallace 2019 (NAACL)
against Wiegreffe & Pinter 2019 (EMNLP), and it stays unsettled largely
because in NLP the sequences are too long to test every subset.

Here there are 9 slices. **All 2^9 = 512 subsets are enumerable.** For a sample
of cases, evaluate the model on every subset of retained slices and measure
whether the attention weights predict the actual effect of removing a slice. If
they do, this project can state that its attention weights are faithful for
this layer, by exhaustive verification rather than by argument.

Keep the claim scoped: CBAM is feature recalibration inside the backbone, not
an attribution over the decision. Do not present CBAM maps and slice-attention
weights as the same kind of object.

**Cost:** 512 forwards x 0.94 s ≈ 8 minutes per case, so a sample of ~15 cases
stratified over conditions and severities, ≈ 2 hours.

## 6. Measured compute budget

Measured on this machine, not estimated: forward 940 ms, forward+backward
957 ms per crop. Backward is nearly free because the cost is dominated by the
BiomedCLIP ViT-B/16 forward pass, while gradients only flow back through the
CBAM branch.

| layer | work | time |
|---|---|---|
| 0 | geometry | done |
| 1 | sanity gate | ~30 min |
| 2 | localisation: Grad-CAM and Layer-CAM on ~500 cases | ~8 min each |
| 2 | localisation: coarse occlusion on ~100 cases | ~2.6 h at 98 grid positions |
| 3 | branch Shapley, 2 extra forwards per case | ~20 min for 500 cases |
| 4 | exhaustive slice subsets, 15 cases | ~2 h |

**No GPU and no training.** Everything is post-hoc on frozen checkpoints
already on local disk. Note that MPS cannot be used: `adaptive_max_pool3d` is
unimplemented on Metal and CBAM depends on it, so all figures above are CPU.

## 7. Limitations the chapter must state

Stating these makes the work stronger; a reader who finds them unaided
concludes the author did not know.

**The field is actively sceptical of this entire approach.** Arun et al. 2021
(*Radiology: AI*) tested 8 saliency methods on real medical images and every one
failed at least one of four trustworthiness criteria, with localisation AUPRC
(0.160-0.519) far below a purpose-built detection network (0.596, P<.005).
Venkatesh et al. 2024 reach the same conclusion on musculoskeletal radiographs,
the closest domain analogue. Frame the output as exploratory evidence, never as
clinically validated localisation.

**Post-hoc explanation may not reflect the model's actual logic.** Rudin 2019
(*Nature Machine Intelligence*) argues for inherently interpretable models in
high-stakes settings instead. Ghassemi, Oakden-Rayner & Beam 2021 (*Lancet
Digital Health*) call current XAI "false hope" at the individual-patient level.
Note that a MICCAI 2024 paper (Wåhlstrand Skärström et al.) rejects saliency
maps for vertebral pathology in favour of a differentiable rule-based
classifier, the field is moving away from post-hoc CAM in this exact anatomy,
which is a sharper motivation than a generic literature review.

**Saliency maps are fragile.** Ghorbani, Abid & Zou 2019 (AAAI) change a map
completely with imperceptible perturbation while prediction and image are
unchanged. A small experiment adding noise and re-rendering would support this
paragraph far better than a bare citation.

**Deletion/insertion has an out-of-distribution confound.** Masking creates
inputs unlike anything in training, so the model may react to the hole rather
than to lost evidence (Rong et al. 2022, ICML, ROAD; Hase et al. 2021,
NeurIPS). On MRI a zeroed voxel is itself abnormal. If deletion curves are
computed at all, use noisy linear imputation rather than zeroing, and state
that residual confound remains.

**No mandatory reporting standard exists.** CLAIM (Mongan, Moy & Kahn 2020,
*Radiology: AI*) item 31 asks only that explanation methods and their
validation be described, without thresholds. So choose τ and the baselines
explicitly and defend them, rather than leaving them implicit.

## 8. Deliverables

Code, under `experiments/explainability/`:
- `geometry_evidence.py`, Layer 0, with border clipping
- `saliency_methods.py`, Grad-CAM, Layer-CAM, coarse occlusion, for both model families
- `sanity_checks.py`, Layer 1 gate
- `localization_audit.py`, Layer 2, including both baselines
- `branch_shapley.py`, Layer 3
- `slice_attention_audit.py`, Layer 4

Outputs, under `experiments/explainability/results/`: one JSON per layer plus
the figures. Figures follow the advisor's writing rules already recorded in
`CLAUDE.md` (rules 6, 7, 8, 10), no title baked into the image that duplicates
the LaTeX caption, `width=\textwidth` rather than a small `scale=`, a unique
`\label` per float, and every float referenced and described in the body text.

(Not to be confused with CLAIM, the medical reporting checklist cited in
Section 7. Different documents, unfortunately similar names.)

## 9. Out of scope

Axial imaging and the subarticular labels. Any retraining. Inherently
interpretable model architectures (Rudin's recommendation), worth citing as
the acknowledged alternative, not worth building. Text-side prompt attribution:
interesting, but BiomedCLIP was pretrained on general biomedical text rather
than spine reports, so it audits prompt design rather than clinical reasoning,
which does not serve the routing claim.

## 10. Citations to verify before the final submission

The research pass flagged these as not fully verified from primary sources:

- TRIPOD+AI (Collins et al. 2024, *BMJ*), citation confirmed, the specific
  explainability item was not readable. Check tripod-statement.org directly.
- Zeiler & Fergus 2014, cite the method, not a specific patch size or stride,
  unless Sec. 4.2 is checked directly.
- Integrated Gradients "20-300 steps" and RISE "4000-8000 masks", consistent
  across secondary sources, not quoted from the primary PDFs.
- A 2026 Research-Square lumbar-spine Grad-CAM paper on the same dataset family
 , fetch was blocked (403). Verify its numbers before citing them.
- No 3D medical validation was found for Layer-CAM, Score-CAM, Ablation-CAM or
  XGrad-CAM. Report as "no evidence found", never as "does not exist", the
  gap between 2D natural-image CAM research and 3D medical CAM research is
  itself a citable observation.
