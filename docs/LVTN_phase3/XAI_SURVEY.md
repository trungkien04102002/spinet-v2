# Survey of explainability approaches for the RSNA lumbar grading models

Written 2026-08-24, in response to the advisor's instruction:

> "heat map chỉ là 1 method cho giải thích kết quả thôi"
> "Em nên survey nhiều hướng tiếp cận khác nhau và nhiều method khác nhau luôn
> (cũng là dùng heatmap nhưng technique khác)"
> "Mục tiêu là tạo ra evidence hỗ trợ cho kết quả đầu ra của mô hình"

This is the survey. It is deliberately broader than what will be implemented:
the narrowing decision and its justification live in
`docs/superpowers/specs/2026-08-24-explainability-evidence-design.md`, and the
architecture-specific analysis lives in
`docs/explainability_cosine_vlm_research.md`. A thesis chapter needs both the
breadth (what exists, what was considered, what was rejected and why) and the
depth (what was actually run).

Citations give author, year and venue so they can be dropped into a `.bib`.
Section 8 lists what could not be verified from a primary source and must be
checked before submission.

---

## 1. The distinction that organises this survey

"Explainability" covers two things that are usually conflated, and the advisor's
word *evidence* picks out the second one.

An **illustration** is shown to a reader who then decides for themselves whether
it looks reasonable. It cannot be wrong, so it cannot be evidence. The project's
current Grad-CAM figure is of this kind, and the MIWAI camera-ready says so in
print: *"a saliency map only shows where the evidence sits and is not by itself
proof of a mechanism."*

**Evidence** is a measurement that can come out against the claim. Producing it
needs two things a saliency map alone does not have: a quantity, and a baseline
that the quantity could fail to beat.

So the survey is organised in two tiers. Tier 1 is the methods that produce an
explanation (Sections 2 to 5). Tier 2 is the protocols that turn an explanation
into a measurement (Section 6). Most XAI papers, and most theses, stop at Tier 1.

---

## 2. Family A: gradient-based CAM

All members compute a weighted combination of a convolutional layer's feature
maps, differing only in how the weights are obtained.

| method | venue | weighting mechanism | cost | what it fixes |
|---|---|---|---|---|
| Grad-CAM | Selvaraju et al. 2017, ICCV | channel weight = spatially averaged gradient | 1 fwd + 1 bwd | removes plain CAM's GAP+single-FC architectural constraint |
| Grad-CAM++ | Chattopadhay et al. 2018, WACV | second-order weighting | 1 fwd + 1 bwd | multiple instances of one class in an image |
| XGrad-CAM | Fu et al. 2020, BMVC | axiom-constrained weights | 1 fwd + 1 bwd | satisfies Conservation and Sensitivity axioms |
| Layer-CAM | Jiang et al. 2021, IEEE TIP | per-location gradient, not pooled | 1 fwd + 1 bwd | Grad-CAM collapses at shallow layers |
| HiResCAM | Draelos and Carin 2020, arXiv:2011.08891 | element-wise grad x activation, no spatial averaging | 1 fwd + 1 bwd | Grad-CAM can highlight locations the model did not use |

**Layer-CAM is the one with numbers behind it.** On the VGG-16 WSOL benchmark
(Jiang et al., Table I), localisation accuracy by stage depth, S5 being the final
conv stage:

| method | S5 | S4 | S3 | S2 | S1 |
|---|---|---|---|---|---|
| Grad-CAM | 43.62% | 18.32% | 8.87% | 19.59% | 13.95% |
| Layer-CAM | 46.62% | 44.05% | 41.83% | 43.18% | 43.71% |

This matters directly here. Our `layer4` is only 7x14 after two strides, while
the neural foramen is a small structure, so the obvious fix is to attach the
explanation to a shallower layer. The table says plain Grad-CAM cannot be moved
there: it collapses from 43.62% to 8.87%, because averaging the gradient over
space into one scalar per channel discards too much when the per-location
variance is high. Layer-CAM holds up by construction.

A related caution from the same paper (Table II): naive multi-layer *fusion* is
method-dependent, not a free win. Fusing shallow Grad-CAM maps into the final map
*hurts* (43.62% to 33.96% as stages are added); fusing shallow Layer-CAM maps
*helps* (46.62% to 47.24%).

**HiResCAM has a special status for our CBAM model, and it is a proof rather than
a method.** The CBAM head is
`layer4 -> CBAM4 -> AdaptiveAvgPool3d -> flatten -> Linear(512, 3)`, verified in
`spinenet/models/grading_attention.py`. That is exactly the "CAM architecture"
Draelos and Carin define, and their Sec. 3.6.3 states that for it, CAM, HiResCAM
and Grad-CAM at the last convolutional layer produce **identical** maps.

Two consequences. Rendering HiResCAM separately would show nothing new. But the
thesis can state, with a citation and a proof rather than a convention, that
Grad-CAM here is not a heuristic: it is an exact decomposition of that layer's
linear contribution to the class score. Boundary condition to state: the
equivalence holds for GAP followed by a single FC, and would not survive a head
with a hidden layer.

---

## 3. Family B: gradient-free CAM

Same output form, no backward pass, so immune to gradient saturation and to
gradient noise.

| method | venue | mechanism | cost |
|---|---|---|---|
| Score-CAM | Wang et al. 2020, CVPR-W | each channel's own activation used as a mask, weight = resulting score | C+1 forwards |
| Ablation-CAM | Desai and Ramaswamy 2020, WACV | weight = score drop when channel is ablated | k+1 forwards |
| Eigen-CAM | Muhammad and Yeasin 2021 | principal component of the activations | 1 forward + PCA |

Cost is the deciding factor: `layer4` has 512 channels, so Score-CAM and
Ablation-CAM need on the order of 512 full 3D forward passes per explanation.
Eigen-CAM is a single pass, but is **class-agnostic by construction**: it shows
where the network's dominant activation lies, not what drove *Severe*. For a
3-class ordinal task where the entire question is the rare class, that is a
substantive limitation rather than a footnote.

---

## 4. Family C: input-space attribution

These do not attach to a conv layer at all, so they never inherit the 7x14
bottleneck. This is a genuinely different *approach*, not another CAM variant,
which is the part of the advisor's instruction that the CAM families do not
satisfy.

| method | venue | mechanism | cost |
|---|---|---|---|
| Occlusion sensitivity | Zeiler and Fergus 2014, ECCV | slide a mask, record the score drop | grid-dependent: tens to thousands |
| Integrated Gradients | Sundararajan et al. 2017, ICML | path integral of gradients from a baseline | 20 to 300 fwd+bwd |
| SHAP / DeepSHAP / KernelSHAP | Lundberg and Lee 2017, NeurIPS | Shapley values over features | DeepSHAP tens; KernelSHAP exponential |
| LIME | Ribeiro et al. 2016, KDD | local surrogate over superpixels | hundreds to 1000 forwards |
| RISE | Petsiuk et al. 2018, BMVC | random masking, score-weighted average | 4000 to 8000 forwards |

**Occlusion is the methodologically important member.** It is the only method in
this survey that measures the actual change in the model's output rather than a
gradient or decomposition proxy. That matters because Zhang et al. 2024
(*Radiology: AI*) show a gradient map staying visually stable while the model's
AUC dropped 8.6% under perturbation. A non-gradient cross-check is therefore
necessary, not decorative. Precedent for the affordable form exists: He et al.
2025 (*Human Brain Mapping*) occlude anatomical regions rather than sweeping a
dense window, about 247 forwards for a whole brain volume.

Integrated Gradients is the strongest exclusion-on-cost: mechanistically
distinct, passes the sanity checks of Section 6, attributes at voxel resolution,
but at 50 integration steps it is roughly 50x a Grad-CAM map.

KernelSHAP is documented as impractical on raw 3D volumes: Brioso et al. 2026
(arXiv:2604.11775) built specialised machinery precisely to work around this.
LIME needs a bespoke 3D supervoxel step with no validated volumetric-medical
precedent found.

---

## 5. Family D: intrinsic explanation

The model produces these itself. No post-hoc method is applied, so there is
nothing to be unfaithful *to* in the usual sense, though see the caveat below.

| source | what it gives | cost |
|---|---|---|
| CBAM channel and spatial attention | per-channel and per-location gating inside the backbone | free, already computed |
| Slice-attention pool | a weight per slice over the 9 input slices | free, already computed |
| Cosine similarity to text anchors | a score per class prototype, directly interpretable | free |
| Branch attribution | how much each of the two frozen branches contributed | 2 extra forwards, exact |

**Whether attention is an explanation is contested, and the answer is
"plausible, not automatically faithful".** The debate ran from Jain and Wallace
2019 (NAACL, "Attention is not Explanation") against Wiegreffe and Pinter 2019
(EMNLP, "Attention is not not Explanation"). For vision specifically there is a
further critique: Mehrani and Tsotsos 2023 (*Frontiers*) argue ViT self-attention
performs perceptual grouping rather than attention, and Wu et al. 2024 (CVPR,
SaCo) find raw attention often scores below a random baseline on faithfulness
while gradient-weighted multi-layer attention does much better.

A distinction the thesis must keep: **CBAM is feature recalibration inside the
backbone, not an attribution over the decision.** Presenting CBAM maps and
slice-attention weights as the same kind of object would be a methodological
error.

**Two things this architecture makes cheap that are usually intractable.**

*Exact branch attribution.* The Hybrid fuses exactly two branches, so the Shapley
value of each is exact with two extra forward passes rather than approximated.
Almost all multimodal attribution work approximates because it has three or more
modalities. This answers a question the thesis cannot currently answer: does the
CBAM branch or the BiomedCLIP branch drive the decision, and does that differ
between canal and foraminal? If foraminal leans on the CLIP branch, which sees
only a few sagittal slices through a generic biomedical encoder, that is a
mechanistic account of the weak foraminal result independent of any saliency map.
Medical precedent for the occlusion-based version: Gapp et al. 2025
(*Int J CARS*), which also names a failure mode to check for, "unimodal
collapse".

*Exhaustive faithfulness audit of the slice pool.* The attention debate above
stays open partly because NLP sequences are too long to enumerate every subset.
There are 9 slices here, so all 2^9 = 512 subsets can be evaluated. Whether the
attention weights predict the actual effect of removing a slice becomes a
settled question for this layer, by exhaustion rather than by argument.

---

## 6. Tier 2: the protocols that make an explanation into evidence

This is the part that answers "tạo ra evidence". Order matters: the gate comes
first, because if it fails nothing after it means anything.

### 6.1 Gate: sanity checks
Adebayo et al. 2018 (NeurIPS) propose two tests: model-parameter randomisation
(cascade-randomise weights from the output backwards) and data randomisation
(train on shuffled labels). If a map does not change when the model is
destroyed, it is an edge detector.

**Cite this precisely.** The methods that *failed* were Guided Backprop and
Guided Grad-CAM. Plain Grad-CAM, Integrated Gradients and gradient x input
passed. Writing "Grad-CAM fails sanity checks" would be wrong. Also cite the
revisit, Yona and Greenfeld 2021 (arXiv:2110.14297), which argues the original
conclusions are confounded by task choice and that under a causal framing both
families pass.

*Proves:* the map depends on the learned weights.
*Does not prove:* the map is clinically correct. Necessary, not sufficient.

### 6.2 Localisation against radiologist annotations
`train_label_coordinates.csv` gives an (x, y) point per condition per level per
study. Ground truth is therefore a **point**, not a mask, which rules out IoU and
the original Pointing Game formulation directly.

The usable convention is TorchRay's `PointingGame`: a hit when the distance from
the saliency peak to the ground-truth point is within a tolerance, default 15 px.
Report several tolerances rather than one, convert to millimetres using the DICOM
pixel spacing, and report the continuous distance distribution alongside the hit
rate, which is the pattern Saporta et al. 2022 (*Nature Machine Intelligence*,
CheXlocalize) follow with mIoU.

**The confound that makes this measurement worthless without a baseline.**
`prep_t1_crops.py:230-231` centres the crop on the annotated coordinate:

```python
x1 = max(0, x_int - crop_w // 2)
y1 = max(0, y_int - crop_h // 2)
```

For a T1 crop the annotation *is* the image centre. A model that only ever looked
at the middle of the frame would score well, and a T1-versus-T2 comparison would
measure crop geometry rather than model behaviour. Two baselines are therefore
mandatory: a uniformly random peak, and a peak fixed at the crop centre. If
saliency does not beat "always guess the middle", it has added no information.

*Proves:* spatial agreement above chance.
*Does not prove:* causation. The model could coincide with the right region for
anatomical reasons without using it. That is what 6.3 is for.

### 6.3 Faithfulness: deletion and insertion
Petsiuk et al. 2018 (RISE) measure the AUC of the prediction as voxels are
removed or added in saliency order.

Known confound, and it is serious on MRI: masking produces inputs unlike
anything in training, so the model may react to the hole rather than to lost
evidence. A zeroed voxel is itself an abnormality. Rong et al. 2022 (ICML, ROAD)
and Hase et al. 2021 (NeurIPS) formalise this; ROAD replaces zeroing with noisy
linear imputation. ROAR (Hooker et al. 2019, NeurIPS) fixes it more thoroughly
by retraining, at much greater cost.

*Proves:* the output depends on the highlighted region, in a
perturbation-correlational sense.
*Does not prove:* the region is anatomically correct. That is 6.2's job.
*State:* residual out-of-distribution confound remains even with ROAD.

### 6.4 Robustness
Infidelity and Sensitivity-Max (Yeh et al. 2019, NeurIPS), both implemented in
`captum.metrics`. Measures stability of the explanation under small input
perturbation. A technical stress test; adds nothing about clinical localisation.

### 6.5 Tooling
Quantus (Hedström et al. 2023, *JMLR* 24(1), arXiv:2202.06861) implements 35+
metrics across faithfulness, robustness, localisation, complexity, randomisation
and axiomatic categories, and is actively maintained. Captum provides Infidelity
and Sensitivity-Max but no localisation metrics. Preferring a maintained library
over hand-rolled metrics is worth a sentence in the methods section.

---

## 7. The case against this entire enterprise

A survey that omits this is weaker, and a committee member who raises it
unprompted has caught the author out.

**Every method tested on real medical images failed something.** Arun et al.
2021 (*Radiology: AI* 3(6), e200267) evaluated 8 saliency methods on SIIM-ACR
Pneumothorax and RSNA Pneumonia against four criteria (localisation AUPRC,
sensitivity to randomisation, repeatability, reproducibility). All 8 failed at
least one. Saliency localisation AUPRC ranged 0.160 to 0.519 against a
purpose-built detection network at 0.596 (P < .005). Venkatesh et al. 2024
(*J Imaging Inform Med*) reach the same conclusion on musculoskeletal
radiographs, the closest domain analogue to spine imaging.

**Post-hoc explanation may not reflect the model's logic at all.** Rudin 2019
(*Nature Machine Intelligence*) argues for inherently interpretable models in
high-stakes settings instead of explaining black boxes. Ghassemi,
Oakden-Rayner and Beam 2021 (*Lancet Digital Health*) call current XAI "false
hope", with no guarantee of correctness at the individual-patient level.

**The field is moving away from saliency in this exact anatomy.** Wåhlstrand
Skärström et al. 2024 (MICCAI) reject saliency maps for vertebral fracture
analysis in favour of a differentiable rule-based classifier. This is a sharper
motivation to state than a generic literature review, because it is the same
anatomical domain.

**Saliency maps are adversarially fragile.** Ghorbani, Abid and Zou 2019 (AAAI)
change a map completely with imperceptible perturbation while the prediction and
the image are unchanged. A small experiment adding noise and re-rendering would
support this paragraph far better than a bare citation.

**No reporting standard mandates anything specific.** CLAIM (Mongan, Moy and
Kahn 2020, *Radiology: AI*) item 31 asks only that explanation methods and their
validation be described, with no thresholds. A 2026 review states plainly that
no solid validation framework exists for XAI in radiology. So thresholds and
baselines must be chosen explicitly and defended, not left implicit.

**Consequence for how the thesis frames its own results:** exploratory evidence
supporting interpretation, at dataset level, not validated clinical
localisation, and not per-patient guarantees.

---

## 8. Verification debts

Flagged during the research pass as not confirmed from a primary source. Do not
quote specifics until checked.

- TRIPOD+AI (Collins et al. 2024, *BMJ*): citation confirmed, the specific
  explainability item was not readable. Check tripod-statement.org directly.
- Zeiler and Fergus 2014: cite the method, not a specific patch size or stride,
  unless Sec. 4.2 is read directly.
- Integrated Gradients "20 to 300 steps" and RISE "4000 to 8000 masks":
  consistent across secondary sources, not quoted from the primary PDFs.
- Adebayo et al. 2018: corroborated through two independent routes but worth
  opening the NeurIPS PDF once when writing Related Work.
- A 2026 Research-Square lumbar-spine Grad-CAM paper on the same dataset family:
  fetch blocked (403). Verify its numbers before citing.
- "Revisiting the Trustworthiness of Saliency Methods in Radiology AI" and a
  PMC occlusion-saliency lung-screening paper: found via search, author and year
  not fully verified.
- **No 3D volumetric medical validation was found for Layer-CAM, Score-CAM,
  Ablation-CAM or XGrad-CAM.** Report this as "no evidence found", never as
  "does not exist". The gap between 2D natural-image CAM research and 3D medical
  CAM research is itself a citable observation about the maturity of the field.

---

## 9. What this survey concludes

Sixteen methods across four families were considered. Five would be implemented,
chosen for **distinct mechanisms rather than variety**, because five variants of
channel-weighted gradient CAM is not a survey of approaches, it is a survey of
one approach.

| chosen | family | reason |
|---|---|---|
| Grad-CAM | A | baseline; provably identical to HiResCAM for this head |
| Layer-CAM | A, shallow layer | the only CAM variant with quantitative evidence that shallow attachment works |
| Occlusion sensitivity | C | model-agnostic; measures real score change, not a proxy |
| Branch Shapley | D | exact here, approximate everywhere else |
| Slice-attention audit | D | exhaustive, settles a debate for this layer |

Excluded and why: Grad-CAM++ and XGrad-CAM (same mechanism as Grad-CAM;
Grad-CAM++'s multi-instance benefit does not apply to a single-disc crop).
Score-CAM and Ablation-CAM (roughly 512 forwards each at layer4; no 3D medical
validation found). Eigen-CAM (class-agnostic, wrong fit for an ordinal rare-class
question). Integrated Gradients (about 50x Grad-CAM's cost; first thing to add if
time allows, and if added, report it on a stated subsample rather than mixing
sample sizes in one table). RISE (4000 to 8000 forwards per explanation).
KernelSHAP and LIME (need bespoke 3D machinery with no validated precedent).

The evaluation protocol carries the *evidence* claim, not the method count: the
sanity-check gate first, then point-localisation with mandatory chance baselines,
then perturbation faithfulness with ROAD-style imputation.

**A finding worth stating in its own right:** the same method is provably
faithful on one model in this project and unjustified on the other. Grad-CAM is
exact for the CBAM head, which is GAP followed by a single linear layer. The
Hybrid scores by cosine similarity against frozen text prototypes, which
violates Grad-CAM's linear-head assumption three ways at once: no learned
per-class weight vector, an L2-normalised embedding, and a fusion of two
branches. Details and the adapted methods are in
`docs/explainability_cosine_vlm_research.md`.
