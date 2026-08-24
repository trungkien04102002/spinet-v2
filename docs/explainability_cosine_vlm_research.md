# Explainability for a Cosine-Similarity Dual-Branch VLM (BiomedCLIP + 3D-CNN/CBAM)

Research notes for the thesis chapter on explainability. Target architecture: frozen 3D ResNet34+CBAM
branch, frozen BiomedCLIP ViT-B/16 branch (applied per-slice), trainable slice-attention pooling over 9
slices, branch fusion, L2-normalize to 512-d, classify by scaled cosine similarity against frozen
text-prompt embeddings (no learned per-class weight vector). Compiled 2026-08-24.

---

## 0. Why the standard toolbox doesn't drop in cleanly

Almost every mainstream CAM method (Grad-CAM, Grad-CAM++, XGrad-CAM, Layer-CAM) was derived under one
assumption: the class score is `s_c = w_c^T · GAP(A) + b_c`, i.e. a **linear readout with a learned,
per-class weight vector** `w_c` on top of global-average-pooled feature maps `A`. Grad-CAM's channel
weights `α_c^k = GAP(∂s_c/∂A^k)` are only exactly equal to `w_c^k` under that linear-GAP assumption.

This architecture violates that assumption in three independent ways simultaneously:

1. There is no `w_c` — the "classifier" is a **frozen text embedding** `t_c`, external to the vision
   network and not trained jointly with it as a weight matrix.
2. The image embedding is **L2-normalized** before the score is computed, so the score is a similarity
   on the unit hypersphere, not an unbounded linear projection.
3. The image embedding is itself a **fusion of two branches**, one of which (BiomedCLIP) is already an
   internally-attentional transformer, not a plain CNN.

None of this makes gradient-based explanation invalid — it makes the *default, unmodified* CAM formula
mathematically mismatched to what's being explained. Section 1 gives the corrected math; Sections 2–6
answer the six research questions in order; Section 7 gives the concrete valid/invalid guidance and
three architecture-specific angles worth building into the thesis.

---

## 1. CAM adaptation for a cosine-similarity / prototype head (RQ1)

**Is gradient-of-score-wrt-feature-map still meaningful?** Yes, but it means something subtly different
than in the softmax-linear case, and the literature has formalized this rather than leaving it ad hoc.

- **The gradient still exists and is informative**, but because of the L2-normalization, `∂s/∂v` (v =
  pre-normalization embedding) is the *tangential component of the text prototype relative to the
  current embedding direction*, scaled by `1/‖v‖`: moving `v` in that direction increases cosine
  similarity fastest. It is not simply "the prototype vector" the way `w_c` is in a linear head — it's
  `t` projected orthogonal to the current `v` and rescaled. Practically: Grad-CAM-style weighting still
  points at the right thing (channels that would most increase agreement with the prompt), but authors
  should not describe it as identical to reading off a class weight vector.
- **This exact problem has its own literature**, i.e. this is not an improvised adaptation:
  - *GAM: Explainable Visual Similarity and Classification via Gradient Activation Maps*
    (arXiv:2109.00951, 2021) computes gradients of a pairwise similarity score w.r.t. feature maps to
    produce CAM-style explanations for similarity/verification models — directly the "no FC layer,
    score is a similarity" case.
  - **SFAM — Liao, Akpudo, Zhang, Gao, Zhou, Zeng, Zhang, *"Visual Explanation via Similar Feature
    Activation for Metric Learning"*, arXiv:2506.01636 (2025)** — states the problem explicitly ("CAM and
    Grad-CAM rely on a fully-connected classifier layer... incompatible with metric learning models that
    lack one") and proposes a **channel-wise Contribution Importance Score (CIS)** derived from the
    similarity between two embeddings, explicitly supporting both Euclidean- and **cosine-similarity**
    embeddings. This is the closest published precursor to your exact head design and is the citation to
    lean on for "how do you adapt CAM when the logit is a cosine similarity."
  - Earlier work in the same vein: *Visual Explanation for Deep Metric Learning* (arXiv:1909.12977, 2019)
    also targets embedding-space (non-classifier) models.
- **Gradient-free CAMs transfer more cleanly and are arguably the safer default here.** Score-CAM (Wang
  et al., CVPRW 2020) and Ablation-CAM (Desai & Ramaswamy, WACV 2020) don't use backprop at all — they
  perturb the input (mask by upsampled activation maps, or ablate channels) and directly measure the
  **change in the actual target score** (forward pass only). Because they only require that you can
  *evaluate* the score function on perturbed inputs — not that it be linear in the features — they need
  **zero modification** to work when the score is a scaled cosine similarity instead of a softmax logit.
  This is precisely the trick **gScoreCAM** (next section) uses for CLIP.
- **Practical recommendation**: use Score-CAM/Ablation-CAM-style (perturbation, forward-pass-only)
  attribution as the primary/default CAM for the fused embedding and the cosine score, since it sidesteps
  the normalization-Jacobian subtlety entirely; use the gradient-based SFAM/GAM formulation as a faster,
  secondary method, and validate that the two agree (a form of sanity check, see below).
- **Don't skip a faithfulness sanity check.** Standard CAM methods (including Grad-CAM) are known to fail
  basic sanity checks in some settings — Adebayo et al., *"Sanity Checks for Saliency Maps"*, NeurIPS 2018,
  showed several saliency methods are insensitive to model/label randomization. Because this head is
  non-standard, running Adebayo-style parameter-randomization and label-randomization checks on whatever
  CAM variant you adopt is good practice to include and will pre-empt a reviewer/committee objection.

---

## 2. CLIP-specific explainability: what works, what's known to fail (RQ2)

**Established, purpose-built methods for CLIP (cite these three as your core baselines):**

- **gScoreCAM — Chen, Li, Biaz, Bui, Nguyen, *"gScoreCAM: What is CLIP looking at?"*, ACCV 2022.**
  Sub-samples the ~300 highest-gradient channels of CLIP's penultimate layer (using the gradient of the
  cosine-similarity-based score purely as a *channel-selection* heuristic, not as the final importance
  weight), then runs Score-CAM-style forward perturbation on just that subset — 8–10× faster than plain
  Score-CAM while beating HilaCAM, RISE and the classic CAM family, especially on multi-object scenes.
  This is the most direct precedent for "CAM adapted specifically to CLIP's cosine-similarity scoring,"
  and it explicitly demonstrates the gradient-free / perturbation strategy from Section 1 is the one that
  scales and works for CLIP in practice.
- **CLIP Surgery — Li, Wang, Duan, Zhang, Xiaomeng Li, *"CLIP Surgery for Better Explainability with
  Enhancement in Open-Vocabulary Tasks"*, arXiv:2304.05653 (2023), later Pattern Recognition (2025).**
  Directly documents the failure mode this RQ asks about: **raw self-attention in CLIP's ViT links to
  inconsistent semantic regions and produces "opposite" visualizations** (attending to background instead
  of the object), caused by redundant/entangled features shared across unrelated categories. Their fix
  ("surgery") is architectural but training-free: modify the query-key self-attention into a
  value-value consistency path and remove feature redundancy, enabling reliable CAM and even multimodal
  (text-conditioned) visualization without fine-tuning.
- **MaskCLIP — Zhou, Loy, Dai, *"Extract Free Dense Labels from CLIP"*, ECCV 2022 (Oral).** Independently
  arrives at the same conclusion via a different fix: **discard the query/key branch of the final
  attention-pooling layer** and use only the value projection (turned into a 1×1 conv), because raw
  attention pooling destroys spatial locality. Raises CLIP's zero-shot segmentation mIoU on COCO-Stuff
  from 5.7% → 16.7% purely from this attention surgery — independent, convergent evidence that naive
  attention consumption from CLIP is unreliable, and that "skip the query-key softmax, use values
  directly" is a recurring, effective fix pattern.
- **Gandelsman, Efros, Steinhardt, *"Interpreting CLIP's Image Representation via Text-Based
  Decomposition"*, ICLR 2024 (Oral).** Decomposes the final CLIP image embedding as a sum over
  patches/layers/heads, and interprets each summand by finding text descriptions ("TextSpan") that best
  span its contribution. This is complementary rather than a CAM competitor: it gives a *global,
  mechanistic* account of what each attention head encodes (many heads turn out to encode a specific
  property like shape or color/texture) and — importantly for your text-prototype head — shows that
  **CLIP's own image embedding is natively decomposable into text-nameable components**, which is a
  useful conceptual bridge for RQ6.
- **Boosting interpretability via fine-tuning — Gong, Lei, Dou, Farnia, ICLR 2025, "Boosting the visual
  interpretability of CLIP via adversarial fine-tuning."** Shows CLIP's interpretability (measured by
  feature-attribution quality and neuron-concept alignment) can be substantially improved by
  norm-regularized adversarial fine-tuning. **Flag this explicitly as inapplicable under your constraints**
  — both branches are frozen by design in your architecture, so this is background/future-work material,
  not something you can apply as-is.

**What is established to *fail* (state this plainly in the thesis, it is a documented, citable finding,
not a hunch):**
- **Raw attention-weight visualization** (a single layer's softmax attention map) on CLIP ViT is
  unreliable — CLIP Surgery's central empirical finding.
- **Naive attention rollout** (Abnar & Zuidema's original recipe, designed for BERT/NLP classification)
  applied unmodified to CLIP's image tower inherits the same background-bias/inconsistency problem,
  because rollout is a *linear recombination of the same flawed per-layer attention matrices* — if the
  underlying per-layer attention is already misleading for CLIP (as CLIP Surgery shows), simple matrix
  products of it don't fix that; you need either gradient information (gScoreCAM, Chefer et al., Section 4)
  or an architectural fix (CLIP Surgery, MaskCLIP) — not just multi-layer aggregation of raw weights.

---

## 3. Is attention a valid explanation? The NLP debate, and whether vision changes the answer (RQ3)

- **Jain & Wallace, *"Attention is not Explanation"*, NAACL 2019** (recurrent/attention NLP classifiers):
  attention weights are frequently **uncorrelated with gradient-based feature importance**, and one can
  construct **adversarial attention distributions** — very different weight patterns that produce
  near-identical predictions — implying the learned weights are not uniquely tied to the decision and so
  cannot, by themselves, be presented as *the* reason for a prediction.
- **Wiegreffe & Pinter, *"Attention is not not Explanation"*, EMNLP 2019** — rebuttal, not a refutation:
  argues the finding depends on how "explanation" is defined and on the architecture tested (mainly
  simple RNN attention). They propose four diagnostic tests (uniform-weight baseline, variance
  calibration across seeds, a frozen-attention diagnostic, and adversarial-attention training) and show
  that even adversarially-constructed alternative attentions **fail** a simple diagnostic — i.e. Jain &
  Wallace's negative result doesn't generalize to "attention can never be explanatory," but they don't
  claim attention always is one either.
- **Where the field landed (~2020–2024)**: attention weights are, at best, **plausible** (a human-readable
  hint of where the model looked) but not automatically **faithful** (a causally accurate account of what
  drove the output). The correct practice, echoed across the follow-up literature, is to treat raw
  attention as one weak, cheap signal and to validate it against faithfulness metrics (deletion/insertion
  curves, gradient agreement, ablation) before using it as an explanation — never present it unvalidated
  as ground truth.

**Does the conclusion differ for spatial/channel attention in vision vs. NLP attention?** Partially yes,
and this is worth a dedicated paragraph in the thesis:

- **Vision-transformer self-attention has its own, independent critique**, not just an inherited one:
  Mehrani & Tsotsos, *"Self-attention in vision transformers performs perceptual grouping, not
  attention"*, Frontiers in Computer Science, 2023. They show formally and empirically (across ViT, DeiT,
  BEiT, Swin, CvT) that the dot-product compatibility function implements **Gestalt-like similarity
  grouping** — tokens group by feature similarity, not by task-driven salience — and that ViT
  attention performs *poorly* at singleton/salience-detection benchmarks compared even to CNN saliency
  baselines. Their conclusion is stronger than Jain & Wallace's: it isn't just "attention weights don't
  always align with importance," it's "the mechanism computationally is not doing selective attention in
  the first place." This applies directly to the BiomedCLIP ViT branch's internal self-attention.
- **A CVPR 2024 paper gives the most recent, most relevant empirical verdict for ViTs specifically**: Wu,
  Kang, Tang, Hong, Yan, *"On the Faithfulness of Vision Transformer Explanations"*, CVPR 2024. They
  introduce a Salience-guided Faithfulness Coefficient (SaCo) and find that **many attention-based
  explanation methods — including raw attention — fail to reliably beat random attribution** on
  faithfulness grounds; but critically, **gradient-informed and multi-layer-aggregated attention variants
  are markedly more faithful than raw single-layer attention weights.** This is a very close mirror of
  where the NLP debate landed (Wiegreffe & Pinter's more favorable reading applies to the *processed*,
  not raw, forms of attention) — so the honest thesis claim is: *raw attention weights (CBAM maps, raw
  ViT attention, raw slice-attention-pool weights) should not be presented as-is as the explanation; a
  gradient- or perturbation-corrected variant should be.*
- **CBAM specifically is a different animal from token-attention and deserves its own framing.** CBAM
  (Woo et al., ECCV 2018) is a **deterministic, per-instance multiplicative gate** (sigmoid channel and
  spatial masks applied to a feature map inside the CNN), not a softmax distribution over a discrete set
  of interchangeable "tokens" competing for a fixed attention budget. That means the Jain & Wallace-style
  "many different attention distributions give the same output" identifiability argument is less directly
  applicable (there's no permutation symmetry over tokens to exploit the way there is over words/patches),
  but a different problem replaces it: **CBAM's gate operates mid-network**, and its output still passes
  through the remaining ResNet stages, GAP, fusion, and cosine-similarity computation before a prediction
  is produced. So a CBAM attention map shows **which channels/locations got up- or down-weighted at that
  layer** — a mechanistic, "internal feature recalibration" fact — but it is **not**, by itself, an
  attribution of the *final* cosine-similarity decision. Conflating the two (i.e., holding up a CBAM
  spatial-attention heatmap and calling it "the model's explanation for grading this disc severe") is the
  single most common and easiest-to-avoid overclaim in CBAM-based medical-imaging papers; the thesis
  should compute a genuine end-to-end attribution (Score-CAM/SFAM/Grad-CAM at the final embedding) and
  present the CBAM map as a secondary, mechanistic visualization, explicitly labeled as such.

---

## 4. Attention rollout / attention flow, and Transformer-specific attribution — applicability to the ViT branch (RQ4)

- **Abnar & Zuidema, *"Quantifying Attention Flow in Transformers"*, ACL 2020.** Motivation: because
  self-attention mixes token representations across layers, raw attention weights from any single layer
  become an increasingly diluted/unreliable proxy for input-token relevance as depth increases. They
  propose two post-hoc corrections computed from the attention weights alone (no gradients): **attention
  rollout** (recursively multiply attention matrices layer-by-layer, with an identity term added per
  layer to account for the residual stream) and **attention flow** (a max-flow/graph formulation over the
  same DAG). Both were shown, on BERT, to correlate substantially better with ablation-based and
  gradient-based importance than raw single-layer attention. **Directly applicable to the ViT branch's
  internal self-attention stack** as a cheap, gradient-free baseline — but per Section 2/3, expect it to
  inherit CLIP's raw-attention background-bias problem unless combined with a fix (CLIP Surgery-style
  value-attention, or the gradient-augmented methods below); use it as a baseline to beat, not the final
  method.
- **Chefer, Gur, Wolf, *"Transformer Interpretability Beyond Attention Visualization"*, CVPR 2021.**
  Uni-modal (single Transformer) relevance propagation: assigns local relevance via Deep Taylor
  Decomposition principles and propagates it through **both attention layers and skip connections**
  (rollout only handles the attention path), integrating gradient information rather than using raw
  weights alone. Outperforms both raw-attention and rollout-only baselines for ViT image classification.
  Directly usable on the BiomedCLIP ViT branch in isolation (e.g. to explain the branch's own embedding
  before fusion). Code: `hila-chefer/Transformer-Explainability`.
- **Chefer, Gur, Wolf, *"Generic Attention-model Explainability for Interpreting Bi-Modal and
  Encoder-Decoder Transformers"*, ICCV 2021 (Oral), arXiv:2103.15679.** Extends the above to
  **bi-modal / co-attention architectures**, and — critically for you — **the official repository ships a
  worked CLIP example** (`Transformer-MM-Explainability/CLIP_explainability.ipynb`), because although
  CLIP's image and text towers are two *separate* uni-modal Transformers (they only interact once, at the
  final dot product/cosine similarity), the authors treat that final interaction as the point where
  per-modality relevance streams get combined, and apply their uni-modal relevance propagation
  independently inside each tower before combining. **This is close to plug-and-play for your ViT branch
  against a chosen text prompt** — it is, to date, the most directly reusable transformer-attribution
  codebase for a CLIP-style cosine-similarity head, and the fact that reference code already targets CLIP
  specifically is worth stating explicitly as justification for adopting it.
- **A detail worth stating explicitly in methodology, because it removes a plausible objection**: none of
  these gradient-based methods require the branch to be trainable. "Frozen" only means no weight updates
  during your training — backpropagation to *compute* gradients for attribution purposes works identically
  on frozen weights. There is no methodological barrier to running Chefer et al.'s method (or Grad-CAM
  variants) through the frozen BiomedCLIP tower.

---

## 5. Attributing the prediction between the two branches (RQ5)

This is the least standardized of the six questions — there is no single canonical "branch attribution"
paper for exactly your setup — but there are three established, composable building blocks, and your
specific architecture (**exactly two branches**) makes the most rigorous of them unusually cheap.

- **Shapley-value modality attribution is an established pattern in multimodal deep learning**, not
  something you'd be inventing from scratch:
  - **SHAPE — *"An Unified Approach to Evaluate the Contribution and Cooperation of Individual
    Modalities"*, arXiv:2205.00302.** Treats each modality as a "player" in a coalition game and computes
    Shapley values over subsets of modalities to measure both individual contribution and cross-modal
    cooperation/redundancy.
  - **SHAP-CAT, *"A interpretable multi-modal framework enhancing WSI classification via virtual staining
    and shapley-value-based multimodal fusion"*, arXiv:2410.01408 (2024).** Applies Shapley-based fusion
    weighting in a medical-imaging (whole-slide pathology) multimodal setting — a close domain precedent.
  - **The practical win specific to your architecture**: exact Shapley values require evaluating the value
    function on every subset of players, which is combinatorially expensive for 3+ modalities (hence most
    papers use Monte Carlo or kernel approximations). **With exactly two branches (CNN, CLIP), the power
    set has only 4 elements**, so the exact Shapley value is `φ_CNN = f(CNN,CLIP) − f(∅,CLIP)` and
    `φ_CLIP = f(CNN,CLIP) − f(CNN,∅)` (averaged if there's any asymmetry in how you define the "missing
    branch" baseline) — **no approximation needed, no combinatorial explosion**, just 2 extra forward
    passes per example with one branch zeroed/masked. This is genuinely a case where the "principled but
    expensive" method is cheap for your architecture, and worth stating as such.
- **Occlusion/ablation-based modality-contribution measures have a direct, peer-reviewed medical-imaging
  precedent for exactly this kind of question**: **Gapp, Tappeiner, Welk, Fritscher, Gizewski, Schubert,
  *"What are you looking at? Modality contribution in multimodal medical deep learning"*, International
  Journal of Computer Assisted Radiology and Surgery (CARS 2025).** They implement a **model- and
  performance-agnostic occlusion-based modality contribution method** across three medical multimodal
  problems and find some networks develop a **"modality preference"** that collapses toward using
  effectively one modality (unimodal collapse) regardless of the other's informativeness. This is a
  directly citable precedent and protocol template: zero-out (or mean-fill / noise-fill) one branch at a
  time, measure the drop in the target cosine-similarity score (or downstream metric), report per-branch
  and per-class contribution — and, importantly, it gives you a **known failure mode to actively check
  for** (a frozen dual-branch fusion collapsing onto one branch), which is a good hypothesis to test and
  report either way.
- **A third, cheaper and very standard option that fits an additive/concatenative fusion point especially
  well: Integrated Gradients**, Sundararajan, Taly, Yan, *"Axiomatic Attribution for Deep Networks"*, ICML
  2017. Applied not at the pixel level but **at the branch-embedding level** — i.e. treat the two 512-d (or
  whatever the pre-fusion dimensionality is) branch vectors as "the features," integrate gradients of the
  final cosine score along the path from a zero/baseline embedding to the actual branch embeddings, and
  sum the per-dimension attributions within each branch to get a scalar per-branch contribution. This is
  cheaper than Shapley (no baseline-subset enumeration needed beyond the standard IG path integral),
  satisfies IG's completeness axiom (the two branch attributions sum exactly to the score minus the
  baseline score, which is a nice, quotable property for a thesis defense), and requires only that fusion
  be differentiable — true regardless of whether fusion is concatenation+projection or weighted sum.
- **Explicit methodological caveat to state in the thesis**: none of the above give a single "objective"
  percentage split. Every one of them is defined **relative to a chosen baseline/missing-branch
  convention** (zero-embedding? mean-embedding over the training set? architecture-specific dropout
  token?), and different reasonable choices will shift the numbers. Report the convention explicitly and,
  ideally, show the branch-attribution result is qualitatively stable across at least two different
  baselines (e.g., zero-fill vs. mean-fill) — this preempts a natural methodological objection and mirrors
  the caveats already standard in the Shapley/IG explainability literature.

---

## 6. Text-side explanation: prompt perturbation for the class prototypes (RQ6)

- **There is now a purpose-built, published method for exactly this**: **Möller, Tilli, Vu, Padó,
  *"Explaining Caption-Image Interactions in CLIP Models with Second-Order Attributions"*, Transactions on
  Machine Learning Research (TMLR) 2025, code `exCLIP`.** This attributes the cosine-similarity score not
  just to individual image regions or individual text tokens in isolation, but to **interactions between
  specific image regions and specific caption words** — i.e. it can answer "which word(s) in the prompt,
  paired with which region(s) of the image, jointly drove this similarity score up." This is the most
  directly relevant citation for RQ6 and is stronger than simple perturbation because it captures
  cross-modal interaction, not just marginal word importance. It is the primary method to cite/adapt.
- **A simpler, fully legitimate fallback you can implement yourself in an afternoon**: classic
  **leave-one-word-out / occlusion-style perturbation** applied to the prompt string (the text-side
  analogue of Zeiler & Fergus-style occlusion sensitivity, or Li, Monroe & Jurafsky's *"Understanding
  Neural Networks through Representation Erasure"* (arXiv:1612.08220, 2016) for text). Concretely: for a
  frozen prompt like `"severe left neural foraminal narrowing"`, re-embed the prompt with each word
  removed (or replaced by a neutral token/synonym) through the frozen text tower, and measure the drop in
  cosine similarity to the fixed image embedding. This is cheap because the text tower is frozen and
  prompts are short (a handful of forward passes per class), doesn't require any gradient access to the
  text tower, and is trivially reproducible/auditable by a thesis committee.
- **Complementary conceptual grounding**: Gandelsman et al. (ICLR 2024, Section 2 above) establish that
  CLIP's *image* embedding is itself natively decomposable into text-describable components — this
  supports the general framing that "text is the natural currency of explanation" for this model family,
  strengthening the case for investing effort in the text-side analysis rather than treating it as an
  afterthought to the visual CAM work.
- **Important limitation to state explicitly, and it's specific to your setup**: BiomedCLIP's text tower
  was pretrained on broad biomedical (PubMed figure-caption) text, **not** on lumbar-spine radiology
  report language specifically, and the class prompts are hand-authored by you/the clinical
  collaborator, not mined from real reports. So word-perturbation results characterize **the frozen text
  encoder's sensitivity to your specific prompt phrasing**, which may reflect general biomedical-text
  statistics or even generic English syntax/salience biases rather than ground-truth radiological
  importance (e.g. it might be highly sensitive to "severe" for reasons unrelated to how radiologists
  actually reason about severity). Frame results as a **prompt-design audit / sanity check**
  ("is the frozen text encoder responding to the clinically intended words, and does swapping
  'left'↔'right' correctly flip the predicted laterality?"), not as a claim about radiological reasoning.

---

## 7. Concrete guidance

### 7.1 What is technically valid to claim

| Claim | Valid? | Basis |
|---|---|---|
| Gradient of cosine score w.r.t. CNN feature maps is a meaningful (if reinterpreted) importance signal | Yes | SFAM (Liao et al. 2025), GAM (2109.00951) — established literature for exactly this head type |
| Score-CAM / Ablation-CAM (perturbation-based) work unmodified on this head | Yes | They only require evaluating the score function on masked inputs; no linearity assumption. gScoreCAM (Chen et al., ACCV 2022) is the direct CLIP precedent |
| gScoreCAM / CLIP-Surgery-style / MaskCLIP-style methods are appropriate for the ViT branch | Yes | Purpose-built for CLIP's cosine-similarity scoring and documented failure of raw attention |
| Chefer et al.'s bi-modal relevance propagation is applicable to the ViT branch against a text prompt | Yes | Official reference implementation targets CLIP directly |
| Backprop-based attribution works even though branches are frozen | Yes | "Frozen" affects weight updates only, not gradient computation |
| Exact 2-player Shapley or Integrated-Gradients branch attribution is tractable and principled here | Yes | Only 2 branches → exact Shapley is cheap; IG gives an exact completeness decomposition |
| Occlusion-based branch/modality contribution measurement | Yes, with caveats | Direct medical-imaging precedent (Gapp et al. 2025); result depends on the missing-branch fill convention — state it |
| Second-order / interaction-based text-image attribution for prompt words | Yes | Purpose-built method exists (Möller et al., TMLR 2025) |
| Simple leave-one-word-out prompt perturbation | Yes, as an audit tool | Standard, cheap, legitimate, but see the specific limitation below |

### 7.2 What would be methodologically wrong to claim

1. **"CLIP attends to the left foraminal region" from a raw attention map or naive rollout on the ViT
   branch.** CLIP Surgery and MaskCLIP independently document that raw CLIP self-attention is unreliable
   (background-biased, "opposite" visualizations). Requires gScoreCAM, Chefer et al.'s method, or a
   CLIP-Surgery/MaskCLIP-style architectural fix instead.
2. **Presenting the Grad-CAM channel weights on this head as literally "the class weight vector"**, the
   way one would for a linear softmax layer. The L2-normalization changes what the gradient represents
   (a tangential/projected direction, not the raw prototype). Either explain this precisely, or use a
   perturbation-based (Score-CAM-family) method that avoids the issue.
3. **Showing CBAM channel/spatial maps or the raw slice-attention-pool weights as *the* explanation for
   the final grading decision**, unvalidated. Per Jain & Wallace (2019) → Wiegreffe & Pinter (2019) →
   Wu et al. (CVPR 2024): raw attention weights are plausible at best, not faithful by default, and CBAM's
   maps specifically describe **mid-network feature recalibration**, not end-to-end attribution. Validate
   with a faithfulness check (deletion/insertion, gradient agreement) or pair with a genuine end-to-end
   CAM at the final embedding before presenting either as "the" explanation.
4. **Applying textbook attention rollout (Abnar & Zuidema, 2020, designed/validated on BERT for NLP
   classification) to the ViT or slice-attention-pool without adaptation** and treating the result as
   established/validated for this setting. It's a reasonable *baseline* to report, not a validated final
   method for this architecture family; expect it to underperform gradient-augmented alternatives, per
   Wu et al. (2024).
5. **Reporting a single Shapley-value or occlusion-based branch-contribution percentage as objective
   ground truth.** These are baseline/convention-dependent by construction. State the convention, and
   ideally show stability across ≥2 conventions.
6. **Treating prompt-word-perturbation results as evidence about clinical/radiological reasoning.** They
   audit the frozen (biomedical-general, not spine-report-specific) text encoder's sensitivity to your
   specific prompt phrasing — a prompt-design sanity check, not a claim about medical semantics.

### 7.3 Three architecture-specific angles worth building into the thesis (genuinely underexplored combination, not just "apply known method X")

1. **Exact two-branch attribution via Shapley/Integrated Gradients at the fusion point**, cross-validated
   against occlusion-based branch ablation (following the Gapp et al. 2025 medical-imaging protocol).
   Because there are exactly two frozen branches, the usually-expensive "how much did each modality
   contribute" question becomes cheap and exact rather than approximated — a case where a generally
   heavyweight method (Shapley) is unusually well-suited to your specific architecture. This directly
   answers RQ5 and is a natural chapter contribution: no comparable dual-branch-attribution study for a
   CNN+CLIP spine-grading model appears to exist.
2. **A tractable, exhaustive faithfulness audit of the 9-slice attention-pooling layer.** Most of the
   "attention is/isn't explanation" literature (Jain & Wallace 2019, Wiegreffe & Pinter 2019) is stuck
   doing indirect, sampled, or adversarially-trained diagnostics because NLP sequences are long. Your
   slice-attention pool has only **9 tokens**, so exhaustive or near-exhaustive tests — leave-one-slice-out
   ablation, uniform-weight-baseline comparison, exact 9-slice Shapley (2⁹ = 512 forward passes, trivial),
   even the Wiegreffe & Pinter frozen-weight / adversarial-attention diagnostics — are computationally
   trivial here in a way they rarely are elsewhere. Running these and reporting *for this specific
   trainable attention layer* whether it lands on the Jain & Wallace side or the Wiegreffe & Pinter side
   of the debate is a clean, self-contained, genuinely novel empirical result the thesis can claim, and it
   directly and rigorously answers the "is attention valid explanation" question for one of your two
   attention mechanisms (the trainable one, where it's actually answerable, as opposed to CBAM/ViT
   attention which are harder to test exhaustively).
3. **A composite, multi-level explanation pipeline that mirrors the model's actual computation graph**:
   branch-level attribution (angle 1) → within-branch spatial CAM (gScoreCAM/CLIP-Surgery-adapted for the
   ViT branch; Score-CAM/SFAM for the 3D-CNN branch) → text-side interaction attribution over the winning
   prompt's words (Möller et al.-style or leave-one-word-out). Presented together (e.g. "68% of this
   prediction came from the CLIP branch; within that branch, attention concentrated on the left foraminal
   region across slices 4–6; within the winning prompt, removing the word 'severe' dropped the score by
   X"), this is a coherent, hierarchical explanation that no single off-the-shelf method in the literature
   provides end-to-end for a dual-branch cosine-similarity VLM — the "novel" contribution is less any one
   piece (each piece is separately established, see Sections 1–6) than the principled composition of them
   along the architecture's actual data flow, explicitly justified against the failure modes documented in
   Section 7.2.

---

## Reference list (author, year, venue)

- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017/2020). Grad-CAM:
  Visual Explanations from Deep Networks via Gradient-based Localization. ICCV 2017 / IJCV 2020.
- Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., & Hu, X. (2020). Score-CAM:
  Score-Weighted Visual Explanations for Convolutional Neural Networks. CVPR Workshops 2020
  (arXiv:1910.01279).
- Desai, S., & Ramaswamy, H. G. (2020). Ablation-CAM: Visual Explanations for Deep Convolutional Network
  via Gradient-free Localization. WACV 2020.
- Liao, Y., Akpudo, U. E., Zhang, J., Gao, Y., Zhou, J., Zeng, W., & Zhang, W. (2025). Visual Explanation
  via Similar Feature Activation for Metric Learning (SFAM). arXiv:2506.01636.
- (2021). GAM: Explainable Visual Similarity and Classification via Gradient Activation Maps.
  arXiv:2109.00951.
- Chen, P., Li, Q., Biaz, S., Bui, T., & Nguyen, A. (2022). gScoreCAM: What is CLIP looking at? ACCV 2022.
- Li, Y., Wang, H., Duan, Y., Zhang, J., & Li, X. (2023/2025). CLIP Surgery for Better Explainability with
  Enhancement in Open-Vocabulary Tasks. arXiv:2304.05653; Pattern Recognition (2025).
- Zhou, C., Loy, C. C., & Dai, B. (2022). Extract Free Dense Labels from CLIP (MaskCLIP). ECCV 2022
  (Oral).
- Gandelsman, Y., Efros, A. A., & Steinhardt, J. (2024). Interpreting CLIP's Image Representation via
  Text-Based Decomposition. ICLR 2024 (Oral), arXiv:2310.05916.
- Gong, S., Lei, H., Dou, Q., & Farnia, F. (2025). Boosting the visual interpretability of CLIP via
  adversarial fine-tuning. ICLR 2025.
- Jain, S., & Wallace, B. C. (2019). Attention is not Explanation. NAACL-HLT 2019.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation. EMNLP 2019.
- Mehrani, P., & Tsotsos, J. K. (2023). Self-attention in vision transformers performs perceptual
  grouping, not attention. Frontiers in Computer Science.
- Wu, J., Kang, W., Tang, H., Hong, Y., & Yan, Y. (2024). On the Faithfulness of Vision Transformer
  Explanations. CVPR 2024.
- Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. ECCV
  2018.
- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL 2020.
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization.
  CVPR 2021.
- Chefer, H., Gur, S., & Wolf, L. (2021). Generic Attention-model Explainability for Interpreting
  Bi-Modal and Encoder-Decoder Transformers. ICCV 2021 (Oral), arXiv:2103.15679.
- (2022). SHAPE: An Unified Approach to Evaluate the Contribution and Cooperation of Individual
  Modalities. arXiv:2205.00302.
- (2024). SHAP-CAT: An interpretable multi-modal framework enhancing WSI classification via virtual
  staining and Shapley-value-based multimodal fusion. arXiv:2410.01408.
- Gapp, C., Tappeiner, E., Welk, M., Fritscher, K., Gizewski, E. R., & Schubert, R. (2025). What are you
  looking at? Modality contribution in multimodal medical deep learning. International Journal of
  Computer Assisted Radiology and Surgery (CARS 2025).
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks (Integrated
  Gradients). ICML 2017.
- Möller, L., Tilli, P., Vu, N. T., & Padó, S. (2025). Explaining Caption-Image Interactions in CLIP
  Models with Second-Order Attributions. Transactions on Machine Learning Research (TMLR).
- Li, J., Monroe, W., & Jurafsky, D. (2016). Understanding Neural Networks through Representation
  Erasure. arXiv:1612.08220.
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity Checks for
  Saliency Maps. NeurIPS 2018.
- Kim, E., Kim, S., Seo, M., & Yoon, S. (2021). XProtoNet: Diagnosis in Chest Radiography with Global and
  Local Explanations. CVPR 2021. (Background: prototype-based, spatially-localized explainable medical
  classification — relevant precedent for prototype-style reasoning in medical imaging, though not
  cosine-similarity-to-text specifically.)

Not independently verified beyond search-snippet level (treat citation details as provisional if used):
Zheng et al., *Visual Explanation for Deep Metric Learning*, arXiv:1909.12977 (2019).
