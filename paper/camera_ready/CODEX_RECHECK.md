# Camera-ready revision (round 2) change log and rebuttal for re-check

**Paper 122, MIWAI 2026.** Files: `main.tex` (12 pp), `RESPONSE_TO_REVIEWERS.{md,tex,pdf}`.

## What changed since the previous re-check

Codex was right. The earlier decision to keep the submitted "linear head" description of the Hybrid was reversed, because:

1. The submitted **Figure 1 caption** ("a text-aligned vector scored by cosine similarity against text prompts, so the label set, whether a trained set or a new one, becomes an input at inference") and the **abstract** ("predictions are made by matching this embedding to text prompts") already describe cosine-to-text for the trained set. Sections 3.1 and 3.2 saying "linear head" therefore contradicted the paper's own figure and abstract. Fixing 3.1 and 3.2 to cosine **aligns them with the submitted figure and abstract**, so it is a clarification, not a method change.
2. Verified in code and run logs (not just narrated): `grading_hybrid.py forward()` computes `logit_scale.exp()*(image_emb @ text_embs.T)`; `train_rsna_hybrid.py` applies focal loss to those cosine logits. The reported Table 3 to 5 runs used, per `experiments/hybrid/best_metrics_hybrid_fixed_e7.json` and the seed logs: `use_focal=True (gamma 2.0)`, `use_supcon=True (weight 0.1)`, `use_uncertainty=True`, `class_weight_mode=sqrt`, `oversample=3`, `augmentation=medium`. So the Hybrid/Multimodal supervised objective is focal-on-cosine + SupCon (0.1) + uncertainty weighting, with a learned logit scale.

## Manuscript changes (all clarify the implementation actually used; no result changed)

- **Sec. 3.1**: Eq. (2) linear head now attributed to the SpineNetV2 and CBAM-only baselines; Multimodal-only and Hybrid use cosine logits (Eq. 3) already during source training; s is the learned logit scale; the over-strong "Phi produces an embedding in the same space as psi" claim was removed.
- **Sec. 3.2 Task heads**: same baseline-vs-cosine distinction; source labels and swapped zero-shot prompts both scored by cosine.
- **Sec. 3.2 trainable list, Fig. 1 caption, Table 2 caption**: now include the scalar logit scale.
- **Sec. 3.3**: adds that focal is applied to cosine logits for Multimodal/Hybrid, plus the supervised-contrastive term (weight 0.1, temp 0.07, `khosla2020supcon`) and homoscedastic-uncertainty task weighting (`kendall2018multitask`) instead of a plain sum. Two new bib entries.
- **R1 fixes retained**: numbered Eq. (1) to (3); C,D,H,W defined; "0.045 to 0.050"; Severe Recall (%) row added (Severe F1 was already present); "nine" to "ten".
- **R2/R3 fixes**: inference-cost note (parameter-efficient to train, not lightweight to deploy); RTX 4090 about 16 GB peak memory; prompt-sensitivity limitation; softened Modic and clinical ("80%" removed); partial text-space compatibility + dedicated alignment loss as future work; "controlled comparisons with newer medical VLMs left to future work" added to related work.
- **Style**: removed all em-dashes and en-dashes from body prose per author preference (ranges written with "to" or single hyphen).
- **Camera-ready**: de-anonymized author block, restored acknowledgments. Trimmed prose to hold 12 pages (no number, table, figure, or claim removed).

## Verification: numbers are unchanged

A line-diff of the submitted `main.tex` (from `HK252_..._latest.zip`) against the current `main.tex`, restricted to lines containing tabular numeric cells, differs by exactly one line: the added `Severe Recall (%)` row. Every other metric in Tables 1 to 5 is byte-identical. No experiment was re-run.

## Round-3 addendum (after Codex round-2 review)

Two blockers Codex raised were verified against code/logs and fixed:

- **Baseline loss.** `train_rsna_baseline.py` uses `nn.CrossEntropyLoss` ("Loss: CrossEntropyLoss" in `run_baseline_seed123.log`); CBAM-only, Multimodal-only, and Hybrid use focal. The earlier "All configurations are trained with the imbalance-aware focal loss" was wrong and is now: "The SpineNetV2 baseline is trained with standard cross-entropy; the other three configurations use the imbalance-aware focal loss" (Sec. 3.1). This also matches the Introduction's "cross-entropy training" baseline framing.
- **Uncertainty scalars.** `UncertaintyLoss` (Kendall) has a learned log-variance per task and is in the optimizer (`train_rsna_hybrid.py` L556). The trained-parameter list (Sec. 3.2, Fig. 1 caption, Table 2 caption) now reads "slice-attention pool, fusion MLP, scalar logit scale, and per-task uncertainty scalars". The 1.18 M count is unchanged (these are a few scalars).

Also applied: cosine head renamed $c^{zs}\to c^{cos}$ since it is used in supervised source training too; the Sec. 3.1 "only the head and label set change" sentence reworded (for the Hybrid the cosine head is fixed and only the prompt-defined label set changes at zero-shot); SupCon described on "the resulting image embeddings" (Multimodal-only has no fusion); the Table 5 footnote per-seed values restored (so no number was removed); contribution wording "as if stopping attention..." changed to "suggesting the frozen branch mitigates dataset-specific overfitting". Still 12 pages, numbers unchanged.

## Re-check checklist

1. Confirm Sec. 3.1 / 3.2 / 3.3 now describe cosine-to-text training for Multimodal/Hybrid, consistent with Fig. 1 and the abstract, and that Eq. 2 is scoped to the baselines.
2. Confirm SupCon (0.1) and uncertainty weighting appear in Sec. 3.3 with citations, and the logit scale is listed among trained parameters (Sec. 3.2, Fig. 1, Table 2).
3. Confirm R3 response says training constrains compatibility with the source anchors, no additional contrastive loss, partial for unseen prompts (Table 4).
4. Confirm 12 pages, no undefined refs, no "??", and no rendered em/en dashes.
5. Confirm all Table 1 to 5 numbers match the submitted paper (only the Severe Recall row is new).
