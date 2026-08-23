# Editorial Decision — RE-REVIEW (revised MIWAI paper, 2026-06-03)

5-reviewer panel on the revised `main.tex` (after C1-C4 fixes + 3-seed + reproducible Table 2).

## Decision: MINOR REVISION (accept-leaning)

All 5 reviewers agree the revision is **genuine and rigorous** and the paper **clears the MIWAI/LNAI bar**. C1+C2 (the two prior CRITICALs) are fixed; C3+C4 are in the manuscript with verified numbers. **No new experiments needed.** The one remaining gating issue is unanimous: **FRAMING / OVERCLAIM**.

| Reviewer | Verdict | Overall |
|---|---|---|
| EIC (RR0) | Minor Revision | 73 |
| Methodology (RR1) | Accept w/ Minor | 69 |
| Domain (RR2) | Accept w/ Minor | ~76 |
| Devil's Advocate (RR3) | Borderline / weak-accept | — |
| Perspective/VLM (RR4) | Accept (borderline) | ~62 |

## Consensus (≥3 reviewers agree)

1. **[CRITICAL — FIXED] Line 93 stale single-seed numbers** (0.528/0.343) contradicted the 3-seed Table 1. → corrected to 0.527/0.356 on 2026-06-03.

2. **[MAJOR — framing] Title + abstract OVERCLAIM** relative to the now-honest body: they lead on zero-shot / architecture superiority, but (a) off-the-shelf BiomedCLIP beats the hybrid on overall zero-shot mean F1 (0.394 vs 0.362), and (b) the trivial ensemble beats the learned fusion on AUC/AUPRC. All 5 flag this. The science is sound; the positioning reads as "honest but no clear win."
   - **Fix (framing only, no experiments):** lead on the ONE finding that is multi-seed, >2σ, novel, and NOT undercut — the **CBAM-hurts-transfer / frozen-BiomedCLIP-as-cross-dataset-regularizer** result + the clinical Severe-class recovery. Scope zero-shot honestly to the disc-morphology subset. Surface the two honest losses in the abstract and name the axis the hybrid wins (F1 / Severe-F1 at the clinical operating point + label-extension capability).
   - EIC suggested retitle e.g. *"When Attention Hurts Transfer: A Frozen BiomedCLIP Branch as a Cross-Dataset Regularizer for Lumbar Disc Grading."*

3. **[MAJOR] Value-prop clarity** (Domain + EIC + DA): 3 competing leads (Severe-recall fix / zero-shot extension / regularizer). Add one triage sentence separating the **clinical** result (Severe recall) from the **scientific** result (VLM-as-regularizer) from the **capability** (zero-shot).

## Per-reviewer notable MAJORs (camera-ready, non-blocking)

- **Methodology:** ">2σ" is really 2.6× and uses raw per-seed σ, not SE-of-difference — run one bootstrap/Welch CI on the existing seeds (or soften wording further).
- **Perspective/VLM:** single prompt template + no prompt-ensembling; **temperature value never printed** (the un-analyzed cause of high-AUC/collapsed-F1, e.g. spondylolisthesis AUC 0.807/F1 0.03 — a threshold sweep would likely recover F1); frame Modic AUC 0.45 as the known fine-grained-ordinal weakness of contrastive medical VLMs.
- **Domain:** Modic also needs the paired T1 sequence (absent) — note as a task-input limit, not only fat-sat physics. Summary leads with AUC 0.899 — foreground F1/recall/AUPRC instead.
- **EIC:** add a "Mean (disc-morphology)" sub-row to Table 2 so the hybrid's selective win is visible next to the honest overall OTS lead; unify bold convention across Tables 1/2; de-bold Mean Accuracy.

## What genuinely survives (all reviewers affirm)
- 3-seed Severe-class recovery on RSNA (Recall 12.3→48.6%, F1 0.152→0.356) — real, reproducible.
- The CBAM-hurts / BiomedCLIP-restores cross-dataset finding (>2σ on SPIDER 3-seed) — the strongest, cleanest thesis.
- Exceptional honesty (reports its own losses) — a positive at this venue.
- Reproducible Table 2, disclosed prompts, real ensemble control.

## Bottom line
Accept-leaning. The fix is **positioning, not new work**: retitle + re-lead abstract on the regularizer headline, surface the honest losses, add a disc-morphology sub-row, print the temperature, add the Modic-T1 caveat. All fit a camera-ready pass within 12 pages.
