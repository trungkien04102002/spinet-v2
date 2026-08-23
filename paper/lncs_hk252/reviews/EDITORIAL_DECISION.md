# Editorial Decision — MIWAI 2026 Submission

**Paper:** Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar IVD Grading and Cross-Dataset Zero-Shot Transfer
**Review mode:** full (5 reviewers, double-blind)
**Date:** 2026-06-02

---

## Decision: MAJOR REVISION

The Devil's Advocate raised CRITICAL issues that, per review protocol, preclude an Accept. The panel is split between Weak Accept (Domain, Perspective) and Major Revision / weak-reject (Methodology, Devil's Advocate), with the EIC at Minor Revision. The deciding factor: **two of the three CRITICAL issues require no new training and are cheaply fixable**, so the paper is recoverable to acceptance well within the MIWAI camera-ready window — but as currently framed the central contribution is not yet defensible.

### Panel scoreboard

| Reviewer | Recommendation | Overall |
|---|---|---|
| EIC | Minor Revision | 72 |
| R1 Methodology | Major Revision | 46 |
| R2 Domain | Weak Accept | 76 |
| R3 Perspective (VLM) | Weak Accept | 58 |
| Devil's Advocate | (CRITICAL flags) | — |

---

## Cross-Reviewer Consensus Matrix

| # | Issue | Raised by | Max severity | New data needed? |
|---|---|---|---|---|
| C1 | Single-seed RSNA headline stated confidently in abstract; SPIDER uses 3 seeds → internal inconsistency with paper's own >2σ standard | EIC, R1, DA | **CRITICAL** | Ideally yes (3-seed RSNA) — but prose hedge works as interim |
| C2 | Trivial-ensemble confound (logit-avg Base+BMC) admitted (L313) but untested — undercuts the #1 "fusion/regularizer" claim | R1, DA | **CRITICAL** | Yes, but CHEAP (reuse existing checkpoints, no training) |
| C3 | Zero-shot prompt strings {q_k} never shown → Table 2 irreproducible | R1, R3 | **CRITICAL** | No — prompts already exist in code, just document them |
| C4 | "Naked BiomedCLIP" beats-on-disc-labels claim (L265) is prose-only, no numbers/table row; it's the only fair zero-shot head-to-head | EIC, R3 | MAJOR | Numbers likely already computed (predictions exist) — add a column |
| C5 | ">2σ" significance invalid: n=3 underpowered, "pooled σ" wrong denominator (should be SE of difference), no named test/p-value | R1 | **CRITICAL** | No — restate honestly as descriptive, not inferential |
| C6 | Abstract/title oversell accuracy-superiority; body walks back to parity-with-baseline + regularizer; contribution ordering is a rhetorical anchor on CBAM-only | EIC, DA | MAJOR | No — reframe prose |
| C7 | T2-FS (SPIDER) vs T2 (RSNA) imaging-physics gap never acknowledged; blamed Modic transfer solely on "semantic distance" | R2 | MAJOR | No — one sentence |
| C8 | Sub-chance zero-shot AUC (Modic 0.422, Pfirrmann 0.492) glossed as "semantic gap" rather than possible prompt/label-ordering error | R2, R3, DA | MAJOR | No — acknowledge honestly |
| C9 | Clinical "right trade-off" asserted with no operating-point / threshold analysis | DA | MAJOR | No — soften wording |
| C10 | Modic "0–3" (L184) vs "4-class" (L257/270) inconsistency | R2 | MINOR | No |
| C11 | Bold-best rule inconsistent (Mean Accuracy bolded "best" in Tab 1 though dismissed in text) | EIC | MINOR | No |
| C12 | Uncited refs in references.bib (hallinan2021 unused; ~7 dangling) → citation-inflation impression; nigru2024 bibentry incomplete | EIC, R2 | MINOR | No |

**Genuine strengths the panel agreed on:** honest reporting (explicitly declines to claim hybrid > baseline); sound metric choice (macro-avg + AUPRC with prevalence baseline); credible second-reader clinical positioning; the regularizer finding is the strongest, statistically-grounded hook; SOTA "not directly comparable" argument is fair, not a dodge; correct citation of Pfirrmann/Modic/SPIDER.

---

## Revision Roadmap (prioritized)

### P0 — must fix (blocks acceptance)

- **P0-1 (C5, C6):** Rewrite the ">2σ" claims as **descriptive** ("the gap exceeds twice the pooled seed standard deviation; with n=3 seeds we report this as a descriptive effect size, not an inferential test"). Remove any wording implying a significance test. Reframe abstract to lead with **regularizer + label-space extension**, not accuracy superiority. *(prose only)*
- **P0-2 (C1):** Explicitly label Table 1 / all RSNA headline numbers as **single-seed (seed 42)**; move the caveat from Limitations into the abstract/results framing; soften "4.0×" to "single-seed 4.0×". *(prose only; stronger fix = run 3-seed RSNA)*
- **P0-3 (C3):** Add the actual zero-shot **prompt strings** (a small table or inline list) + state whether prompts were fixed a priori or tuned on SPIDER val, and the temperature value. *(documentation — prompts already exist in code; DO NOT invent them, pull from the repo)*
- **P0-4 (C2):** Either (a) run the **trivial logit-averaging ensemble** (Base+BMC, no training, reuse checkpoints) and report it, or (b) if not run, soften the "complementary fusion / regularizer mechanism" claim to a hypothesis and state the ablation is the immediate next step. *(prefer (a) — cheap and decisive)*

### P1 — strongly recommended

- **P1-1 (C4):** Add a **Naked-BiomedCLIP** column/row to Table 2 with real numbers (verify they exist in `experiments/`; do not fabricate). If unavailable, downgrade the L265 claim to clearly-hedged.
- **P1-2 (C7):** Add one sentence acknowledging the **T2-FS vs T2** marrow-signal imaging gap as a co-explanation for poor Modic transfer.
- **P1-3 (C8):** Acknowledge that **sub-chance AUC** (Modic/Pfirrmann) indicates possible prompt/label-ordering failure, not only semantic distance.
- **P1-4 (C9):** Soften "the right clinical trade-off" — note it depends on the deployment operating point.

### P2 — polish

- **P2-1 (C10):** Fix Modic "0–3" vs "4-class" wording consistently.
- **P2-2 (C11):** Make the bold-best rule consistent (don't bold Mean Accuracy as "best" if the text argues it's not the right metric, or add a footnote).
- **P2-3 (C12):** Cite `hallinan2021` for the 3-of-5 condition split (it's already in the .bib); complete the `nigru2024` bibentry; remove or cite dangling .bib entries.

---

## Note on scientific integrity

Items P0-3, P0-4, P1-1 reference data/experiments. The author MUST pull real prompt strings and real numbers from the repository, or run the (cheap, no-training) ensemble — and MUST NOT invent any value. If a number cannot be obtained before the deadline, the corresponding claim must be hedged/removed, not fabricated.
