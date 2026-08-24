# Multi-Perspective Peer Review Report — REPORT_PAPER.tex

**Date:** 2026-05-14
**Paper:** Hybrid 3D-CBAM Attention with Frozen Vision-Language Embeddings for Lumbar Intervertebral Disc Grading on Multi-Center MRI
**Target venue:** IEEE EMBC 2027 (Rank-C, 4-page conference short paper)
**Authors:** Trung Kien Ha, Nhan Phan
**Review skill:** `academic-research-skills:academic-paper-reviewer` v1.9.0
**Reviewers:** 5 independent personas (EIC + 3 peer + Devil's Advocate)

---

## CONTEXT NOTES (added 2026-05-14 by user)

### 1. Visibility of internal artifacts
The user does NOT plan to publish `RESULTS_LOG.md`, raw checkpoint metrics files, internal `.md` notes,
or this `REVIEW_REPORT.md` to a public repository. The reviewers' "any reviewer will spot-check the
raw logs and catch the number mismatch" framing assumed public-repo access, which does not apply here.

**Therefore the priority of fixes shifts:**
- Numerical accuracy of paper claims is still mandatory (for the advisor's review, for LVTN defense,
  for the author's own honesty, and for any future public release).
- The "5-minute reviewer spot-check" risk is reduced, but the **advisor will verify against the same logs**
  during LVTN defense, so the inconsistency must still be fixed.
- Citation accuracy and methodological disclosure remain mandatory regardless of repo visibility.

### 2. Outstanding experiment status (as of 2026-05-14)
- **SPIDER 8-label retrain (4 configs)**: code ready (commit f0e9611), **NOT YET RUN on GPU**.
- **RSNA 3-seed runs**: **NOT YET RUN**. Planned for next GPU session.
- **BMC-only SPIDER 4-label**: **NOT YET RUN**. Planned alongside 8-label.
- **Linear-probe naked BiomedCLIP RSNA**: **NOT RUN** (would require small new script).
- **Mendeley external eval**: **NOT RUN** (deferred per user, post-LVTN).

---

## EXECUTIVE SUMMARY

**Editorial Decision: 🟡 MAJOR REVISION** (risk of Reject under strict reviewer)

**Reviewer scores:**
| Reviewer | Score(s) |
|---|---|
| EIC | Major Revision conditional on length + citation + seed variance |
| R1 Methodology | Rigor 5.0/10 · Reproducibility 4.5/10 |
| R2 Domain | Domain Contribution 6/10 · Literature Coverage 4/10 |
| R3 Perspective | Originality 6.5/10 · Significance 6.0/10 |
| Devil's Advocate | "Likely Major Revision, leaning Reject if strict" |

**Top 5 critical issues (consensus across ≥3 reviewers):**
1. Numerical inconsistency in headline Severe Recall (abstract: 11.9%→49.6%, raw logs: 11.5%→46.4%, ×4.2 should be ×4.0)
2. Single-seed evaluation; fusion gain +0.026 F1 < documented seed swing 0.10
3. BMC-only baseline is degenerate (epoch 3, 3.2 min vs CBAM 56 min)
4. Title "Multi-Center MRI" overclaim — training is single-dataset (RSNA only)
5. SPIDER Hybrid Table II uses v2 checkpoint (undisclosed), not v3 — different model from RSNA Table I

---

## PHASE 0 — FIELD ANALYSIS

**Primary discipline:** Medical Image Analysis / Computer Vision (Biomedical AI)
**Secondary:** Vision-Language Models in Healthcare; Clinical AI for Spine MRI
**Research paradigm:** Empirical (deep learning ablation study)
**Methodology type:** Architectural innovation + ablation
**Target venue:** EMBC 2027 (4-page short paper, Rank-C)
**Paper maturity:** Pre-submission draft, post-advisor revision round 1

**Reviewer personas configured:**
1. **EIC** — IEEE EMBC track editor (Clinical AI / Medical Imaging); strategic and venue-fit perspective
2. **R1 Methodology** — Senior ML researcher; CV ablation rigor, class imbalance, statistical validity
3. **R2 Domain** — Clinical CV / radiologist familiar with spine MRI; literature coverage, clinical claims
4. **R3 Perspective** — Foundation-models / medical VLM researcher; alternative methods, broader impact
5. **Devil's Advocate** — Adversarial; cherry-picking detection, logical fallacies, strongest counter-argument

---

## CONSENSUS CRITICAL ISSUES (Tier 1)

### C1. Numerical inconsistency in headline Severe Recall claim
**Flagged by:** R1 (Critical C1), Devil's Advocate (Critical C2)

**Paper claim** (Abstract L64-66, Conclusion L623):
- Baseline Severe Recall: 11.9%
- Hybrid Severe Recall: 49.6%
- Improvement: ×4.2

**Verified from raw logs** (`/Users/kienha/spinet-v2/experiments/v3_20260503/RESULTS_LOG.md` + `~/Desktop/v3_check/checkpoints/v3_20260503/hybrid/hybrid_best_metrics.txt`):

Baseline per-condition Severe Recall:
- spinal_canal: 0.346
- left_foraminal: 0.000
- right_foraminal: 0.000
- **Mean: 0.115 = 11.5%**

Hybrid per-condition Severe Recall:
- spinal_canal: 0.704
- left_foraminal: 0.375
- right_foraminal: 0.313
- **Mean: 0.464 = 46.4%**

**Actual multiplier:** 0.464 / 0.115 = 4.03 → rounds to ×4.0

**Action required:** Update Abstract, §V.A "Severe class is the headline result" (L421-425), §VI Conclusion (L623). Replace 11.9%→11.5%, 49.6%→46.4%, ×4.2→×4.0.

### C2. Single-seed evaluation
**Flagged by:** EIC (issue #1 to flag), R1 (Critical C2), R2 (Major M4), R3 (Minor m3), Devil's Advocate (Critical C1)

**Issue:** All Table I numbers from one seed (random_state=42). Hybrid-over-CBAM Mean F1 gain is +0.026 (0.502→0.528); Severe F1 gain +0.039. RESULTS_LOG.md internally documents:
- v2 vs v3 CBAM Severe AUPRC delta: -0.028 ("within seed-to-seed noise")
- v2 vs v3 SPIDER Hybrid Mean F1: 0.622 → 0.519 (-0.10, "stochastic")

**Documented seed variance ≈ 4× the headline fusion gain.** Without multi-seed mean±std or paired statistical test, the central "Hybrid > either branch" claim is not defensible.

**Action required:** Run seeds {0, 7, 42} for all 4 RSNA configs. Existing code accepts `--seed` flag (commit `fc0eacc`). Cost: ~$15 on Vast.ai. Report mean ± SD for {Mean F1, Severe F1, Severe AUC, Severe AUPRC}.

### C3. BMC-only is a degenerate run, not a fair baseline
**Flagged by:** R1 (Critical C3), Devil's Advocate (Critical C3)

**Evidence** (`hybrid_biomedclip_only_best_metrics.txt`):
- Best epoch: 3
- Train time: 3.2 min
- Compare: Baseline ep 18 (16.4 min), CBAM ep 14 (56 min), Hybrid ep 10 (24.8 min)
- Patience: 15 epochs — so 3 epochs is the FIRST best-on-val checkpoint, not a patience artifact
- Severe Recall on right_foraminal: 9.6% only

Almost certainly capacity-limited under-fitting, not real convergence. Using this as the "BiomedCLIP-only" ablation against converged CBAM/Hybrid is methodologically unfair.

Additionally, the "BMC-only" model is *not* a true frozen-VLM linear probe — it loads the CBAM checkpoint, routes through the Hybrid fusion MLP with CBAM features zero-masked, and inherits the full 4-component imbalance pipeline. It's a partially-ablated Hybrid.

**Action required:**
1. Either re-run BMC-only with `min-epochs=10` or `--no-early-stop` to ensure fair comparison
2. AND add a true linear-probe naked BiomedCLIP baseline (Run 8 was in the skipped list)
3. Rename row to "Hybrid w/o CBAM features" and document zero-masking surgery in §III

### C4. Title overclaim: "Multi-Center MRI"
**Flagged by:** R2 (Critical C3), R3 (Critical C2), Devil's Advocate (Critical C4)

**Issue:** Title states "Multi-Center MRI" but training+validation use only RSNA 2024 (single 80/20 patient-level split, `random_state=42`). SPIDER (Europe) is used only for zero-shot evaluation on different labels — not multi-center for the same task.

Per Park & Han *Radiology* 2018 standards, "multi-center validation" requires same-task evaluation across institutions, which is not done.

**Action required:** Either:
- (a) Change title to "Lumbar Intervertebral Disc Grading on RSNA 2024 with Zero-Shot Transfer to SPIDER", OR
- (b) Frame as "multi-institution within RSNA 2024 (5 contributing institutions)" with per-institution F1 in supplementary

### C5. SPIDER Hybrid uses v2 checkpoint, not v3
**Flagged by:** R1 (Critical C3 expansion), Devil's Advocate (Major M4)

**Evidence** (RESULTS_LOG.md line 638-667, 729):
- v3 RSNA Hybrid retrain for SPIDER REGRESSED to Mean F1 = 0.519
- Decision (2026-05-04, user approved): "Use v2 hybrid_spider_unfreeze.slim.pth ckpt"
- Table II Hybrid row uses: `checkpoints/spider_phase4/best_model_hybrid_spider_unfreeze.slim.pth` (v2)
- Table I Hybrid row uses: `checkpoints/v3_20260503/hybrid/best_model_hybrid.pth` (v3)

**These are different trained models.** Paper does not disclose this.

**Action required:** Disclose in Methods. Either re-run v3 Hybrid-SPIDER with multiple seeds (recover reproducibility) OR explicitly state "SPIDER zero-shot evaluation uses checkpoint X trained under config Y; RSNA evaluation uses checkpoint Z trained under config W."

---

## MAJOR ISSUES (Tier 2)

### M1. Missing SOTA comparison table
**Flagged by:** EIC (issue #2 to flag), R2 (Critical C1), Devil's Advocate (Major M5)

Authors' own `SOTA_DEEP_RESEARCH.md` lists 8+ direct comparators on RSNA 2024, none cited:
- **M-SCAN** (Hong 2025, arXiv:2503.01634) — AUROC 0.971 canal
- **Aktan/Khan 2025** (IJCIS) — F1 0.957 (only peer-reviewed RSNA 2024 method paper)
- **Phaphuangwittayakul 2026** (Diseases) — F1 0.956 with external Phayao validation
- **nshen7 (GitHub 2024)** — Severe F1 0.501 on public split (most directly comparable; paper's 0.343 is 16pp behind)
- **Kaggle top-3** (Ian Pan, Bartley, Yuji, SonySpine) — best-documented SOTA architectures
- **Lin et al. 2024** — already cited (Bioengineering, balanced subset binary)

**Action required:** Add SOTA Table 1 with 5-7 comparators + footnotes about subset/binarization mismatches. Without this, "first BiomedCLIP on RSNA 2024" claim cannot be evaluated.

### M2. Citation issues
**Flagged by:** EIC, R1 (minor m5), R2 (Major M6, Minor), R3 (Minor m2), Devil's Advocate

Specific items to fix:
- L687 `datascarcity2024`: "L. Doe et al." placeholder name → replace with real authors
- L137 (Sec II): "McSweeney~et~al." named in prose but no `\cite{}` and no bibliography entry
- SPIDER paper `van der Graaf et al. *Sci. Data* 2024` not cited — used dataset without citing source
- Pfirrmann 2001 original grading paper not cited — using 5-class scale without citing definition
- Author affiliation block placeholder: "University Name, City, Country" + `@example.edu`
- Reference [12] (Decipher-MR npj Digital Medicine 2026), [13] (Merlin Nature 2026) — verify these are real publications, given current date is 2026-05-14
- Bibliography missing DOIs (EMBC may require)

### M3. Nigru citation error: F1 vs Balanced Accuracy
**Flagged by:** R2 (Critical C2, central issue)

**Issue:** Paper invokes "Nigru et al. F1 ≈ 0.84 balanced accuracy" in two locations:
- §II L138-141: "Nigru reports foraminal F1 ≈ 0.84"
- §V-C L523: "literature ceiling for sagittal-only foraminal grading... (≈0.84 balanced accuracy)"

**Reality** (per authors' own `research_deep_5_papers.md`):
- Nigru foraminal F1 = 0.838 (left) / 0.841 (right) ✓ "0.84 F1"
- Nigru foraminal **Balanced Accuracy = 0.691-0.702** (NOT 0.84)
- Nigru F1 is on a binary task with ~95% prevalence of normal — not directly comparable to authors' 3-class macro-F1

**Action required:**
1. Fix §V-C wording: 0.84 is F1 not balanced accuracy
2. Add binary-vs-3-class disclosure
3. Real defense: balanced-accuracy literature ceiling for foraminal is 0.69-0.70, not 0.84

### M4. Paper length exceeds EMBC 4-page limit
**Flagged by:** EIC (most important structural issue)

Current: ~6 pages IEEEtran. EMBC hard limit: 4 pages including references.

Compression strategy:
- Drop CBAM equations (1)-(3) — well-known module, cite Woo 2018 instead
- Drop AUC/AUPRC equations (5)-(7) — standard metrics
- Merge Sections IV (Experiments) and V (Results)
- Drop one of Figures 3/4/5 (likely CM, duplicates Table I info)

### M5. CBAM transfer hurts SPIDER — anti-evidence not discussed
**Flagged by:** R2, R3, Devil's Advocate (Major M3)

**Evidence** (RESULTS_LOG.md Run 6):
- SPIDER Baseline F1 = 0.606
- SPIDER CBAM F1 = 0.577 (**lower than baseline**)
- v2 and v3 both show CBAM hurts SPIDER transfer
- Author's own log: "CBAM transfer hurts on SPIDER"

This contradicts the paper's claim that CBAM adds task-generalizable structural priors. Paper currently silent on this.

**Action required:** Add Discussion paragraph honestly addressing "Why does CBAM hurt SPIDER transfer?" Possible answer: CBAM attention prior over-specializes to RSNA stenosis label space → distributional drift to SPIDER's disc-pathology labels.

### M6. "Selective transfer" framing as post-hoc rationalization
**Flagged by:** R2 (Major M7), R3 (Critical C3), Devil's Advocate (Major M2)

**Issue:** Paper groups SPIDER labels into "disc-related (transfer strongly)" vs "non-disc (semantic gap)". But:
- Modic F1=0.117, Pfirrmann F1=0.147, Spondylolisthesis F1=0.030 (all near or below random)
- Mean F1 across all 8 labels = 0.362 vs Naked BiomedCLIP ≈ 0.398 → **Hybrid loses on mean!**

The grouping appears post-hoc to favor Hybrid wins.

**Reframe as scientific finding:** "RSNA-supervised fine-tuning of a frozen-BMC fusion architecture causes representational drift that helps for in-distribution disc tasks and hurts for out-of-distribution non-disc tasks." This is a generalizable lesson for medical-VLM-adaptation, not a face-saving narrative.

Alternative explanation (R2): Modic is endplate signal (T1+T2 needed, paper uses T2-only); spondylolisthesis is vertebral alignment (not visible in per-IVD crop). The issue is input-modality mismatch, not semantic distance.

### M7. RSNA 2024 sequence content factually wrong
**Flagged by:** R2 (Critical C3)

**Paper §VI-B claim:** "foraminal stenosis is best assessed on axial or coronal reformats, not provided in RSNA 2024"

**Reality:** RSNA 2024 includes Sagittal T1, Sagittal T2/STIR, AND **Axial T2**. Kaggle top teams use axial T2 for subarticular stenosis. The "axial not provided" claim is incorrect.

**Real defense:** "We elected sagittal-only input to retain compatibility with SpineNetV2 weights" — methodological choice, not dataset constraint.

### M8. Logit scale $s$ frozen vs learnable ambiguity
**Flagged by:** R3 (Critical C1)

§III-B Eq. 9: "$s$ is a learned temperature inherited from BiomedCLIP pretraining" — ambiguous. Is $s$:
- Frozen at BiomedCLIP pretrained value (~ln 100 = 4.6)?
- Updated during RSNA training?
- Re-initialized?

If updated during RSNA, calibration is tuned for RSNA prompts, then applied to SPIDER prompts at zero-shot time. Can silently destroy cosine-similarity ranking.

**Action required:** State unambiguously + sensitivity check (SPIDER F1 at frozen $s$ vs trained $s$). 1-day experiment.

### M9. Class-imbalance pipeline not decomposed
**Flagged by:** R1 (Major M1)

4 components applied together (focal + sqrt class weights + oversampling 3-5x + medical augmentation). No isolated ablation. "Kitchen sink" risk.

**Action required:** Mini-ablation on Hybrid: (i) no rebalancing, (ii) +focal, (iii) +focal +class weights, (iv) full pipeline. Or cite existing internal comparisons (RESULTS_LOG.md focal-γ=1.8 vs 2.0 in v2 vs v3 as unintended ablation).

### M10. Clinical pathway not specified
**Flagged by:** R2 (Major M1), R3 (Major M5)

Paper says "clinically appropriate trade-off in screening" but doesn't commit:
- **Screening** (asymptomatic): 49.6% Recall with ~26% Precision → not clinically useful
- **Triage / worklist prioritization**: 49.6% sensitivity insufficient (misses half severes)
- **Second-reader / QA**: defensible if radiologist remains primary

**Action required:** Add 1 paragraph specifying clinical pathway. Anchor to FDA/CE-aligned framings (Char NEJM 2018; Kelly BMC Med 2019). Honest framing: "research-cohort retrospective querying or radiologist-in-loop triage, NOT autonomous deployment."

---

## MINOR ISSUES

- m1. Severe AUPRC "6.5×" baseline framing — actual ratio 7.6× using exact Severe prevalence 0.042 (not 0.05). Direction of rounding should be stated.
- m2. AUC=NaN cells for Pfirrmann/Modic in Table II silently omitted. Footnote: "multi-class AUC not reported due to one-vs-rest formulation."
- m3. Trainable parameter discrepancy: 1.18M trainable + 218M total. Does CBAM-3D backbone (~63M) count as trainable or frozen? Provide parameter breakdown table.
- m4. Mean F1 macro definition: two-stage averaging (across classes, then across conditions) not stated.
- m5. Code release plan absent. Add anonymized repo URL for review.
- m6. "approximately \$2 per Hybrid run" (L322) — Vast.ai pricing not scientifically relevant; move to footnote or remove.
- m7. Inference latency 9.65ms / 103 sa/s framed as "well above clinical real-time" without numeric clinical threshold reference.
- m8. Grad-CAM shown on single case only. 2-3 cases (TP + FN) would strengthen.
- m9. §V-C confusion matrix: "70.4% recall on canal stenosis" — specify this is Severe class. Also confusion: matrix is for CBAM-only, not Hybrid (which is the headline model).
- m10. §V-C PR curves: "moderate-recall regime (Recall ∈ [0.2, 0.6])" — Recall=0.2 not clinically usable. Clinical regime is Recall ≥ 0.8.
- m11. Foraminal Severe F1 averaged with canal (0.500 dominates 0.278/0.252) inflates headline Severe F1 0.343. Per-condition disclosure recommended.

---

## REVISION ROADMAP (PRIORITIZED)

### 🔴 P0 — Must-fix before submission (~5 days local + $15 GPU)

1. **Fix numerical errors** (1 buổi)
   - Abstract: 11.9%→11.5%, 49.6%→46.4%, ×4.2→×4.0
   - Conclusion + §V.A consistent
   - Recompute all Table I cells from raw `*_best_metrics.txt`
2. **Fix citations** (½ ngày)
   - Replace `L. Doe et al.` with real citation
   - Add SPIDER paper (van der Graaf 2024)
   - Add `mcsweeney2023` or remove name
   - Verify Decipher-MR + Merlin publication dates
   - Add Pfirrmann 2001
3. **Fill author affiliation** (5 phút)
4. **Fix Nigru F1 vs balanced accuracy** in §V-C + §VI-B (½ ngày)
5. **Fix RSNA 2024 "axial not provided" error** in §VI-B (15 phút)
6. **Decide title**: drop "Multi-Center" or add multi-institution eval (decision needed)
7. **Cut paper 6p → 4p** for EMBC compliance (1-2 ngày)
8. **Add SOTA Table 1** with 5-7 comparators (½ ngày)

### 🟡 P1 — Strongly recommended (~$15 GPU + 2 ngày local)

9. **3-seed mean±std** for 4 RSNA configs ({0, 7, 42}). Cost ~$15. Replace single-seed numbers in Table I.
10. **Linear-probe naked BiomedCLIP baseline** on RSNA (~$2). True frozen-VLM upper bound.
11. **Disclose v2 checkpoint substitution** for SPIDER Hybrid in Methods.
12. **Clarify logit scale $s$** + 1-day sensitivity check.
13. **Reframe selective transfer** as scientific finding (generalizable lesson).
14. **Clinical pathway paragraph** (research-cohort + radiologist-in-loop, NOT autonomous).
15. **Discuss CBAM hurts SPIDER** honestly in Discussion.

### 🟢 P2 — Nice-to-have (~$5 GPU + 1 ngày)

16. **BiomedCoOp prompt tuning** ablation (~$2). Recover Pfirrmann/Modic zero-shot gap.
17. **Per-condition Severe F1 row** in Table I.
18. **Bootstrap CI** on existing predictions (30 sec, no GPU).
19. **Class-imbalance pipeline decomposition** (4-row mini-ablation).
20. **Mendeley external validation** (currently future work).
21. **DeLong + McNemar tests** for statistical significance.

---

## DEVIL'S ADVOCATE STRONGEST COUNTER-ARGUMENT

> "The headline gains do not survive scrutiny because the experimental design cannot distinguish signal from seed-to-seed noise, and the most-cited 'wins' are arithmetically inconsistent with the underlying logs. A peer reviewer will reach this conclusion in under five minutes."

**Specifically:**
- Severe Recall 11.9%→49.6% (paper) vs 11.5%→46.4% (logs): numbers don't appear in any raw artifact
- Single seed; +0.026 F1 fusion gain is ¼ of documented internal seed swing
- BMC-only converged at epoch 3 (degenerate baseline)
- Title "Multi-Center" — training is single-center

---

## ACTION DECISION POINTS (USER)

The following decisions require user input before fixes proceed:

1. **Title**: Keep "Multi-Center MRI" + add real multi-center eval? OR change title to "RSNA 2024 with Zero-Shot Transfer to SPIDER"?
2. **3-seed run timing**: Run tomorrow alongside SPIDER 8-label (~9h total, $20)? Or defer P1 to next week?
3. **Linear-probe BiomedCLIP**: Run alongside other GPU work tomorrow?
4. **BiomedCoOp**: Run as Tier 2 priority or defer?
5. **Length cut**: Reduce 6p → 4p for EMBC submission, OR keep 6p version for LVTN + create separate 4p EMBC version?

---

## FILES REFERENCED FOR VERIFICATION

- **Paper**: `paper/REPORT_PAPER.tex` (716 lines, 6 pages PDF)
- **Vietnamese detailed**: `paper/REPORT_DETAILED_VI.tex` (31 pages PDF)
- **Raw RSNA results**: `experiments/v3_20260503/RESULTS_LOG.md` (796 lines)
- **Hybrid metrics**: `~/Desktop/v3_check/checkpoints/v3_20260503/hybrid/hybrid_best_metrics.txt`
- **BMC-only metrics**: `checkpoints/v3_20260503/bmc_only/hybrid_biomedclip_only_best_metrics.txt`
- **SOTA research**: `paper/SOTA_DEEP_RESEARCH.md`
- **VLM survey**: `paper/research_foundation_models_spine.md`
- **Methods survey**: `paper/research_methods_survey.md`

---

*End of Review Report. Saved to `paper/REVIEW_REPORT.md` for future reference.*
