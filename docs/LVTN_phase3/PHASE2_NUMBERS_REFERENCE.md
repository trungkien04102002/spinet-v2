# Phase 2 numbers, in full, for answering the advisor

One place to look up what the model scored before Phase 3, broken down by
condition, by grade, and by class. Sources are named on every block so any
figure can be traced back.

---

## Read this first: two number sets exist, and one table is wrong

**Two sets, never to be mixed inside one table or one talk.**

| Set | Where it lives | Seeds |
| --- | --- | --- |
| 3-seed | MIWAI paper Table 1, `checkpoints/v3_20260503/*_best_metrics.json` | 42, 123, 456 |
| single-seed | thesis `chapter/chap05.tex` | 42 only |

**An error in the thesis summary table.** `chap05.tex` Table `tab:rsna_ablation`
reports Mean F1 macro as 0.343 for the baseline and 0.474 for the hybrid, and
the abstract repeats it as a +0.131 improvement. Recomputing the macro from the
per-class F1 in `tab:rsna_perclass` on the same page gives:

    baseline  (0.898 + 0.211 + 0.152) / 3 = 0.4203
    hybrid    (0.786 + 0.406 + 0.333) / 3 = 0.5083

The recomputed baseline agrees with the 3-seed paper figure of 0.4202 to four
decimals, so 0.4203 is the right number and 0.343 is not. 0.343 is the hybrid's
Severe F1, two rows further down the same table: the wrong cell was copied.

So the real improvement is **+0.088 single-seed** or **+0.106 over three
seeds**, not +0.131. If the advisor asks about the headline improvement, quote
0.4202 to 0.5265 from the paper and say the thesis table is being corrected.
Nothing else in either table is affected; only that one row.

---

## 1. Four configurations, three seeds (MIWAI paper Table 1)

Use this set by default. It is published, reviewed, and the safest to quote.
"SpineNetV2" here means its grading backbone retrained by us, not the published
pipeline.

| Metric | SpineNetV2 | CBAM-only | Multimodal-only | Hybrid (Ours) |
| --- | --- | --- | --- | --- |
| Mean Accuracy (%) | **81.0** | 69.5 | 71.2 | 72.4 |
| Mean F1 macro | 0.420 | 0.477 | 0.482 | **0.527** |
| Mean Recall | 0.412 | 0.531 | 0.532 | **0.592** |
| Mean Precision | 0.493 | 0.499 | 0.508 | **0.526** |
| Mean AUC | 0.835 | 0.803 | 0.826 | **0.838** |
| Mean AUPRC | **0.530** | 0.498 | 0.513 | 0.533 |
| **Severe Recall** | 0.123 | 0.288 | 0.316 | **0.486** |
| **Severe F1** | 0.152 | 0.262 | 0.268 | **0.356** |
| **Severe AUC** | 0.872 | 0.865 | 0.885 | **0.899** |
| **Severe AUPRC** | 0.284 | 0.278 | 0.275 | **0.333** |

Headline for the advisor: Severe recall goes from 12.3% to 48.6%, about four
times, and Severe F1 from 0.152 to 0.356. Mean accuracy drops nine points
because the baseline answers "Normal" almost every time; that trade is the
point, not a side effect.

---

## 2. Per condition and per grade, single-seed (thesis Table `tab:rsna_per_cond_acc`)

This is the breakdown the advisor is most likely to ask for. Baseline to Hybrid.

| Condition | Grade | Recall | Precision | F1 |
| --- | --- | --- | --- | --- |
| Spinal canal | Normal/Mild | 98.6 to 92.8 | 91.1 to 96.4 | 0.947 to 0.946 |
| Spinal canal | Moderate | 7.9 to 42.4 | 33.3 to 32.4 | 0.128 to 0.368 |
| Spinal canal | **Severe** | 35.8 to **70.4** | 63.0 to 55.9 | 0.457 to **0.623** |
| Left foraminal | Normal/Mild | 96.2 to 54.5 | 80.4 to 93.4 | 0.876 to 0.688 |
| Left foraminal | Moderate | 18.6 to 76.4 | 46.2 to 28.2 | 0.265 to 0.412 |
| Left foraminal | **Severe** | **0.0** to 18.8 | **0.0** to 17.2 | **0.000** to 0.180 |
| Right foraminal | Normal/Mild | 96.2 to 58.6 | 79.5 to 94.1 | 0.871 to 0.723 |
| Right foraminal | Moderate | 16.7 to 75.8 | 43.4 to 30.9 | 0.241 to 0.439 |
| Right foraminal | **Severe** | **0.0** to 21.7 | **0.0** to 18.0 | **0.000** to 0.197 |

Accuracy per condition: spinal canal 89.5 to 88.3, left foraminal 77.8 to 57.1,
right foraminal 76.8 to 60.4.

**The single most quotable fact in this table.** On both foraminal conditions
the baseline scores Severe recall of exactly 0.0: it never once predicted
Severe. So the improvement there is not a percentage gain, it is the difference
between a model that cannot do the task at all and one that can partly do it.

**And the weakness to admit before being asked.** Foraminal Severe F1 reaches
only 0.180 and 0.197 against 0.623 on the canal. That gap is what the whole of
Phase 3 went after, and it is why the T1 routing work exists.

---

## 3. Per condition and per grade, AUC and AUPRC (thesis Table `tab:rsna_per_cond_auc`)

Threshold-free, so these say whether the representation improved rather than
whether the decision rule did.

| Condition | Grade | AUC base to hybrid | AUPRC base to hybrid |
| --- | --- | --- | --- |
| Spinal canal | Normal/Mild | 0.892 to 0.947 | 0.984 to 0.993 |
| Spinal canal | Moderate | 0.851 to 0.884 | 0.272 to 0.319 |
| Spinal canal | **Severe** | 0.922 to **0.963** | 0.513 to **0.602** |
| Left foraminal | Normal/Mild | 0.794 to 0.789 | 0.932 to 0.924 |
| Left foraminal | Moderate | 0.758 to 0.655 | 0.387 to 0.262 |
| Left foraminal | **Severe** | 0.815 to **0.858** | 0.147 to **0.174** |
| Right foraminal | Normal/Mild | 0.802 to 0.807 | 0.929 to 0.931 |
| Right foraminal | Moderate | 0.760 to 0.657 | 0.375 to 0.283 |
| Right foraminal | **Severe** | 0.842 to 0.837 | 0.167 to **0.181** |
| **Severe, mean of three** | | 0.860 to **0.886** | 0.276 to **0.319** |

**Expect a question about the Moderate row.** Foraminal Moderate AUC falls by
0.103 on both sides. The cause is known and recorded in the thesis: the
horizontal-flip augmentation did not swap the left and right labels, so
Moderate foraminal labels were being corrupted. Fixed in the code release.
Severe was less affected because those cases sit centrally and are less
sensitive to that particular label noise.

---

## 4. Per class, averaged over the three conditions (thesis Table `tab:rsna_perclass`)

| Class | Recall | Precision | F1 | AUC | AUPRC |
| --- | --- | --- | --- | --- | --- |
| Normal/Mild | 97.0% to 68.6% | 83.7% to 94.6% | 0.898 to 0.786 | 0.829 to 0.848 | 0.948 to 0.949 |
| Moderate | 14.4% to 64.9% | 41.0% to 30.5% | 0.211 to 0.406 | 0.790 to 0.732 | 0.345 to 0.288 |
| **Severe** | 11.9% to **37.0%** | 21.0% to **30.4%** | 0.152 to **0.333** | 0.860 to **0.886** | 0.276 to **0.319** |

Severe improves on all five metrics at once, with nothing traded away inside
that class. The cost is paid by Normal/Mild recall, which is the intended
direction: a missed Severe case is worse than a false alarm.

---

## 5. What Phase 3 changed against this baseline

| | Phase 2 | Phase 3 | Change |
| --- | --- | --- | --- |
| Severe AUPRC | 0.3211 | 0.3696 | +0.0485 |
| Severe F1 | 0.3434 | 0.3696 | +0.0262 |
| Severe Recall | 0.4640 | 0.4889 | +0.0249 |
| Severe Precision | 0.2732 | 0.3014 | +0.0282 |
| Mean Accuracy | 0.7208 | 0.7318 | +0.0110 |
| Mean F1 macro | 0.5277 | 0.5286 | +0.0009 |
| Macro AUC | 0.8371 | 0.8368 | -0.0003 |

**Say what this is honestly.** It is a correct initialisation, not a new idea:
the earlier run had started from a checkpoint with the wrong class-weight mode.
It beats Phase 2 on nine of ten metrics and no epoch of the twenty fell below
Phase 2's published Severe AUPRC peak, which makes it a clean baseline to
measure Phase 3 work against rather than a contribution in itself.

**The T1 routing result is under revision.** The report currently gives macro
AUPRC +0.054 left and +0.039 right for the T1 foraminal branch. Those two
numbers compare each branch on its own validation set, and the two sets share
only 214 of 395 patients. Paired on identical rows the left gain is +0.035 and
the right is -0.007, and the assembled routed system sits 0.020 below T2-only
on Mean F1 macro with p=0.122. See `experiments/routing/assemble_routed_system.py`.

---

## Sources

| Block | File |
| --- | --- |
| 3-seed table | `paper/camera_ready/main.tex` Table 1; `experiments/v3_rsna_multiseed/SOURCE_OF_TRUTH.md` |
| per condition and grade | `paper/lvtn_overleaf/chapter/chap05.tex`, `tab:rsna_per_cond_acc` and `tab:rsna_per_cond_auc` |
| per class | same file, `tab:rsna_perclass` |
| Phase 3 comparison | `docs/LVTN_phase3/PHASE3_CHANGES_AND_RESULTS.md` |
| routed system | `experiments/routing/results/routed_system.json` |
