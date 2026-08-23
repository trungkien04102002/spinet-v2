# Phase 3: what changed on top of Phase 2, and what it bought

Report basis for the advisor meeting. Written 2026-08-23.

Everything here is seed 42 on the same patient-level validation split, so the
numbers are directly comparable. Where a Phase 2 published figure came from a
3-seed mean instead, that is stated explicitly: **do not mix the two sets in one
table.**

---

## 1. Where Phase 2 stood

Reference artifact: `checkpoints/v3_20260503/hybrid/hybrid_best_metrics.json`
(seed 42, best epoch 10). Recipe: frozen CBAM 3D-ResNet34 + frozen BiomedCLIP
ViT-B/16, slice-attention pool, concat_mlp fusion, L2-normalised cosine logits
against frozen text anchors; focal (gamma 2.0) + SupCon (0.1) +
UncertaintyLoss, sqrt class weights, oversample x3, augmentation medium.

Validation split: 395 patients, 1942 IVD crops.

| level | metric | value |
|---|---|---|
| overall | Mean F1 macro | 0.5277 |
| overall | Mean Accuracy | 0.7208 |
| overall | Severe F1 (mean of 3) | 0.3434 |
| overall | Severe AUPRC | 0.3211 |
| overall | Macro AUC | 0.8371 |
| overall | Macro AUPRC | 0.5258 |
| spinal canal | AUC / AUPRC | 0.9270 / 0.6284 |
| spinal canal | Severe F1 / Recall / Precision | 0.5000 / 0.7037 / 0.3878 |
| left foraminal | AUC / AUPRC | 0.7880 / 0.4670 |
| left foraminal | Severe F1 / Recall / Precision | 0.2778 / 0.3750 / 0.2206 |
| right foraminal | AUC / AUPRC | 0.7964 / 0.4820 |
| right foraminal | Severe F1 / Recall / Precision | 0.2524 / 0.3133 / 0.2114 |

Class support in validation: canal 1722/139/81, left foraminal 1497/360/80,
right foraminal 1482/372/83 (Normal-Mild / Moderate / Severe).

**The weak point is unmistakable**: canal AUPRC 0.6284 against foraminal 0.4670
and 0.4820. Phase 3 targets that gap.

---

## 2. What Phase 3 changed

### 2.1 Diagnosis: the foraminal crops were wrong (evidence, not opinion)

Three findings, each verified against our own data rather than taken from a
paper:

**The wrong image series.** Joining `train_label_coordinates.csv` to
`train_series_descriptions.csv` shows foraminal findings are annotated **100% on
Sagittal T1** (9860/9859 rows), canal findings **99.9% on Sagittal T2** (9748),
and subarticular **100% on Axial T2**. Canal and foraminal share a series once
in 6291 studies. Our pipeline (`rsna_dataloader.py:255-310`) cut **one** crop per
level from **T2, centred on the canal coordinate**, then attached all three
labels to it. Two of the three labels were therefore read off an image series
the radiologist never used for them.

**The wrong location.** Left and right foraminal annotations sit a **median of 7
slices apart** (n=9824; 22.8% more than 8 slices apart), while the crop window is
only 9 slices. One shared window cannot cover both sides.

**Independent corroboration.** The 1st-place RSNA 2024 solution lists "sagt2
image for nfn" under *what did not work*. No top-10 team used a shared per-level
crop as the primary path for foraminal. brendanartley's `infer.py:244` deletes
the shared-crop foraminal output whenever a Sag-T1 series exists.

### 2.2 Fix: an augmentation bug that had been corrupting laterality

`RandomHorizontalFlip` flipped `dims=[-1]`, which for a `(9, 112, 224)` crop is
the **anterior-posterior** axis, not left-right. Swapping `left_*`/`right_*`
labels on that flip therefore mislabelled the data. The correct laterality
augmentation flips the slice axis (`dims=[0]`).

Changes in `spinenet/augmentation.py`:
- `RandomHorizontalFlip` now refuses `swap_labels=True` unless
  `allow_wrong_axis_swap=True` is passed, with an error explaining the axis.
- New `RandomSliceReverse` performs the correct left-right flip.
- `assert_slice_reverse_is_safe()` blocks slice-reverse on datasets where a crop
  carries only one side's label. Measured: T1 is 100.0% one-sided (blocked), T2
  is 0.3% (allowed).
- 20 axis assertions in `experiments/f1_improvement/test_augmentation_axes.py`,
  which encode each voxel with its own `(slice, row, col)` so any axis
  permutation is detectable. All pass.

### 2.3 Fix: the published run had been initialised from the wrong checkpoint

Two Phase 3 runs on 2026-08-22 used `checkpoints/rsna/best_model_attention.pth`,
whose `args.class_weight_mode` is `None`. The published recipe needs the sqrt
variant. The surviving correct init is
`checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth`
(md5 `511fcc8796c43805ea100ed2c2d9c55b`, `class_weight_mode='sqrt'`,
`oversample_factor=5`, epoch 19).

Re-running the published recipe on the correct init is **run (c)**. This is a
correction, not a contribution: its value is that later claims cannot be
dismissed with "your baseline was broken".

### 2.4 New capability: per-side T1 crops and a foraminal specialist

`experiments/f1_improvement/prep_t1_crops.py` produces one crop per
`(study, level, side)` from Sagittal T1, centred on that side's own annotation:
19689 crops over 1972 studies (17 GB).

`train_rsna_hybrid.py` gained flags so one script covers both branches:
`--split` (metadata stem), `--conditions` (which heads to train),
`--select-by {severe_f1, severe_auprc, macro_auprc, weighted_logloss}`,
`--fast-dev`, `--no-ap-flip`, `--slice-reverse`. Six hardcoded
three-condition lists were removed, and `--hflip-swap-labels` now defaults to
False.

The Hybrid model needed no change: it is condition-agnostic (cosine against text
embeddings, no per-condition heads), so a 2-condition run reuses it directly.

### 2.5 New: an evaluation framework that can tell signal from noise

Phase 2 reported F1 alone. With 80 Severe positives per condition, F1 at n=80
has SD ~0.047, and within a single 20-epoch run it ranged 0.121 (0.2685 to
0.3901). F1 alone cannot separate two models here.

- `robust_metrics.py`: AUPRC, QWK, RSNA weighted log loss, with 95% CIs from a
  **patient-clustered** bootstrap (Severe findings cluster within a patient, so a
  row-level bootstrap gives intervals that are too narrow). F1 demoted to
  secondary.
- `compare_models.py`: paired patient-clustered bootstrap, refusing to run
  unless both dumps cover identical rows.
- `temperature_calibration.py`: temperature cross-fitted over patient halves.
- `threshold_sweep.py`: per-class decision weights by coordinate ascent, plus
  logit adjustment and tau-normalisation as comparators.
- `dump_logits.py`: gained `--split` and `--conditions` so a 2-condition T1
  checkpoint can be dumped at all; results filename now carries the split and
  conditions, because a T1 and a T2 sweep were both writing
  `hybrid_seed42_threshold_results.json` and silently overwriting each other.

Measured epoch-to-epoch stability within one run: Severe AUPRC SD 0.0048 against
Severe F1 SD 0.0286. **AUPRC is roughly six times more stable, which is why it is
the primary metric.**

---

## 3. Results

### 3.1 Run (c): published recipe on the correct init

Same recipe as Phase 2, correct sqrt init, seed 42, same split.

| metric | Phase 2 | run (c) best (ep5) | delta |
|---|---|---|---|
| Severe AUPRC | 0.3211 | **0.3696** | **+0.0485** |
| Severe F1 | 0.3434 | 0.3696 | +0.0262 |
| Severe Recall | 0.4640 | 0.4889 | +0.0249 |
| Severe Precision | 0.2732 | 0.3014 | +0.0282 |
| Mean Accuracy | 0.7208 | 0.7318 | +0.0110 |
| Mean F1 macro | 0.5277 | 0.5286 | +0.0009 |
| Macro AUC | 0.8371 | 0.8368 | -0.0003 |

No epoch of the 20 fell below Phase 2's published Severe AUPRC peak of 0.3211.
Plateau (last 5 epochs) Severe AUPRC 0.3543.

### 3.2 Gated fusion: a null result, and worth reporting as one

`--fusion gated` on the same correct init, seed 42. Early-stopped at epoch 18
(best epoch 3).

| metric | run (c) concat | gated | delta |
|---|---|---|---|
| Severe AUPRC peak | 0.3696 | 0.3680 | -0.0016 |
| Severe AUPRC plateau | 0.3543 | 0.3619 | +0.0076 |
| Severe F1 peak | 0.3901 | 0.3696 | -0.0205 |
| Severe F1 plateau | 0.3118 | 0.3078 | -0.0040 |

The peak difference (0.0016) is a third of the epoch-to-epoch SD (0.0048).
An earlier paired patient-clustered bootstrap on the same question gave Severe
F1 p = 0.125 and Severe AUPRC p = 0.152. **Gated is not better; the fusion
mechanism is not the bottleneck.** Report it as an ablation.

### 3.3 T1 foraminal specialist: the main result

Trained on `rsna_preprocessed_t1`, `--conditions left_foraminal
right_foraminal`, same init, seed 42, same 395 validation patients.
Train 15749 / val 3940 rows (per-side crops; the label set is unchanged, so
Severe support stays 80 left and 72 right).

Ran the full 20 epochs (no early stop). Selector `severe_auprc` saved
**epoch 14**.

| metric | Phase 2 | run (c) best of 20 | T1 best of 20 | vs Phase 2 |
|---|---|---|---|---|
| left foraminal AUPRC | 0.4670 | 0.4850 | **0.5321** (ep16) | **+0.065** |
| right foraminal AUPRC | 0.4820 | 0.4975 | **0.5311** (ep16) | **+0.049** |
| left foraminal AUC | 0.7880 | 0.8097 | 0.8281 (ep16) | +0.040 |
| right foraminal AUC | 0.7964 | 0.8104 | 0.8268 (ep16) | +0.030 |

**All 20 epochs exceeded run (c)'s best-of-20 AUPRC on both sides.** That
consistency matters more than the peak: it rules out a lucky epoch, which has
misled this project twice before. The margin widened over training (+0.027 at
epoch 10, +0.038 by epoch 15).

Severe AUPRC rose almost monotonically from epoch 9 to 14: 0.2467, 0.2558,
0.2589, 0.2639, 0.2665, 0.2705.

At the **saved epoch-14 checkpoint**, which is the honest number to quote since
it is what the selection rule actually picks:

| metric | left foraminal | right foraminal | Phase 2 left | Phase 2 right |
|---|---|---|---|---|
| Macro AUPRC | **0.5205** | **0.5209** | 0.4670 | 0.4820 |
| Severe AUPRC | 0.2538 | 0.2872 | - | - |
| QWK | 0.3624 | 0.3521 | 0.351 | 0.409 |
| Weighted log loss | 0.8380 | 0.8501 | - | - |
| Severe F1 | 0.2247 | 0.2902 | 0.2778 | 0.2524 |

Mean across the two sides, with patient-clustered 95% CIs (2000 draws, 395
patients):

| metric | value | 95% CI | CI width |
|---|---|---|---|
| Macro AUPRC | 0.5207 | [0.4952, 0.5539] | 0.059 |
| QWK | 0.3572 | [0.3264, 0.3863] | 0.060 |
| Weighted log loss | 0.8440 | [0.8251, 0.8630] | 0.038 |
| Severe AUPRC | 0.2705 | [0.2027, 0.3583] | 0.156 |
| Severe F1 | 0.2574 | [0.1919, 0.3222] | 0.130 |
| Severe AUC | 0.8852 | [0.8647, 0.9051] | 0.040 |

**AUPRC improves but QWK does not.** QWK goes 0.351 to 0.3624 on the left
(+0.011) and 0.409 to 0.3521 on the right (-0.057). QWK weights *how far* a
prediction is from the truth on the ordinal scale, so the T1 branch ranks
Severe cases better while not improving, and on the right side worsening, the
ordinal ordering as a whole. This must be reported, not omitted: it is the one
primary metric that does not support the story.

Temperature scaling, cross-fitted over patient halves, fits T = 0.72 and 0.72
(left) and 0.69 and 0.75 (right) - below 1, so the model is *under*-confident -
and improves weighted log loss from 0.8440 to 0.8308 (-0.0132).

### 3.4 Post-hoc threshold tuning

Coordinate ascent on per-class decision weights, cross-fitted over patient
halves, on the final epoch-14 T1 checkpoint:

| decision rule | macro F1 | Mean Acc | Severe F1 | CostErr (1:2:4) |
|---|---|---|---|---|
| argmax | 0.468 | 59.2% | 0.257 | 0.402 |
| coordinate ascent, held out | **0.509** | - | - | - |
| coordinate ascent, in-split (optimistic) | 0.529 | 75.8% | 0.270 | 0.327 |
| logit adjustment tau=0.5 | 0.283 | 30.0% | 0.236 | 0.616 |

Per side, held out: left 0.482, right 0.536.

**+0.041 macro F1 held out**, above the +0.019 precedent. Logit adjustment
collapses (mean accuracy 27%), consistent with the earlier finding.
Tau-normalisation is a mathematical no-op for this architecture: the cosine
prototypes are L2-normalised by construction, so every class already has
||w|| = 1.

This gain **stacks** with the T1 gain, because it converts better ranking into
better thresholded decisions rather than competing for the same headroom.

### 3.5 Backbone fine-tuning: a negative result that defends the design

Only 1.18M of 260M parameters (0.45%) were ever trainable. Since every
intervention downstream of the frozen features moved the ranking metrics by
roughly nothing, while the one change to the *input* moved AUPRC by +0.054, the
representation was the obvious suspect. `--unfreeze-cbam layer4` trains the last
residual stage (39M parameters, 15.4% of the model) at a 10x smaller learning
rate, with BatchNorm statistics held fixed.

Stopped at epoch 10 of 20. Against run (c) at the same epoch:

| metric | run (c) | unfreeze layer4 | delta |
|---|---|---|---|
| train loss | 0.9280 | 0.8306 | -0.097 |
| **val loss** | **0.1421** | 0.1453 | **+0.003** |
| Mean Accuracy | 0.7222 | **0.7507** | +0.029 |
| avg Severe F1 | **0.3850** | 0.3628 | -0.022 |
| **Severe AUPRC** | **0.3598** | 0.3230 | **-0.037** |
| Macro AUPRC | 0.5326 | 0.5305 | -0.002 |
| canal Severe F1 | 0.5586 | **0.6369** | +0.078 |

Train loss down, validation loss up: textbook overfitting, with 39M parameters
against 7806 training crops. The important detail is *where* it hurts. Mean
accuracy **rises** 0.029 while Severe AUPRC **falls** 0.037, and Macro AUPRC
barely moves. Overfitting does not spread evenly; it eats the rare class first,
because the rare class has only ~80 examples to memorise. That is precisely the
pathology this whole project exists to fix.

**Report this as a defence of the frozen-backbone design**, not as a failure.
"Why didn't you fine-tune the encoder?" now has a measured answer instead of an
assumption. If pursued later, the direction is to unfreeze *less* (a single
residual block) or to stop at epoch 3-5, not to raise the learning rate.

---

## 4. Honest limitations to state before being asked

**The routed system has not been evaluated end to end.** The canal numbers come
from a T2 three-condition model and the foraminal numbers from the T1
specialist. Combining them on paper is not the same as running one system.
Assembling it needs no retraining, only joining two prediction dumps on
`(study, level)`, but it has not been done yet, so there is currently no Mean F1
macro or Mean Accuracy for the routed system.

**T1 and run (c) were not evaluated on the same patients.** This is the most
serious caveat here and it was only found on 2026-08-23.

`train_test_split` ran on whatever patient list each metadata file contained.
`rsna_preprocessed` has 1973 studies, `rsna_preprocessed_t1` has 1972. A
three-study difference does not perturb the partition, it replaces it: the two
validation sets share **214 of 395 patients**, and **181 patients sit in one
run's validation set while being in the other run's training set**.

What this does *not* invalidate: either model's own numbers. Each was scored on
patients it never trained on.

What it does invalidate: the direct comparison between them, which is measured
on two different samples rather than paired, so its uncertainty is wider than
quoted. And it blocks the routed system entirely, because for those 181
patients the canal branch would be predicting on a case it memorised. An honest
routed evaluation would be restricted to the 214 shared patients, where Severe
support drops to 40-48 per condition from 80-83 and the SD of Severe F1 rises
from about 0.047 to about 0.061.

`--split-mode hash` (added the same day) fixes the cause: a patient's fold
becomes a property of its id rather than of its position in a list, so any two
datasets agree. Measured on this pair: 406 validation patients, identical in
both, zero cross-contamination. **Everything reported above predates that flag
and used the old behaviour.** Re-running run (c) and the T1 specialist under
`--split-mode hash` (about 2.5 GPU-hours) is what would make the headline
comparison paired and testable.

Note that `compare_models.py` refusing to run on these two dumps was correct,
and for a reason more fundamental than the row-count mismatch it reports.

**Single seed.** Every Phase 3 number here is seed 42. The thesis table needs 3
seeds.

**AUC improves far less than AUPRC** (+0.03 to +0.04 against +0.05 to +0.065).
The gain is concentrated where the model is confident, not across the whole
ranking. For screening that is the useful region, but it should be described
accurately rather than as a uniform improvement.

**Checkpoint selection uses `--select-by severe_auprc`.** Epoch 16 was the best
epoch on macro AUPRC and AUC, while the selector saved epoch 14. Whatever
criterion is chosen must be applied identically to every configuration being
compared, or the comparison becomes "best epoch for each", which a committee
will challenge.

**Training labels are single-annotator with no adjudication** (confirmed by the
LumbarDISC dataset paper, arXiv:2506.09162). Human inter-reader agreement is
kappa 0.73 for canal and 0.58 for foraminal (Lurie et al., *Spine*, 2008). Our
QWK of 0.641 on canal approaches human agreement; foraminal at 0.351/0.409 does
not, which corroborates the crop diagnosis from an independent direction.

---

## 5. A correction owed to the thesis document

`paper/lvtn_overleaf/main.tex:162` currently reads:

> "Hybrid cải thiện Mean F1 macro từ 0.343 lên 0.474 (+0.131), Severe Recall
> tăng từ 11.5% lên 46.4%"

Recomputed from the six Phase 2 metrics files (now in
`experiments/f1_improvement/results_20260823/v3_20260503_phase2/`):

| | Baseline | Hybrid | delta |
|---|---|---|---|
| 3-seed Mean F1 macro | 0.4202 | 0.5265 | +0.106 |
| 3-seed avg Severe F1 | 0.1518 | 0.3556 | +0.204 |
| 3-seed Severe Recall | 12.25% | 48.62% | x4.0 |
| 3-seed Mean Accuracy | 0.8103 | 0.7237 | -0.087 |
| seed 42 Mean F1 macro | 0.4288 | 0.5277 | +0.099 |
| seed 42 Severe Recall | 11.52% | 46.40% | x4.0 |

The sentence is wrong twice over. **0.343 is seed 42's avg Severe F1** (0.3434)
labelled as Mean F1 macro, and **0.474 matches nothing** in any of the six files
-- not baseline, not hybrid, no seed, not the 3-seed mean. Meanwhile
"11.5% to 46.4%" is a seed-42 figure, in a paragraph that claims 3-seed
evaluation, so one sentence mixes two number sets.

Consistent 3-seed replacement:

> Mean F1 macro từ 0.4202 lên 0.5265 (+0.106), Severe Recall tăng từ 12.3% lên
> 48.6% (gấp 4 lần)

Worth stating alongside it: **Mean Accuracy falls** (0.8103 to 0.7237). That is
the direct cost of quadrupling rare-class recall, and a committee reading the
table will see it whether or not the abstract mentions it.

*(The document itself has deliberately not been edited.)*

---

## 6. Reproduction

```bash
# T2 three-condition baseline on the correct init  (run (c))
python3 train_rsna_hybrid.py \
  --cbam-checkpoint checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth \
  --epochs 20 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
  --use-focal --focal-gamma 2.0 --use-supcon --supcon-weight 0.1 \
  --use-uncertainty True --class-weight-mode sqrt --augmentation medium \
  --oversample-factor 3 --select-by severe_auprc --seed 42 \
  --num-workers 4 --save-freq 5 --early-stop-patience 15 \
  --save-dir checkpoints/t2_sqrt_baseline

# T1 foraminal specialist
python3 train_rsna_hybrid.py \
  --data-dir rsna_preprocessed_t1 --split t1 \
  --conditions left_foraminal right_foraminal \
  --cbam-checkpoint checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth \
  --epochs 20 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
  --use-focal --focal-gamma 2.0 --use-supcon --supcon-weight 0.1 \
  --use-uncertainty True --class-weight-mode sqrt --augmentation medium \
  --oversample-factor 3 --select-by severe_auprc --seed 42 \
  --num-workers 4 --save-freq 5 --early-stop-patience 15 \
  --save-dir checkpoints/t1_hybrid

# Full metrics on either checkpoint
python3 experiments/f1_improvement/dump_logits.py --model hybrid \
  --data-dir rsna_preprocessed_t1 --split t1 \
  --conditions left_foraminal right_foraminal \
  --cbam-checkpoint checkpoints/fresh_cbam/best_model_attention_sqrt_cw_e20.pth \
  --hybrid-checkpoint checkpoints/t1_hybrid/best_model_hybrid_left_right_t1_severe_auprc.pth \
  --seed 42 --device cuda --output logits/t1_val.npz

python3 experiments/f1_improvement/robust_metrics.py --npz logits/t1_val.npz \
  --conditions left_foraminal right_foraminal --n-boot 2000
python3 experiments/f1_improvement/threshold_sweep.py --npz logits/t1_val.npz
python3 experiments/f1_improvement/temperature_calibration.py --npz logits/t1_val.npz
```

Data transfer note: Google Drive download quota is enforced **per account, not
per file**, so re-uploading a fresh copy does not lift it. Transfer the 17 GB T1
set with rsync instead (measured 22-29 MB/s, about 13 minutes). macOS ships
openrsync, which has no `--info=progress2`; use `rsync -a --stats`, and add `-r`
when using `--files-from` since `-a` does not imply it there.
