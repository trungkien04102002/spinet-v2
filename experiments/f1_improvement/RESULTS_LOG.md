# F1 improvement — GPU run log (kì cuối)

> Running record of the #1/#2/#3 experiments. Written as results arrive so nothing
> is lost when the rented box is destroyed. Numbers here are copied from the box;
> the authoritative artifacts are the `*_best_metrics.json` files pulled down
> alongside this file.

## Box

Vast.ai instance `48374995`, `ssh -p 46108 root@115.75.223.236` (rented 2026-08-22).
1x RTX 4090 24 GB, driver 595.58.03, AMD EPYC 7742, 80 GB disk, Lexar ARES 4TB NVMe.
Env: `spinenet-venv` (torch 2.13.0+cu130, `cuda.is_available()` True).
Repo at `fe1c715` (shallow clone, includes the 2026-08-22 fixes).

Data verified before training: `rsna_preprocessed/` 8.3 GB (1973 studies),
`rsna_preprocessed_t1/` 17 GB (1972 studies, **19,689 crops** — matches the expected
count), CBAM warm-start checkpoint md5 `50101203903b1aa3bcf3d6103daa650e` (matches).

## Smoke tests — all three passed on GPU

| Run | Time | Warm-start | Trainable | Class weights |
|---|---|---|---|---|
| `#1 t1_foraminal --fast-dev` | 7.2 s | 228/234 | 50,881 / 63,507,969 | 1.000 / 1.000 / 1.000 |
| `#2 multiview --fusion concat --fast-dev` | 20.2 s | 2x 228/234 | 109,451 / 127,023,627 | 0.312 / 0.861 / 1.828 |
| `#3 multiview --fusion gated --fast-dev` | 20.3 s | 2x 228/234 | 370,065 / 127,284,241 | 0.312 / 0.861 / 1.828 |

Trainable counts and class weights matched a local CPU run exactly, so the runs are
reproducible across machines. Class weights near 1.0 in `--fast-dev` are expected:
the smoke subset is class-stratified, so it is balanced by construction.

T2/T1 pairing confirmed at load time:
`T1 metadata 19689 rows -> 9860 left / 9829 right`, `9748 T2 samples, T1 present=True`.
No `FileNotFoundError` — `--allow-missing-t1` covers the 51 T2 rows that have no T1
crop (50 of which already carry a `-1` label; exactly 1 real label is replaced by a
zero volume).

---

## Run #1 — T1-foraminal (DONE 2026-08-22, stopped by hand at epoch 20 of 25)

**Final: best epoch 20, Severe F1 (mean of left+right) = 0.3095**, val loss 0.5472,
accuracy left 75.4% / right 74.0%, `eval_samples` 3940, 4073 s of training (68 min).
Artifacts pulled to `experiments/f1_improvement/results/` and `run_t1.log`.

**Read the "Oscillation" subsection below before quoting 0.3095 anywhere.** The run
plateaued into a noisy band and `--select-by severe_f1` saved the top of that band.
The defensible plateau level is ~0.291, not 0.3095.


```
python3 experiments/f1_improvement/train_t1_foraminal.py \
    --cbam-checkpoint checkpoints/rsna/best_model_attention.pth \
    --class-weight-mode effective --select-by severe_f1 \
    --epochs 25 --batch-size 32 --lr 1e-3 \
    2>&1 | tee experiments/f1_improvement/run_t1.log
```

Setup confirmed from the log:

```
Loaded 19689 samples;  Train 15749 (1577 patients) / Val 3940 (395 patients)
Warm-started 228/234 tensors (backbone + 12 CBAM tensors; heads skipped)
Freezing backbone (training heads only)  ->  Trainable 50,881 / 63,507,969 (0.1%)
Class weights (effective): Normal=0.192  Moderate=0.556  Severe=2.252
```

Only the two foraminal heads are trained; `spinal_canal` is dropped because it is
`-1` for all 19,689 T1 rows. **So this run does NOT produce a 3-condition Mean F1
comparable to the paper's 0.527** — it is a foraminal-only diagnostic.

Throughput: 493 steps/epoch, ~2.73 it/s, **203.4 s per epoch** (25 epochs ~= 85 min).

### Severe F1 by epoch (mean of left+right)

| Epoch | val loss | Severe F1 |
|---|---|---|
| 1 | 0.6995 | 0.0260 |
| 2 | 0.6036 | 0.2051 |
| 3 | 0.6036 | 0.2247 |
| 4 | 0.5710 | 0.2612 |
| 5 | 0.5710 | 0.2662 |
| 6 | 0.5699 | 0.2722 |
| 7-9 | — | no improvement over epoch 6 |
| 10 | 0.5445 | 0.2753 |

### Per-class Severe, epoch 5 vs epoch 10

| | P (ep5) | R (ep5) | F1 (ep5) | P (ep10) | R (ep10) | F1 (ep10) | support |
|---|---|---|---|---|---|---|---|
| Left foraminal | 0.184 | 0.600 | 0.282 | 0.188 | 0.613 | 0.287 | 80 |
| Right foraminal | 0.171 | 0.472 | 0.251 | 0.167 | 0.625 | 0.263 | 72 |

### Per-epoch detail, epochs 8-13

Precision is not in the CSV, but it follows exactly from the logged F1 and recall:
`P = F1*R / (2R - F1)`.

| ep | Severe F1 | best | L F1 | L rec | L prec | R F1 | R rec | R prec |
|---|---|---|---|---|---|---|---|---|
| 8 | 0.2594 | | 0.271 | 0.637 | 0.172 | 0.248 | 0.514 | 0.164 |
| 9 | 0.1643 | | 0.173 | 0.138 | 0.234 | 0.155 | 0.111 | 0.258 |
| 10 | 0.2753 | * | 0.287 | 0.613 | 0.188 | 0.263 | 0.625 | 0.167 |
| 11 | 0.2951 | * | 0.293 | 0.500 | 0.207 | 0.297 | 0.514 | 0.209 |
| 12 | 0.2942 | | 0.285 | 0.575 | 0.189 | 0.304 | 0.472 | 0.224 |
| 13 | 0.3010 | * | 0.307 | 0.525 | 0.216 | 0.295 | 0.667 | 0.190 |

### Reading (2026-08-22)

**No collapse.** The July failure produced `Severe F1 = 0.0000` for 10 straight
epochs; here it is non-zero from epoch 1. The class-weight fix (commit `8d5432e`,
device placement) works on GPU — this was its first real GPU run.

**Correction to an earlier reading.** Comparing only epoch 5 and epoch 10 suggested
the curve had plateaued and precision was stuck at 0.17-0.19. The per-epoch CSV shows
otherwise: F1 went 0.2753 -> 0.2951 -> 0.2942 -> 0.3010 over epochs 10-13, and
precision rose with it to 0.19-0.22. The 5-epoch sampling was too sparse for a curve
this noisy — epoch 9 alone swung down to F1 0.164 (recall collapsed to 0.11-0.14 while
precision rose to 0.23-0.26), so single epochs are not reliable read-outs.

**Against the T2 baseline** (foraminal Severe F1 0.27-0.29, precision 0.22-0.23, the
bottleneck being precision): at epoch 13 this run reaches Severe F1 **0.3010** —
above the baseline range — with precision 0.19-0.22, roughly at the baseline's low
end. So a modest genuine gain on F1, and precision no longer clearly worse. Still
climbing at epoch 13 of 25.

**Caveat that cuts in T1's favour:** the comparison is unfair to T1. The 0.27-0.29
figure comes from the full Hybrid recipe (focal + SupCon + uncertainty weighting +
oversample x3 + augmentation, converged). This run is CBAM + plain cross-entropy +
class weights with a frozen backbone and 50,881 trainable parameters.

**Caveats that cut against it:** single seed (42); high epoch-to-epoch variance means
`--select-by severe_f1` may be picking a favourable epoch, so a 3-seed run is required
before claiming the gain. `--unfreeze-backbone --lr 1e-4` is still untested and
remains the natural next lever if the curve flattens near 0.30.

### Oscillation, and why 0.3095 must not be quoted as the result

Epochs 13-20, same derivation for precision (`P = F1*R / (2R - F1)`):

| ep | Severe F1 | best | L F1 | L rec | L prec | R F1 | R rec | R prec |
|---|---|---|---|---|---|---|---|---|
| 13 | 0.3010 | * | 0.307 | 0.525 | 0.216 | 0.295 | 0.667 | 0.190 |
| 14 | 0.2978 | | 0.307 | 0.738 | 0.194 | 0.288 | 0.639 | 0.186 |
| 15 | 0.3019 | * | 0.298 | 0.750 | 0.186 | 0.306 | 0.625 | 0.203 |
| 16 | 0.2765 | | 0.295 | 0.412 | 0.229 | 0.258 | 0.542 | 0.170 |
| 17 | 0.2813 | | 0.305 | 0.662 | 0.199 | 0.257 | 0.694 | 0.158 |
| 18 | 0.2731 | | 0.284 | 0.537 | 0.193 | 0.262 | 0.667 | 0.163 |
| 19 | 0.2897 | | 0.295 | 0.775 | 0.182 | 0.285 | 0.667 | 0.181 |
| 20 | 0.3095 | * | 0.322 | 0.625 | 0.216 | 0.298 | 0.500 | 0.212 |

Eight consecutive epochs sit in the band **0.273-0.310**. Epoch-to-epoch swing is
**+/-0.02 to 0.03**, which is larger than the +0.0076 "gain" from epoch 15 to epoch 20.
The curve stopped climbing around epoch 13; after that it oscillates.

**Mean of epochs 13-20 = 0.2914.** That is the honest plateau level.
**`--select-by severe_f1` saved 0.3095** — the top of the oscillation, 0.018 above the
plateau. Precision oscillates too (epoch 16 had left precision 0.229 with F1 only
0.2765), so the epoch-20 precision of 0.216/0.212 is not a settled trend either.

Consequences for the write-up:
- Report the plateau (~0.29), or report best-epoch with the oscillation stated. Do not
  present 0.3095 as a clean single number.
- **A 3-seed run is mandatory before claiming any improvement.** Against the T2
  baseline range (foraminal Severe F1 0.27-0.29), a plateau of 0.291 is *the top of
  that range*, i.e. parity to marginal gain — not the step change the hypothesis
  predicted.
- The `--select-by severe_f1` flag exists to avoid saving a majority-collapsed epoch
  (the July failure). It does that job, but on a noisy curve it also biases the saved
  number upward. Worth a sentence in the methods section.

### Decision rule that was set for epoch 15

Continue to 25 epochs only if **both**: Severe F1 >= 0.30 **and** Severe precision
>= 0.21. As of epoch 13, F1 is met (0.3010) and precision is borderline (left 0.216,
right 0.190) — so the run was allowed to continue rather than being cut for run #2.

---

## Course correction, 2026-08-22 — runs #1/#2/#3 were built on the wrong base

`GradingMultiView` uses `GradingModelWithCBAM` for BOTH branches. It has no BiomedCLIP
branch, no text anchors and no cosine head: it ends in a plain `nn.Linear`. So the whole
multi-view line does not extend the published Hybrid, it **replaces** it with a
CBAM-only two-branch model and starts from a weaker base. That is why it cannot reach
0.356 -- it was never the same architecture.

Meanwhile `train_rsna_hybrid.py:166` has had `--fusion gated` all along, `GatedFusion`
(GMU, Arevalo et al. 2017) is implemented at `grading_hybrid.py:82`, and the comment at
line 171 says it exists to "test whether a smarter fusion improves over the default".
**Every published Phase 2 run used `concat_mlp`; gated has never been run.** It also
comes with the full Phase 2 recipe already wired -- focal + SupCon + uncertainty
weighting + sqrt class weights + oversample x3 + augmentation -- i.e. the five things
the multi-view scripts were missing, at zero code cost.

### The CBAM init used by the published Hybrid is gone

`hybrid_best_metrics.json` records `cbam_checkpoint =
checkpoints/v3_20260503/cbam/best_model_attention.pth` (epoch 14, Severe F1 0.3044,
`class_weight_mode=sqrt`, `oversample_factor=5`). That file was deleted on 2026-08-04.
Every `*.json` in the repo carrying `val_accuracies` was checked against its recorded
accuracies (canal 0.878476 / L 0.613836 / R 0.577181); only the original metrics file
matches, so **no surviving copy exists**. Note also that
`checkpoints/rsna/best_model_attention.pth` is byte-identical (md5 `50101203...`) to
`checkpoints/no_class_weight/best_model_attention.pth`, i.e. it is the no-class-weight
ablation, not the `sqrt` checkpoint the Hybrid was warm-started from.

Consequence: a `gated` run cannot be compared directly against the published
`concat_mlp` number, because the init differs too. Hence two runs, both from
`checkpoints/rsna/best_model_attention.pth`: a `concat_mlp` re-baseline and then
`gated`. The A/B is internally valid whatever the init.

### Comparison targets -- do not mix seed sets

The published Hybrid numbers per seed: **seed 42 = 0.3434**, seed 123 = 0.3482,
seed 456 = 0.3753, 3-seed mean = 0.3556. A seed-42 run must be compared against
**0.3434**, not against 0.356.

### What "stuck mode" looks like

`checkpoints/v3_20260503/hybrid/hybrid_log.csv` turns out to hold THREE appended runs:

| run | epochs | max Severe F1 | max foraminal | mean val_loss |
|---|---|---|---|---|
| 1 | 20 | 0.2264 | 0.080 | 0.1781 |
| 2 | 6 | 0.2402 | 0.082 | 0.1841 |
| 3 | 20 | **0.3434** | 0.287 | 0.1447 | <- the published run |

In runs 1 and 2 the foraminal heads never left ~0 and the mean stalled at 0.22-0.24.
In run 3 foraminal woke up at epoch 2 and the mean jumped to 0.3128.

**Do not read this as seed fragility.** Runs 1-2 sit at val_loss ~0.178-0.184 against
run 3's ~0.145; a systematic gap that size points at a different configuration rather
than seed variance, and the CSV stores no per-row args, so the earlier runs' flags are
unknown and unrecoverable. What it does give us is a signature for the failure mode:
foraminal pinned at 0 with val_loss around 0.18.

## Run #2 — multi-view concat (DONE 2026-08-22, early-stopped)

Ran the full `--epochs 20` and stopped itself at epoch 14 after 10 epochs without
improvement, so this number needs no "killed by hand" caveat.

**Best epoch 4, Severe F1 0.3323**, val_loss 0.5373, accuracy canal 0.895 / L 0.682 /
R 0.683, `eval_samples` 1942 (directly comparable to the paper's split), 20 min of
training. Plateau over the last six epochs: mean **0.3227**, range 0.3184-0.3282 -- so
0.3323 is again a peak rather than the converged level.

Beats both external SOTA baselines (0.271, 0.257) but sits below the seed-42 Hybrid
(0.3434). Foraminal never passed 0.26 in any epoch, which is the whole gap.

## Run #2b — multi-view concat + Phase 2 recipe (ABANDONED before running)

Superseded by the course correction above: fixing the recipe on the wrong architecture
is worth less than running the right architecture with the recipe it already has. The
`--use-focal` / `--focal-gamma` / `--oversample-factor` flags added to
`train_multiview.py` (commit `51ea246`) stay available and default to off, so runs #2
and #3 remain comparable.

## ⚠️ READ FIRST if you are pulling numbers for the report

Nothing in the (a)/(b) section below is a final number. They are **screening runs**:
single seed (42), warm-started from a CBAM checkpoint that is *not* the one the
published Hybrid used (that file was deleted, see above), and possibly cut short.

**Comparison rules that must not be broken:**
1. A seed-42 run compares against the seed-42 published number **0.3434**, never
   against the 3-seed mean 0.356.
2. (a) and (b) must be compared at the **same epoch budget**. If one ran 12 epochs and
   the other 20, compute best-within-12 for both from the per-epoch CSVs; a longer run
   has more chances to catch a peak, and on this task the peak-to-trough swing is
   +/-0.03, which is larger than any effect being measured.
3. Report the plateau (mean of the last several epochs), not only the best epoch.
   `--select-by severe_f1` saves the top of the oscillation; on run #1 that inflated the
   saved number by 0.018 over the plateau.

**To produce a number that belongs in the thesis:** whichever fusion wins the screen,
run it AND its counterpart for the full 20 epochs on seeds 42/123/456, then report
mean +/- std. That is 6 runs, about 5 hours, no code changes.

## Run (a) — Hybrid `concat_mlp` re-baseline (2026-08-22)

Flags generated directly from the published run's saved `args` so the only intended
difference is the CBAM init. 144 s/epoch, 420 steps, ~48 min for 20 epochs; VRAM only
2.8 GB because both backbones are frozen. BiomedCLIP downloaded to `/workspace/.hf_home`
(748 MB) -- the Vast image sets `HF_HOME` there, not `~/.cache/huggingface`.

**DONE, full 20 epochs.** Best: **epoch 15, Severe F1 0.3212** (canal 0.557 / L 0.206 /
R 0.201), val_loss 0.1488, accuracy canal 0.874 / L 0.614 / R 0.621, `eval_samples` 1942,
36 min of training. Artifacts in `experiments/hybrid/hybrid_rebase_*`.

### The published 0.3434 is a lucky peak, and this run shows it

| | best epoch | plateau (last 8 epochs) |
|---|---|---|
| published (seed 42) | **0.3434** | 0.2969 |
| re-baseline | 0.3212 | **0.3050** |
| delta | **-0.0222** | **+0.0080** |

The re-baseline is *worse at the peak but more stable*: the published run's best sits
**+0.046 above its own plateau**, the re-baseline's only +0.016. This is the ordinary
consequence of selecting the best epoch on validation — standard practice, not
misconduct — but it means the Hybrid's converged performance is materially below the
number that was reported. If every seed's peak sits ~0.04 above its plateau, the
"converged" 3-seed figure would be around 0.31 rather than 0.356.

**Worth stating proactively in the thesis.** It is a much weaker position to have a
committee notice it first. The honest framing: report the best-epoch number as the paper
does, and add the plateau alongside it.

Comparison targets for run (b), which shares this init and epoch budget:
**best 0.3212** and **plateau 0.3050**. A win on the peak alone could be luck; a win on
both is evidence.

Epoch-by-epoch against the published run (same recipe, same split, same seed; only the
CBAM init differs):

| ep | published | re-baseline | delta | rebase canal / L / R |
|---|---|---|---|---|
| 1 | 0.1973 | 0.1731 | -0.0242 | 0.519 / 0.000 / 0.000 |
| 2 | 0.3128 | 0.2908 | -0.0220 | 0.415 / 0.217 / 0.241 |
| 3 | 0.3358 | 0.3129 | -0.0230 | 0.444 / 0.231 / 0.263 |
| 4 | 0.3046 | 0.2888 | -0.0158 | 0.533 / 0.177 / 0.156 |
| 5 | 0.3283 | 0.3192 | -0.0091 | 0.564 / 0.203 / 0.190 |
| 6 | 0.2670 | 0.2596 | -0.0074 | 0.483 / 0.166 / 0.130 |
| 7 | 0.2545 | 0.2602 | +0.0057 | 0.571 / 0.111 / 0.098 |
| 8 | 0.2944 | 0.2958 | +0.0014 | 0.550 / 0.168 / 0.170 |
| 9 | 0.3289 | 0.3069 | -0.0220 | 0.465 / 0.231 / 0.224 |
| 10 | **0.3434** | 0.3081 | -0.0353 | 0.558 / 0.184 / 0.183 |
| 11 | 0.2521 | 0.2857 | +0.0336 | 0.573 / 0.159 / 0.125 |
| 12 | 0.3366 | 0.3154 | -0.0212 | 0.544 / 0.183 / 0.219 |
| 13 | 0.2821 | 0.2705 | -0.0115 | 0.519 / 0.157 / 0.136 |
| 14 | 0.3319 | 0.3176 | -0.0142 | 0.541 / 0.215 / 0.197 |
| 15 | 0.3302 | **0.3212** | -0.0090 | 0.557 / 0.206 / 0.201 |
| 16 | 0.2990 | 0.3108 | +0.0118 | 0.562 / 0.211 / 0.160 |
| 17 | 0.2818 | 0.3035 | +0.0217 | 0.551 / 0.203 / 0.156 |
| 18 | 0.2919 | 0.3165 | +0.0247 | 0.554 / 0.219 / 0.176 |
| 19 | 0.2799 | 0.2982 | +0.0183 | 0.563 / 0.190 / 0.142 |
| 20 | 0.2790 | 0.3013 | +0.0223 | 0.566 / 0.190 / 0.148 |

Qualitatively identical dynamics: foraminal is exactly 0.000 on both sides at epoch 1,
wakes at epoch 2, and canal drops as capacity shifts to foraminal (0.519 -> 0.415 here,
0.592 -> 0.455 published). So the deleted init does **not** put the run into the stuck
mode described above -- val_loss reached 0.1439 by epoch 12 against the published
0.1436.

The gap is not a constant offset: it narrows to +0.006 by epoch 7, then reopens to
-0.035 at epoch 10, the published run's peak. **Two of my own mid-run readings were
wrong** and are recorded here as a caution: at epoch 4 I called it "a stable -0.02", and
at epoch 8 "the gap closes, the init only causes delay". Both were extrapolations from
2-4 points on a curve that swings +/-0.03. On this task, do not infer a trend from fewer
than about five epochs.

Where the gap actually sits: at epoch 10 the published foraminal is 0.278 / 0.252 while
the re-baseline is 0.184 / 0.183 — canal is comparable. The deficit is concentrated in
the hardest condition, which fits the init being the `no_class_weight` ablation rather
than the `sqrt`-weighted checkpoint the published run started from.

## Run (b) — Hybrid `--fusion gated` (DONE 2026-08-22, 20/20 epochs)

**Verdict: gated LOSES to concat_mlp. Do not use it.** Artifacts pulled to
`experiments/hybrid/gated/` (metrics json/txt + per-epoch csv).

| | Severe F1 best | at epoch | plateau (last 3) |
|---|---|---|---|
| (a) `concat_mlp` — same init, same 20 epochs | **0.3212** | 15 | **0.3053** |
| (b) `gated` | 0.2954 | 5 | 0.2812 |

Behind at the peak by 0.026 and at the plateau by 0.024. **19 of 20 epochs were below
(a)**; only epoch 1 was above. (b)'s best was set at epoch 5 and never beaten — early
stopping fired at epoch 20 after 15 epochs without improvement. The single result is
therefore not a noise artifact: the sign is consistent across the whole run.

Mean Severe F1 per condition over all 20 epochs:

| condition | (a) concat_mlp | (b) gated | delta |
|---|---|---|---|
| spinal_canal | 0.5316 | 0.5217 | -0.010 |
| left_foraminal | 0.1811 | 0.1484 | -0.033 |
| right_foraminal | 0.1658 | 0.1262 | -0.040 |

Gated is worse on every condition, but the damage is 3-4x larger on the foraminal heads
than on canal. That asymmetry is the interesting part: the gate degrades most where the
BiomedCLIP semantic branch should be contributing most.

### Correction to the mid-run reading (recorded deliberately)

At epoch 10 this log's author reported "gated is BETTER on canal (0.547 vs 0.524) and
the gate has collapsed onto the CBAM branch". Over the full 20 epochs that is **wrong**:
canal ends slightly worse (-0.010), not better. The epoch-10 claim came from a
3-epoch trailing window that happened to catch a canal peak. What survives is the
weaker, still-useful statement: **the foraminal heads are hurt far more than canal**,
consistent with the gate under-weighting the semantic branch — but there is no canal
"gain" to trade against.

This is the third mid-run trend call in one day that a longer window overturned (the
others are recorded above). Reinforced rule: **do not report a per-condition trend from
a trailing window shorter than the full run.** Report per-condition numbers only at the
end, or state the window explicitly.

### Why it plausibly fails (mechanism, for the write-up)

`GatedFusion` forms `z*tanh(h_a(cbam)) + (1-z)*tanh(h_b(bmc))` with `z` a 512-dim
per-sample sigmoid. To lean fully on one branch the gate must saturate, and the two
branches are forced through a single convex-ish blend. `ConcatMLPFusion` has no such
constraint: its MLP can route different feature subspaces from each branch
independently. Gated also has **11% fewer trainable parameters** (1,050,626 vs
1,181,442), so capacity is a confound that cannot be ruled out from this run alone --
state that limitation rather than claiming a clean architectural conclusion.

For the thesis this is a usable negative result: it answers the advisor's
"leader-supporter" suggestion (#2) empirically, on the published architecture, with the
zero-shot cosine head intact.

### The diagnostic that came out of this run

Per-class table at epoch 19, left foraminal:

| class | support | precision | recall | F1 |
|---|---|---|---|---|
| Normal/Mild | 1497 | 0.926 | 0.639 | 0.757 |
| Moderate | 360 | 0.318 | **0.750** | 0.447 |
| Severe | 80 | 0.200 | **0.138** | 0.163 |

Severe recall 13.8% while Moderate recall is 75% and Moderate precision is only 31.8%:
**the Severe cases are being absorbed into Moderate.** This is an adjacent-class,
ordinal failure. Every technique tried so far (focal, class weights, oversampling)
targets rarity, not ordinality -- cross-entropy penalises Severe->Moderate exactly as
much as Severe->Normal. This reframes what to try next; see the research notes below.

---

Exact command (flags generated from the published run's saved `args`, not typed from
memory — `--use-uncertainty` takes a value, `--oversample-factor` defaults to 5 but the
paper used 3, and `--hflip-swap-labels` defaults to True but the paper used False):

```bash
cd ~/spinet-v2
source spinenet-venv/bin/activate          # required: bare python3 has no torch
CK=checkpoints/rsna/best_model_attention.pth
RECIPE="--data-dir rsna_preprocessed --val-split 0.2 --epochs 20 --batch-size 32 \
        --lr 0.0001 --weight-decay 0.0001 --slice-strategy static \
        --use-focal --focal-gamma 2.0 --use-supcon --supcon-weight 0.1 \
        --use-uncertainty True --class-weight-mode sqrt --augmentation medium \
        --no-hflip-swap-labels --oversample-factor 3 --num-workers 4 \
        --save-freq 5 --early-stop-patience 15 --seed 42 \
        --ablate-branch none --modality-dropout 0.0"

python3 train_rsna_hybrid.py --cbam-checkpoint $CK --fusion gated $RECIPE \
    --save-dir checkpoints/hybrid_gated \
    2>&1 | tee experiments/hybrid/run_hybrid_gated.log
```

Confirm `Fusion head: gated` appears at startup — the script only prints that line when
fusion differs from `concat_mlp`, so its absence means the flag did not take. ~144 s per
epoch, about 48 min for 20 epochs.

Same command as (a) with `--fusion gated`. This is the advisor's leader-supporter ask evaluated
**on the published architecture**, and it reuses the BiomedCLIP branch and cosine head,
so it keeps the zero-shot label-extension contribution intact. BiomedCLIP is now cached
at `/workspace/.hf_home` on the box, so no download.

Exactly one component changes. Everything else — frozen CBAM branch, frozen BiomedCLIP
branch, slice-attention pool, cosine head against frozen text anchors, focal + SupCon +
uncertainty weighting + sqrt weights + oversample x3 + augmentation, data, split, seed —
is byte-identical to (a):

```python
# (a) ConcatMLPFusion  -- "the MLP is the only place where per-modality information mixes"
proj(cat([feat_cbam, feat_bmc]))                  # 1024 -> MLP(768) -> 512

# (b) GatedFusion (GMU, Arevalo et al. 2017)
ha = tanh(h_a(feat_cbam)); hb = tanh(h_b(feat_bmc))
z  = sigmoid(gate(cat([feat_cbam, feat_bmc])))    # 512-dim, per sample
z * ha + (1 - z) * hb
```

Note this is a *different* gate from `GMUConditionGate` in `experiments/multiview/`: it
projects each branch separately before gating, so the capacity concern raised about the
multi-view gated variant (convex combination collapsing 1024 -> 512) does **not** apply
here.

Compare against **(a)'s best, not the published 0.3434**, since (b) shares (a)'s init.

## Remaining untested knobs on the published architecture (0 lines of code)

All already wired into `train_rsna_hybrid.py`; none was ever varied in Phase 2.

| Knob | Published value | Why it might matter |
|---|---|---|
| `--fusion gated` | `concat_mlp` | run (b) |
| `--slice-strategy dynamic` | `static` (3 centre slices) | Picks slices by cosine similarity to the text anchor instead of hard-coding the three central ones. The foramina are **lateral** structures, so a fixed midline crop may simply not contain them — which is the standing explanation for foraminal being the worst condition. Best remaining idea after (b). |
| `--modality-dropout 0.1-0.2` | `0.0` | Zeroes one branch at random during training; regularises and forces each branch to stand alone. May damp the oscillation. |
| `--class-weight-mode effective` | `sqrt` | cheap sweep |
| `--oversample-factor 5` | `3` | the CBAM run used 5 |
| `--supcon-weight` | `0.1` | never swept |
| post-hoc threshold tuning on the winner | — | already proven +0.019 on the Hybrid (seed 42); costs no training and **stacks** on top of whatever wins |

```
python3 experiments/multiview/train_multiview.py --fusion concat \
    --cbam-checkpoint checkpoints/rsna/best_model_attention.pth \
    --allow-missing-t1 --class-weight-mode effective --select-by severe_f1 \
    --epochs 20 --batch-size 16 --lr 1e-3 \
    2>&1 | tee experiments/multiview/run_concat.log
```

Why this is now the more important experiment: run #1 **replaces** T2 with T1 for the
foraminal heads. Run #2 **keeps both** and fuses them, and it produces all three
conditions, so its Mean F1 is directly comparable to the existing T2-only numbers.

## Run #3 — gated leader-supporter (PENDING)

Same command with `--fusion gated`. Expect it to lose to concat on F1: the GMU gate is
a convex combination (weights sum to 1) giving a 512-dim fused vector, while concat
keeps both and gives 1024 dim. Its value is interpretability —
`model.get_gate_weights()` returns the per-case T1/T2 split, which is what the advisor
asked to see.

---

## Pulling artifacts off the box (do this before destroying the instance)

```bash
# from the Mac
scp -P 46108 root@115.75.223.236:'/root/spinet-v2/checkpoints/t1_foraminal/*.{json,csv,txt}' \
    experiments/f1_improvement/results/
scp -P 46108 root@115.75.223.236:'/root/spinet-v2/experiments/multiview/checkpoints/*.{json,csv,txt}' \
    experiments/multiview/
scp -P 46108 root@115.75.223.236:/root/spinet-v2/experiments/f1_improvement/run_t1.log \
    experiments/f1_improvement/
scp -P 46108 root@115.75.223.236:'/root/spinet-v2/experiments/multiview/run_*.log' \
    experiments/multiview/
```

Checkpoints themselves (243 MB for #1, 489 MB each for #2/#3) only need pulling if
further inference, threshold sweeps, or Grad-CAM are planned.

---

## Research notes, 2026-08-22 (after run (b)) — where the remaining headroom probably is

Literature checked against the evidence above, not in the abstract. Ranked by
value-per-GPU-hour, cheapest first.

### Tier 1 — no training required

**A. Inspect the geometry of the three frozen text anchors.** The head is
`logits = logit_scale * image_emb @ text_embs.T` (`grading_hybrid.py:347`): three frozen
BiomedCLIP text anchors per condition. Nothing constrains Moderate and Severe to be far
apart in that space. If `cos(Moderate, Severe)` is very high, that IS the mechanism
behind the Severe->Moderate absorption measured above, and the fix is rewriting the
anchor prompts — zero training. Cost: ~2 minutes, one model load, print a 3x3 matrix.
Highest value-to-cost ratio of anything on this list; do it first. Falsifiable either way.

**B. Post-hoc threshold tuning on the winning config.** Already proven +0.019 on the
Hybrid (seed 42). With Severe precision 0.200 and recall 0.138 the decision boundary is
far from the F1 optimum, so headroom here may exceed the earlier +0.019. Stacks on top
of whatever else wins, so run it last.

### Tier 2 — small code change, one run each

**C. Ordinal-aware loss.** Targets the diagnostic directly. Cheapest form is Class
Distance Weighted CE (arXiv:2412.01246): scale each error by `|pred - true|^alpha` so
Severe->Normal costs more than Severe->Moderate. A few lines in `spinenet/losses.py`,
**keeps the cosine head**. CORAL/CORN are stronger but replace the head with K-1 binary
outputs, which would destroy the zero-shot label-extension contribution — do not use
them here.

**D. OGM-GE gradient modulation** (Peng et al., CVPR 2022 oral,
arXiv:2203.15332). Monitors each branch's contribution and damps the gradient of the
dominant one. This is the treatment for the imbalance measured in run (b), and it
sequences well in the write-up: gated fusion diagnosed the problem, OGM-GE treats it.

**E. `--slice-strategy dynamic`.** Still 0 lines of code. Foramina are lateral; the
static crop takes three central slices.

### Tier 3 — larger, and one thing to explicitly NOT do

Checked the actual winning solutions on this dataset. The 2nd place solution
(github.com/brendanartley/RSNA-2024-Competition — **already vendored in this repo as a
SOTA baseline**) is two-stage: predict per-level coordinates first, then classify;
foraminal from Sag-T1, canal and subarticular from Sag-T2; encoder -> LSTM -> attention
over 24 frames, pseudo-labels, 9-rotation TTA, ~24 h on an A100 40GB. 1st place is also
localize-then-classify over multiview 2.5D crops.

Two takeaways:
- **Our sequence routing matches the 2nd place solution.** No change needed there.
- **Axial adds little.** 2nd place used sagittal only; their axial variant moved CV by
  0.01-0.02 and did not help the ensemble. So the advisor's suggestion #1 (add axial) is
  high-effort / low-return — recommend dropping it for the 10-week window, and saying so
  explicitly with this citation rather than silently skipping it.

**Unfreezing the backbone is an open empirical question, not a known win.** Only 0.4% of
parameters (1.05M/260M) are currently trainable. The literature is genuinely split:
frozen features can beat fine-tuning on small medical datasets and can help rare classes
(arXiv:2204.00484), while linear probing scores worst in other benchmarks. One run
answers it for this dataset; do not assume either direction.

### Suggested order

A (2 min) -> E (0 code) -> C -> D, evaluating after each. B last, since it stacks.
