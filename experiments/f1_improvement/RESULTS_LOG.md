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

## Run #2 — multi-view concat (PENDING)

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
