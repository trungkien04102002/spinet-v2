# Re-training model A

This repository is the research half of the thesis. It trains the grading
model the app ([spine-labeling-app](https://github.com/trungkien04102002/spine-labeling-app))
calls **model A**: a 3D ResNet34 with CBAM attention that grades spinal canal
stenosis and left/right foraminal stenosis (Normal/Mild, Moderate, Severe) on
each lumbar disc of a sagittal T2 MRI. The app loads it as
`backend/models/weights/phase2_cbam.pth`.

You only need this file to train a new version of that model. To run the
app, see the app's README; its weights are downloaded for you.

Most of the repository root is thesis material (reports, slides, papers,
experiment logs). The files that matter for training are listed below; the
rest can be ignored.

## Files that matter

| File | Role |
|---|---|
| `prepare_rsna_data.py` | RSNA DICOM -> one `(9, 112, 224)` float32 `.npy` crop per disc + metadata CSV |
| `rsna_preprocessed_dataloader.py` | loads those crops (fast path, use this) |
| `train_rsna_attention.py` | trains model A (CBAM); `train_rsna_baseline.py` is the same without CBAM |
| `spinenet/models/grading_attention.py` | model A architecture (`GradingModelWithCBAM`) |
| `spinenet/models/attention.py` | CBAM |
| `spinenet/losses.py` | focal loss (`ignore_index=-1`), uncertainty weighting, class weights |
| `spinenet/augmentation.py` | 3D augmentation, minority oversampling |
| `eval_rsna_auc.py`, `test_rsna_preprocessed.py` | evaluation |
| `CLAUDE.md` | longer reference: every script, flag and checkpoint convention |

## 1. Environment

Python 3.11 and an NVIDIA GPU (a 3090 does an epoch in a few minutes; a CPU
takes hours).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)     # required: the repo is not an installed package
```

## 2. Pretrained backbone

Model A starts from the original SpineNet grading network
([Windsor et al.](https://github.com/rwindsor1/SpineNet)); only the
backbone is reused, the heads and CBAM are trained. The file is SpineNet's
`grading/ckpt1.pt`, which the app's release already hosts:

```bash
curl -LO https://github.com/trungkien04102002/spine-labeling-app/releases/download/assets-v1/spinenet_weights.tar
tar -xf spinenet_weights.tar weights/grading/ckpt1.pt
mkdir -p ~/.spinenet/weights && mv weights/grading/ckpt1.pt ~/.spinenet/weights/
```

## 3. Data

[RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)
(~35 GB of DICOM; accept the competition rules on Kaggle first; the data may
not be redistributed).

```bash
kaggle competitions download -c rsna-2024-lumbar-spine-degenerative-classification
unzip -q rsna-2024-lumbar-spine-degenerative-classification.zip \
      -d rsna-2024-lumbar-spine-degenerative-classification
python3 prepare_rsna_data.py --skip-download     # -> rsna_preprocessed/ (~8 GB)
python3 visualize_npy.py --show-stats --num-samples 3   # sanity check
```

Labels: `-1` means "not labelled" and must stay `-1` through augmentation and
class weighting; the losses ignore it.

## 4. Train

The checkpoint the app ships was trained with these settings (they are stored
inside the `.pth` file under `args`):

```bash
python3 train_rsna_attention.py \
    --epochs 15 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 \
    --focal-gamma 1.8 --use-uncertainty false \
    --augmentation medium --oversample-factor 5 --num-workers 8 \
    --no-hflip-swap-labels
# -> checkpoints/best_model_attention.pth  (+ metrics JSON/CSV next to it)
```

CBAM, focal loss and a frozen backbone are on by default (`--no-cbam`,
`--use-focal`, `--no-freeze` change them). `--no-hflip-swap-labels`
reproduces the shipped model exactly: when that model was trained, a
horizontal flip did not swap left/right foraminal labels. That was a bug,
fixed afterwards; **for a new model leave the flag out** (the default swaps
them correctly). `--class-weight-mode sqrt` gives the minority (Severe) class
more weight and was the best setting in later experiments.

Validation is 20% of patients, split with `train_test_split(..., random_state=--seed)` (default 42). Watch the
per-class F1 printed every 5 epochs, not only accuracy: the point of the
whole imbalance stack (focal loss, oversampling, class weights) is the
Severe class, which a model can ignore and still score high accuracy.

## 5. Evaluate

```bash
python3 eval_rsna_auc.py --model cbam --checkpoint checkpoints/best_model_attention.pth
python3 test_rsna_preprocessed.py --model checkpoints/best_model_attention.pth --num-patients 5
```

## 6. Put it in the app

```bash
cp checkpoints/best_model_attention.pth \
   <spine-labeling-app>/backend/models/weights/my_model.pth
```

Then in the app's `/admin/models` page, activate `my_model.pth` (it writes the
name into `backend/models/weights/ACTIVE`; reverting is one click). The
checkpoint format is the one `train_rsna_attention.py` saves: a dict with
`model_state_dict` and `args`.

## Beyond model A

The thesis also trains a **Hybrid** model (3D CNN + frozen BiomedCLIP text
branch, grades by comparing with label sentences so new labels need no
retraining; `train_rsna_hybrid.py`, `HYBRID_TRAINING_GUIDE.md`) and tests on
**SPIDER** (`train_spider.py`, `SPIDER_TRAINING_GUIDE.md`). The app does not
use those.
