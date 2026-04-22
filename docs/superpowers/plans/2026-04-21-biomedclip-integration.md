# BiomedCLIP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate BiomedCLIP text encoder with existing CBAM ResNet34 backbone to enable zero-shot label classification while preserving the Severe-class recall improvements from Focal Loss training.

**Architecture:** Replace fixed Linear classification heads with a single learnable Linear projection (512→512) that aligns image features from CBAM backbone with text embeddings from frozen BiomedCLIP text encoder. Label predictions become cosine similarity scores. Label set extensibility: adding a new label only requires encoding its text prompt through BiomedCLIP, no retraining.

**Tech Stack:** PyTorch, `open_clip_torch`, HuggingFace `transformers`, existing SpineNetV2 codebase (CBAM ResNet34 3D + Focal Loss), RSNA 2024 preprocessed dataset, SPIDER dataset for zero-shot evaluation.

---

## Cross-cutting constraints

1. **Branch discipline**: Work on branch `biomedclip-integration` (already created). Do NOT work on `main` or `phase1-baseline-analysis`.
2. **PYTHONPATH**: All scripts need `export PYTHONPATH=$PYTHONPATH:$(pwd)` because repo is not pip-installed.
3. **Preserve existing work**: Load existing `checkpoints/best_model_attention.pth` (CBAM+Focal RSNA). Do not retrain CBAM backbone from scratch.
4. **Frozen components**: BiomedCLIP text encoder is frozen. CBAM backbone is frozen (unless Task 11 LoRA option is chosen).
5. **Only projection layer + temperature parameter are trainable** in baseline runs. ~262K params.
6. **Binary convention for reporting**: Moderate|Severe → 1, Normal/Mild → 0 (consistent with Phase 1 / Phase 2 evaluators).
7. **Language policy**: Code and comments in English; chat with user in Vietnamese (per project memory).
8. **All commits**: Co-author `Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.

## Execution environment

The user runs training on rented Vast.ai GPU instances, not locally. Reference workflow from `script_vast_ai.txt`:

```bash
# On Vast.ai instance (tmux session for persistence across disconnects):
tmux set -g mouse on
./vast_setup.sh --branch biomedclip-integration     # checkout branch + install deps
pip3 install -r requirements.txt
./2_download_preprocessed.sh <GDRIVE_FILE_ID>       # download rsna_preprocessed.zip (~8GB)
./3_download_weights.sh <GDRIVE_WEIGHTS_ID>         # download ckpt1.pt backbone

# Run training; detach with Ctrl+B then D, reattach with `tmux attach`:
python3 train_biomedclip.py --epochs 30 --batch-size 32 --lr 1e-3

# After training, copy checkpoint back to local machine:
scp -P <PORT> root@<VAST_IP>:/root/spinet-v2/checkpoints/best_model_biomedclip.pth checkpoints/
```

**Important for Task 3**: BiomedCLIP weights (~500MB) must be downloaded on the Vast.ai instance during training. The user's local dataset `rsna_preprocessed/` (8-10GB) must already be on the instance via `2_download_preprocessed.sh`.

**Local vs remote execution**:
- Tasks 1-4 (setup + model code + tests): Can run locally on CPU
- Task 7 (full training): Must run on Vast.ai GPU
- Tasks 8-9 (evaluation): Can run locally after copying checkpoint back

---

## File structure

### New files
| Path | Responsibility |
|---|---|
| `spinenet/models/grading_biomedclip.py` | `SpineNetCBAMWithBiomedCLIP` model class |
| `spinenet/label_prompts.py` | Label text prompt templates for RSNA + SPIDER |
| `build_label_embeddings.py` | Pre-compute and cache BiomedCLIP text embeddings |
| `train_biomedclip.py` | Training script (contrastive) |
| `evaluate_biomedclip.py` | Supervised + zero-shot evaluation |
| `tests/test_grading_biomedclip.py` | Unit tests for model |
| `tests/test_label_prompts.py` | Unit tests for prompt generation |
| `tests/test_label_embeddings.py` | Smoke test for embedding pipeline |
| `experiments/paper_results/label_embeddings_rsna.pt` | Cached text embeddings (generated) |
| `experiments/paper_results/label_embeddings_spider.pt` | Cached for SPIDER unseen labels |
| `experiments/paper_results/biomedclip_results.md` | Results writeup |

### Modified files
| Path | Change |
|---|---|
| `requirements.txt` | Add `open_clip_torch`, pin transformers version |
| `spinenet/models/__init__.py` | Export new class |

---

## Pre-flight

- [ ] **P.0.1: Verify working directory**

```bash
cd /Users/kienha/spinet-v2 && pwd && git branch --show-current
```

Expected: `/Users/kienha/spinet-v2` and some existing branch (e.g., `phase1-baseline-analysis`).

- [ ] **P.0.2: Create new branch for this work**

```bash
cd /Users/kienha/spinet-v2 && git checkout -b biomedclip-integration
```

Expected: `Switched to a new branch 'biomedclip-integration'`.

- [ ] **P.0.3: Verify existing checkpoint exists**

```bash
ls -la /Users/kienha/spinet-v2/checkpoints/best_model_attention.pth
```

Expected: File exists, ~100MB. If missing, Phase 2 work cannot start — recover from training first.

- [ ] **P.0.4: Verify GPU available**

```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected: `CUDA available: True`. If CPU only, Tasks 8–13 cannot run in reasonable time.

---

## Task 1: Install dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **1.1: Check current dependencies**

```bash
cd /Users/kienha/spinet-v2 && grep -E "open_clip|transformers" requirements.txt
```

Expected: likely no match or only transformers. Need to add `open_clip_torch`.

- [ ] **1.2: Add new dependencies to requirements.txt**

Append these lines to `/Users/kienha/spinet-v2/requirements.txt`:

```
open_clip_torch>=2.24.0
transformers>=4.35.0
```

- [ ] **1.3: Install**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && pip install open_clip_torch>=2.24.0 transformers>=4.35.0
```

Expected: Packages install successfully. May take 2-5 min.

- [ ] **1.4: Verify installation**

```bash
python3 -c "import open_clip; from open_clip import create_model_from_pretrained; print('open_clip OK, version:', open_clip.__version__)"
```

Expected: prints version, no ImportError.

- [ ] **1.5: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add requirements.txt && git commit -m "deps: add open_clip_torch and transformers for BiomedCLIP integration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Define label prompt templates

**Files:**
- Create: `spinenet/label_prompts.py`
- Test: `tests/test_label_prompts.py`

- [ ] **2.1: Write failing test**

Create `/Users/kienha/spinet-v2/tests/test_label_prompts.py`:

```python
"""Unit tests for label prompt generation."""
from spinenet.label_prompts import (
    build_rsna_prompts,
    build_spider_unseen_prompts,
    RSNA_CONDITIONS,
    RSNA_SEVERITY_CLASSES,
)


def test_rsna_prompts_count_is_9():
    prompts = build_rsna_prompts()
    assert len(prompts) == 9, f"expected 9 prompts for 3 conditions x 3 classes, got {len(prompts)}"


def test_rsna_prompts_contain_condition_names():
    prompts = build_rsna_prompts()
    joined = " ".join(prompts).lower()
    assert "spinal canal" in joined
    assert "left" in joined and "foraminal" in joined
    assert "right" in joined and "foraminal" in joined


def test_rsna_prompts_contain_severity_levels():
    prompts = build_rsna_prompts()
    joined = " ".join(prompts).lower()
    for sev in ["normal", "mild", "moderate", "severe"]:
        assert sev in joined, f"severity '{sev}' missing from prompts"


def test_spider_prompts_is_nonempty_list_of_strings():
    prompts = build_spider_unseen_prompts()
    assert len(prompts) > 0
    assert all(isinstance(p, str) for p in prompts)


def test_rsna_constants_match():
    assert RSNA_CONDITIONS == ("spinal_canal", "left_foraminal", "right_foraminal")
    assert RSNA_SEVERITY_CLASSES == ("normal_mild", "moderate", "severe")
```

- [ ] **2.2: Run test to verify it fails**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_label_prompts.py -v
```

Expected: ModuleNotFoundError for `spinenet.label_prompts`.

- [ ] **2.3: Implement label prompts**

Create `/Users/kienha/spinet-v2/spinenet/label_prompts.py`:

```python
"""Label text prompt templates for BiomedCLIP-based label embedding.

The prompts follow the pattern documented by BiomedCLIP papers: prefix with
"a magnetic resonance image of lumbar spine showing" then describe the label
in natural clinical language. This maximizes alignment with text embeddings
learned from PMC-15M pretraining.
"""
from __future__ import annotations

RSNA_CONDITIONS: tuple[str, ...] = (
    "spinal_canal",
    "left_foraminal",
    "right_foraminal",
)

RSNA_SEVERITY_CLASSES: tuple[str, ...] = (
    "normal_mild",
    "moderate",
    "severe",
)

_RSNA_CONDITION_DISPLAY = {
    "spinal_canal": "spinal canal stenosis",
    "left_foraminal": "left neural foraminal narrowing",
    "right_foraminal": "right neural foraminal narrowing",
}

_RSNA_SEVERITY_DISPLAY = {
    "normal_mild": "normal or mild",
    "moderate": "moderate",
    "severe": "severe",
}


def build_rsna_prompts() -> list[str]:
    """Return 9 prompts for 3 RSNA conditions x 3 severity classes.

    Index convention (matters for training label mapping):
        0: spinal_canal x normal_mild
        1: spinal_canal x moderate
        2: spinal_canal x severe
        3: left_foraminal x normal_mild
        ...
        8: right_foraminal x severe
    """
    prompts = []
    for cond in RSNA_CONDITIONS:
        for sev in RSNA_SEVERITY_CLASSES:
            cond_text = _RSNA_CONDITION_DISPLAY[cond]
            sev_text = _RSNA_SEVERITY_DISPLAY[sev]
            prompts.append(
                f"a magnetic resonance image of lumbar spine showing {sev_text} {cond_text}"
            )
    return prompts


def build_spider_unseen_prompts() -> list[str]:
    """Return prompts for SPIDER labels the model has NOT been trained on.

    Used for zero-shot evaluation: SPIDER's pfirrmann grading, disc herniation,
    and spondylolisthesis are conceptually related to RSNA stenosis but have
    different label space.
    """
    return [
        # Pfirrmann grade 1-5
        "a magnetic resonance image of intervertebral disc with Pfirrmann grade 1",
        "a magnetic resonance image of intervertebral disc with Pfirrmann grade 2",
        "a magnetic resonance image of intervertebral disc with Pfirrmann grade 3",
        "a magnetic resonance image of intervertebral disc with Pfirrmann grade 4",
        "a magnetic resonance image of intervertebral disc with Pfirrmann grade 5",
        # Disc herniation (binary)
        "a magnetic resonance image of intervertebral disc without herniation",
        "a magnetic resonance image of intervertebral disc with herniation",
        # Spondylolisthesis (binary)
        "a magnetic resonance image of lumbar spine without spondylolisthesis",
        "a magnetic resonance image of lumbar spine with spondylolisthesis",
    ]


def label_index_to_name(idx: int) -> str:
    """Convert RSNA global label index (0-8) to 'condition/severity' string."""
    cond_idx, sev_idx = divmod(idx, len(RSNA_SEVERITY_CLASSES))
    if cond_idx >= len(RSNA_CONDITIONS):
        raise IndexError(f"label index {idx} out of range for RSNA (max 8)")
    return f"{RSNA_CONDITIONS[cond_idx]}/{RSNA_SEVERITY_CLASSES[sev_idx]}"
```

- [ ] **2.4: Run test to verify pass**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_label_prompts.py -v
```

Expected: 5 passed.

- [ ] **2.5: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add spinenet/label_prompts.py tests/test_label_prompts.py && git commit -m "feat: add label prompt templates for RSNA and SPIDER zero-shot

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Pre-compute BiomedCLIP label embeddings

**Files:**
- Create: `build_label_embeddings.py`
- Create: `tests/test_label_embeddings.py`

- [ ] **3.1: Write smoke test**

Create `/Users/kienha/spinet-v2/tests/test_label_embeddings.py`:

```python
"""Smoke test for BiomedCLIP label embedding builder.

This test requires network access to download BiomedCLIP weights (one-time,
~500MB). If offline, skip by setting PYTEST_SKIP_BIOMEDCLIP=1.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    os.environ.get("PYTEST_SKIP_BIOMEDCLIP") == "1",
    reason="BiomedCLIP download skipped (set PYTEST_SKIP_BIOMEDCLIP=0 to enable)",
)
def test_build_label_embeddings_produces_correct_shape(tmp_path):
    out_file = tmp_path / "label_embeddings.pt"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "build_label_embeddings.py"),
            "--label-set", "rsna",
            "--output", str(out_file),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_file.exists(), "output file not created"

    embs = torch.load(out_file)
    assert embs.shape[0] == 9, f"expected 9 RSNA labels, got {embs.shape[0]}"
    assert embs.shape[1] > 0, "embedding dim must be > 0"
    # L2 normalized (unit norm).
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "embeddings should be L2-normalized"
```

- [ ] **3.2: Run test to verify it fails**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_label_embeddings.py -v
```

Expected: FileNotFoundError or ModuleNotFoundError.

- [ ] **3.3: Implement build_label_embeddings.py**

Create `/Users/kienha/spinet-v2/build_label_embeddings.py`:

```python
"""Pre-compute BiomedCLIP text embeddings for label prompts.

Run once before training. Embeddings are cached to disk so training loop does
not need to load BiomedCLIP text encoder again.

Usage:
    python3 build_label_embeddings.py --label-set rsna --output embs.pt
    python3 build_label_embeddings.py --label-set spider --output spider_embs.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from spinenet.label_prompts import build_rsna_prompts, build_spider_unseen_prompts


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def encode_prompts(prompts: list[str], model_name: str, device: torch.device) -> torch.Tensor:
    """Encode a list of text prompts via frozen BiomedCLIP text encoder.

    Returns: [num_prompts, text_dim] L2-normalized tensor.
    """
    from open_clip import create_model_from_pretrained, get_tokenizer

    model, _ = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    return text_features.cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-set", choices=["rsna", "spider"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.label_set == "rsna":
        prompts = build_rsna_prompts()
    else:
        prompts = build_spider_unseen_prompts()

    print(f"Encoding {len(prompts)} {args.label_set} prompts via {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embs = encode_prompts(prompts, args.model, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embs, args.output)
    print(f"Saved {embs.shape} tensor to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **3.4: Run builder for RSNA**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 build_label_embeddings.py --label-set rsna --output experiments/paper_results/label_embeddings_rsna.pt
```

Expected: First run downloads BiomedCLIP (~500MB, may take 2-5 min). Then writes `[9, 512]` tensor. If encounters HuggingFace auth issue, try `huggingface-cli login` first.

- [ ] **3.5: Run builder for SPIDER**

```bash
cd /Users/kienha/spinet-v2 && python3 build_label_embeddings.py --label-set spider --output experiments/paper_results/label_embeddings_spider.pt
```

Expected: `[9, 512]` tensor for SPIDER unseen labels.

- [ ] **3.6: Run smoke test**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_label_embeddings.py -v
```

Expected: 1 passed (after model cached from step 3.4).

- [ ] **3.7: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add build_label_embeddings.py tests/test_label_embeddings.py experiments/paper_results/label_embeddings_rsna.pt experiments/paper_results/label_embeddings_spider.pt && git commit -m "feat: build BiomedCLIP label embeddings for RSNA and SPIDER

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Implement SpineNetCBAMWithBiomedCLIP model

**Files:**
- Create: `spinenet/models/grading_biomedclip.py`
- Test: `tests/test_grading_biomedclip.py`
- Modify: `spinenet/models/__init__.py`

- [ ] **4.1: Write failing test for model shape and API**

Create `/Users/kienha/spinet-v2/tests/test_grading_biomedclip.py`:

```python
"""Unit tests for SpineNetCBAMWithBiomedCLIP."""
import pytest
import torch
import torch.nn.functional as F

from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP


def test_model_forward_returns_logits_with_correct_shape():
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9, freeze_backbone=True).eval()
    # Populate label embeddings with random unit vectors (real use: load from disk).
    model.set_label_embeddings(F.normalize(torch.randn(9, 512), dim=-1))

    x = torch.randn(2, 1, 9, 112, 224)
    with torch.no_grad():
        out = model(x)

    assert "logits" in out and "per_condition" in out
    assert out["logits"].shape == (2, 9)
    for key, tensor in out["per_condition"].items():
        assert tensor.shape == (2, 3), f"expected (2, 3) for {key}, got {tensor.shape}"


def test_add_new_labels_extends_embedding_matrix():
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9, freeze_backbone=True).eval()
    model.set_label_embeddings(F.normalize(torch.randn(9, 512), dim=-1))

    assert model.label_embeddings.shape == (9, 512)
    new_embs = F.normalize(torch.randn(4, 512), dim=-1)
    model.add_new_labels(new_embs)
    assert model.label_embeddings.shape == (13, 512)

    x = torch.randn(1, 1, 9, 112, 224)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape == (1, 13)


def test_freeze_backbone_sets_requires_grad():
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9, freeze_backbone=True)
    backbone_trainable = [
        p for name, p in model.named_parameters()
        if (name.startswith("layer") or name.startswith("conv1") or name.startswith("bn1"))
    ]
    for p in backbone_trainable:
        assert not p.requires_grad, "backbone params should be frozen"

    projection_trainable = [
        p for name, p in model.named_parameters() if name.startswith("image_projection")
    ]
    for p in projection_trainable:
        assert p.requires_grad, "projection should be trainable"
```

- [ ] **4.2: Run test to verify it fails**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_grading_biomedclip.py -v
```

Expected: ImportError (module does not exist yet).

- [ ] **4.3: Implement the model class**

Create `/Users/kienha/spinet-v2/spinenet/models/grading_biomedclip.py`:

```python
"""CBAM ResNet34 3D backbone aligned with BiomedCLIP text embeddings for zero-shot grading."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spinenet.models.grading_baseline import BasicBlock, conv1x1
from spinenet.models.attention import CBAM


class SpineNetCBAMWithBiomedCLIP(nn.Module):
    """3D ResNet34 + CBAM image encoder aligned with BiomedCLIP text embeddings.

    Args:
        num_labels: number of label embedding slots at init. Can grow via add_new_labels().
        text_dim: dimension of BiomedCLIP text embeddings (512 for PubMedBERT_256-vit_base).
        freeze_backbone: if True, freezes backbone + CBAM; only projection + temperature train.
        cbam_reduction: CBAM channel reduction ratio (matches existing GradingModelWithCBAM).

    Forward input: [B, 1, 9, 112, 224] IVV volume.
    Forward output: dict with keys:
        "logits": [B, num_labels] raw similarity scores (scaled by learnable temperature)
        "per_condition": dict[str, tensor] — for backward compatibility with existing
          RSNA 3-condition evaluator. Keys: "spinal_canal", "left_foraminal",
          "right_foraminal". Each tensor shape [B, 3].
    """

    def __init__(
        self,
        num_labels: int = 9,
        text_dim: int = 512,
        freeze_backbone: bool = True,
        cbam_reduction: int = 16,
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.text_dim = text_dim

        # Backbone: 3D ResNet34 + CBAM (matches GradingModelWithCBAM).
        self._norm_layer = nn.BatchNorm3d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
            padding=(1, 3, 3), bias=False,
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
        )

        self.layer1 = self._make_layer(BasicBlock, 64, 3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=1)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.cbam1 = CBAM(in_channels=64, reduction=cbam_reduction)
        self.cbam2 = CBAM(in_channels=128, reduction=cbam_reduction)
        self.cbam3 = CBAM(in_channels=256, reduction=cbam_reduction)
        self.cbam4 = CBAM(in_channels=512, reduction=cbam_reduction)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Image projection: maps 512-dim backbone output to text embedding space.
        self.image_projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, text_dim),
        )

        # Learnable temperature for contrastive scaling (CLIP convention).
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

        # Label embeddings (populated from cached file via set_label_embeddings).
        # Registered as buffer so it moves with .to(device).
        self.register_buffer(
            "label_embeddings",
            torch.zeros(num_labels, text_dim),
            persistent=False,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, 1, 64, 1, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze backbone + CBAM; keep projection + temperature trainable."""
        for name, param in self.named_parameters():
            should_freeze = (
                name.startswith("conv1")
                or name.startswith("bn1")
                or name.startswith("layer")
                or name.startswith("cbam")
            )
            if should_freeze:
                param.requires_grad = not freeze

    def set_label_embeddings(self, text_embeddings: torch.Tensor) -> None:
        """Replace label embeddings (e.g., after loading cached .pt file)."""
        assert text_embeddings.dim() == 2 and text_embeddings.shape[1] == self.text_dim
        self.label_embeddings = text_embeddings.to(self.label_embeddings.device).float()
        self.num_labels = text_embeddings.shape[0]

    def add_new_labels(self, new_text_embeddings: torch.Tensor) -> None:
        """Append new label embeddings without retraining. Supports zero-shot extension."""
        assert new_text_embeddings.shape[1] == self.text_dim
        self.label_embeddings = torch.cat(
            [
                self.label_embeddings,
                new_text_embeddings.to(self.label_embeddings.device).float(),
            ],
            dim=0,
        )
        self.num_labels = self.label_embeddings.shape[0]

    def _extract_image_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.cbam1(x)
        x = self.layer2(x)
        x = self.cbam2(x)
        x = self.layer3(x)
        x = self.cbam3(x)
        x = self.layer4(x)
        x = self.cbam4(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, volume: torch.Tensor) -> dict:
        img_feat = self._extract_image_features(volume)
        img_emb = self.image_projection(img_feat)
        img_emb = F.normalize(img_emb, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = logit_scale * (img_emb @ self.label_embeddings.T)  # [B, num_labels]

        # Build per-condition dict for backward compat with RSNA evaluator.
        # Assumes num_labels == 9 and layout (cond_idx * 3 + sev_idx) per label_prompts.py.
        per_condition = {}
        if self.num_labels == 9:
            per_condition["spinal_canal"] = logits[:, 0:3]
            per_condition["left_foraminal"] = logits[:, 3:6]
            per_condition["right_foraminal"] = logits[:, 6:9]

        return {"logits": logits, "per_condition": per_condition}

    def load_cbam_checkpoint(self, checkpoint_path: str, verbose: bool = True) -> None:
        """Load weights from existing best_model_attention.pth.

        Only loads matching keys for backbone + CBAM. Skips old Linear heads
        (fc_spinal_canal / fc_left_foraminal / fc_right_foraminal) because they
        no longer exist in this model.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)

        own_keys = set(self.state_dict().keys())
        loaded = {}
        skipped = []
        for k, v in state.items():
            if k.startswith("fc_"):
                skipped.append(k)
                continue
            if k in own_keys and v.shape == self.state_dict()[k].shape:
                loaded[k] = v
            else:
                skipped.append(k)

        self.load_state_dict(loaded, strict=False)
        if verbose:
            print(f"Loaded {len(loaded)} keys from {checkpoint_path}")
            if skipped:
                print(f"Skipped {len(skipped)} keys (legacy heads or mismatched): {skipped[:5]}...")
```

- [ ] **4.4: Run tests to verify pass**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_grading_biomedclip.py -v
```

Expected: 3 passed.

- [ ] **4.5: Update spinenet/models/__init__.py**

Add these lines to `/Users/kienha/spinet-v2/spinenet/models/__init__.py` (check current content first and append):

```python
from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP  # noqa: F401
```

- [ ] **4.6: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add spinenet/models/grading_biomedclip.py spinenet/models/__init__.py tests/test_grading_biomedclip.py && git commit -m "feat: add SpineNetCBAMWithBiomedCLIP model

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Verify checkpoint loading integration

**Files:**
- Modify: `tests/test_grading_biomedclip.py` (add integration test)

- [ ] **5.1: Add checkpoint loading test**

Append to `/Users/kienha/spinet-v2/tests/test_grading_biomedclip.py`:

```python
import os
from pathlib import Path

CHECKPOINT_PATH = Path("/Users/kienha/spinet-v2/checkpoints/best_model_attention.pth")


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="CBAM checkpoint not available")
def test_load_cbam_checkpoint_warm_starts_backbone():
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9, freeze_backbone=True)

    # Save random backbone state for comparison.
    random_conv1 = model.conv1.weight.clone()

    model.load_cbam_checkpoint(str(CHECKPOINT_PATH), verbose=False)

    # After load, backbone weights should differ from random init.
    loaded_conv1 = model.conv1.weight
    assert not torch.allclose(random_conv1, loaded_conv1), "conv1 should have been loaded"
```

- [ ] **5.2: Run test**

```bash
cd /Users/kienha/spinet-v2 && python3 -m pytest tests/test_grading_biomedclip.py::test_load_cbam_checkpoint_warm_starts_backbone -v
```

Expected: 1 passed (or skipped if checkpoint missing).

- [ ] **5.3: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add tests/test_grading_biomedclip.py && git commit -m "test: verify CBAM checkpoint loads into new model

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Implement training script

**Files:**
- Create: `train_biomedclip.py`

**Convention alignment** (from existing `train_rsna_baseline.py` / `train_rsna_attention.py`):
- Default `--data-dir rsna_preprocessed`
- Save best model to `checkpoints/best_model_biomedclip.pth` (flat, not in subdirectory)
- Support `--resume <checkpoint_path>` for mid-run recovery
- Print per-class metrics every 5 epochs like other scripts
- `--early-stop-patience 15`, `--save-freq 5`
- Match val_accuracies dict structure so existing `test_rsna_preprocessed.py` can load it

- [ ] **6.1: Write training script**

Create `/Users/kienha/spinet-v2/train_biomedclip.py`:

```python
"""Train SpineNetCBAMWithBiomedCLIP via contrastive alignment on RSNA.

Trainable components:
- image_projection (~262K params)
- logit_scale (1 param)

Frozen: CBAM backbone (loaded from best_model_attention.pth) + BiomedCLIP text
encoder (pre-computed into label_embeddings_rsna.pt).

Usage:
    python3 train_biomedclip.py \\
        --cbam-checkpoint checkpoints/best_model_attention.pth \\
        --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \\
        --epochs 30 --batch-size 32 --lr 1e-3
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP


RSNA_CONDITIONS = ("spinal_canal", "left_foraminal", "right_foraminal")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def per_condition_labels_to_global_index(label_dict: dict) -> torch.Tensor:
    """Convert per-condition class labels [B] into global label indices [3*B].

    Each sample produces 3 training examples (one per condition) with ignore_index
    for missing labels.

    Output shape: [3B] tensor of global label indices (0-8), with -1 for missing.
    """
    B = label_dict["spinal_canal"].shape[0]
    global_labels = torch.full((3 * B,), -1, dtype=torch.long)
    for cond_idx, cond in enumerate(RSNA_CONDITIONS):
        cls = label_dict[cond]  # [B]
        for b in range(B):
            c = cls[b].item()
            if c == -1:
                continue
            global_labels[cond_idx * B + b] = cond_idx * 3 + c
    return global_labels


def expand_batch_for_all_conditions(volumes: torch.Tensor) -> torch.Tensor:
    """Repeat each volume 3 times so every condition gets a training example.

    Input: [B, 1, 9, 112, 224]
    Output: [3B, 1, 9, 112, 224]
    """
    return volumes.repeat(3, 1, 1, 1, 1)


def evaluate(model, loader, device) -> dict:
    """Compute per-condition accuracy + macro F1 on val set."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    model.eval()
    all_preds = {c: [] for c in RSNA_CONDITIONS}
    all_labels = {c: [] for c in RSNA_CONDITIONS}

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Eval", leave=False):
            volumes = volumes.unsqueeze(1).to(device)
            out = model(volumes)
            for cond in RSNA_CONDITIONS:
                preds = torch.argmax(out["per_condition"][cond], dim=1).cpu().numpy()
                all_preds[cond].extend(preds.tolist())
                all_labels[cond].extend(labels[cond].numpy().tolist())

    results = {}
    for cond in RSNA_CONDITIONS:
        y_true = np.array(all_labels[cond])
        y_pred = np.array(all_preds[cond])
        mask = y_true != -1
        if mask.sum() == 0:
            results[cond] = {"acc": 0.0, "macro_f1": 0.0}
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[mask], y_pred[mask], labels=[0, 1, 2], average="macro", zero_division=0,
        )
        results[cond] = {"acc": acc, "macro_f1": f1}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbam-checkpoint", type=Path, required=True)
    parser.add_argument("--label-embeddings", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("rsna_preprocessed"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from (e.g., mid-run interruption)")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset.
    full = RSNAPreprocessedDataset(data_dir=str(args.data_dir), split="train", transform=None)
    patients = full.metadata["study_id"].unique()
    train_p, val_p = train_test_split(patients, test_size=args.val_split, random_state=args.seed)
    train_idx = full.metadata[full.metadata["study_id"].isin(train_p)].index.tolist()
    val_idx = full.metadata[full.metadata["study_id"].isin(val_p)].index.tolist()

    train_loader = DataLoader(
        Subset(full, train_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full, val_idx),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Build model and load CBAM checkpoint.
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9, freeze_backbone=True)
    model.load_cbam_checkpoint(str(args.cbam_checkpoint), verbose=True)
    model.set_label_embeddings(torch.load(args.label_embeddings))
    model.to(device)

    # Resume from checkpoint if requested.
    start_epoch = 0
    if args.resume is not None and Path(args.resume).exists():
        print(f"Resuming from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=False)
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"  -> starting from epoch {start_epoch}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable)
    print(f"Model params: {trainable_count:,} trainable / {total:,} total ({100 * trainable_count / total:.2f}%)")

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.freeze_backbone(freeze=True)  # ensure BN stays in eval-ish state via hooks if needed
        running_loss = 0.0
        n_batches = 0
        for volumes, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            volumes = volumes.unsqueeze(1).to(device)
            expanded = expand_batch_for_all_conditions(volumes)
            global_labels = per_condition_labels_to_global_index(labels).to(device)

            out = model(expanded)
            loss = F.cross_entropy(out["logits"], global_labels, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train = running_loss / max(n_batches, 1)
        val_results = evaluate(model, val_loader, device)
        val_acc_mean = np.mean([v["acc"] for v in val_results.values()])
        val_f1_mean = np.mean([v["macro_f1"] for v in val_results.values()])

        scheduler.step(-val_f1_mean)  # maximize f1
        print(f"\nEpoch {epoch+1}: train_loss={avg_train:.4f}, val_acc_mean={val_acc_mean:.3f}, val_f1_mean={val_f1_mean:.3f}")
        for cond, m in val_results.items():
            print(f"  {cond}: acc={m['acc']:.3f}, macro_f1={m['macro_f1']:.3f}")

        # Save best by validation macro-F1 (not train loss) — Severe recall is our objective.
        is_best = val_f1_mean > (-best_val_loss)  # we store -f1 as "loss" for scheduler compat
        if is_best:
            best_val_loss = -val_f1_mean
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_path = args.save_dir / "best_model_biomedclip.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracies": {c: v["acc"] for c, v in val_results.items()},
                "val_macro_f1": val_f1_mean,
                "val_results": val_results,
                "args": vars(args),
            }, save_path)
            print(f"Saved best model to {save_path} (epoch {best_epoch}, macro_f1={val_f1_mean:.3f})")
        else:
            epochs_without_improvement += 1

        # Periodic checkpoint every save_freq epochs.
        if (epoch + 1) % args.save_freq == 0:
            periodic_path = args.save_dir / f"checkpoint_biomedclip_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1_mean,
            }, periodic_path)

        # Early stopping.
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {args.early_stop_patience} epochs)")
            break


if __name__ == "__main__":
    main()
```

- [ ] **6.2: Dry-run locally (confirm code compiles and dataloader works)**

If RTX 4090/5090 available locally:
```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 train_biomedclip.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --epochs 1 --batch-size 4 --num-workers 0
```

Or CPU-only smoke (very slow, only for structural validation — kill after 1-2 batches):
```bash
CUDA_VISIBLE_DEVICES="" python3 train_biomedclip.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --epochs 1 --batch-size 2 --num-workers 0
```

Expected: Training starts, logs batch loss, completes 1 epoch, saves checkpoint. If crashes at dataloader, check RSNAPreprocessedDataset label keys match RSNA_CONDITIONS.

- [ ] **6.3: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add train_biomedclip.py && git commit -m "feat: add BiomedCLIP contrastive training script

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Full training on Vast.ai GPU instance

**Files:** none new. Runs `train_biomedclip.py` on rented GPU.

### 7.A — Provision Vast.ai instance

- [ ] **7.A.1: Push branch to GitHub before renting instance**

```bash
cd /Users/kienha/spinet-v2 && git push -u origin biomedclip-integration
```

- [ ] **7.A.2: Rent GPU instance on Vast.ai**

Recommended specs:
- GPU: RTX 4090 (24GB) or RTX 5090 (32GB) — user's preferred hardware
- Disk: ≥50GB (for rsna_preprocessed 8GB + BiomedCLIP 500MB + checkpoints)
- Image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` or similar with CUDA

Note the `SSH IP` and `SSH Port` Vast shows after launch.

- [ ] **7.A.3: SSH into instance and start tmux**

```bash
# On local machine:
ssh -p <PORT> root@<VAST_IP>

# Inside instance:
tmux new -s train
tmux set -g mouse on
```

### 7.B — Setup environment on instance

- [ ] **7.B.1: Clone + checkout branch**

```bash
cd /root
git clone https://github.com/trungkien04102002/spinet-v2.git
cd spinet-v2
./vast_setup.sh --branch biomedclip-integration
pip3 install -r requirements.txt
```

Expected: Clone succeeds, `vast_setup.sh` checks out `biomedclip-integration` branch, pip installs including new `open_clip_torch` dependency.

- [ ] **7.B.2: Download preprocessed data**

```bash
chmod +x 2_download_preprocessed.sh
./2_download_preprocessed.sh <GDRIVE_PREPROCESSED_ID>
```

Where `<GDRIVE_PREPROCESSED_ID>` is the same Google Drive ID user used before (e.g., `1DCndO_ppTDMGT1H19b8XIqfureMqABi_`).

Expected: Creates `rsna_preprocessed/` directory with volumes + metadata.

- [ ] **7.B.3: Download pretrained backbone weights**

```bash
./3_download_weights.sh <GDRIVE_WEIGHTS_ID>
```

Expected: Downloads `ckpt1.pt` backbone weights to `~/.spinenet/weights/`.

- [ ] **7.B.4: Copy existing CBAM checkpoint to instance**

Option 1 (from Google Drive or similar):
```bash
# If checkpoint is on cloud storage
wget -O checkpoints/best_model_attention.pth <URL>
```

Option 2 (from local laptop):
```bash
# On local machine, in another terminal:
scp -P <PORT> checkpoints/best_model_attention.pth root@<VAST_IP>:/root/spinet-v2/checkpoints/
```

Expected: File at `/root/spinet-v2/checkpoints/best_model_attention.pth`.

- [ ] **7.B.5: Verify GPU**

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: Shows GPU name, CUDA True.

### 7.C — Pre-compute label embeddings + smoke test

- [ ] **7.C.1: Build label embeddings on instance (SKIP if already committed to repo from Task 3)**

```bash
# Check first — if files exist from git pull, skip this step:
ls -la experiments/paper_results/label_embeddings_*.pt

# Otherwise build them:
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 build_label_embeddings.py --label-set rsna --output experiments/paper_results/label_embeddings_rsna.pt
python3 build_label_embeddings.py --label-set spider --output experiments/paper_results/label_embeddings_spider.pt
```

Expected: Downloads BiomedCLIP weights (~500MB, 2-5 min first time), produces 2 .pt files.
If Task 3 ran locally and embeddings were committed, `git pull` already has them.

- [ ] **7.C.2: Smoke test (1 epoch, small batch)**

```bash
python3 train_biomedclip.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --epochs 1 --batch-size 16 --num-workers 4
```

Expected: 1 epoch completes in ~10-30 min depending on dataset size. Checkpoint saved. If crashes → fix before full run.

### 7.D — Full training run

- [ ] **7.D.1: Start full training inside tmux**

```bash
mkdir -p experiments/paper_results/logs
python3 train_biomedclip.py \
    --cbam-checkpoint checkpoints/best_model_attention.pth \
    --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --epochs 30 --batch-size 32 --lr 1e-3 \
    --save-dir checkpoints \
    2>&1 | tee experiments/paper_results/logs/biomedclip_train.log
```

Expected on RTX 4090: ~15-20 GPU-hours total (~40-50 min/epoch × 30 epochs, or faster if early stopping triggers).

- [ ] **7.D.2: Detach tmux (training continues in background)**

```bash
# Press: Ctrl+B then D
# Now safe to close SSH. Training keeps running.
```

- [ ] **7.D.3: Reattach to check progress periodically**

```bash
# When reconnecting:
ssh -p <PORT> root@<VAST_IP>
tmux attach -t train
```

### 7.E — After training completes

- [ ] **7.E.1: Verify final checkpoint on instance**

```bash
ls -la /root/spinet-v2/checkpoints/best_model_biomedclip.pth
```

Expected: File exists, ~100-110MB (backbone + projection + buffer for label_embeddings).

- [ ] **7.E.2: Copy checkpoint + log back to local machine**

```bash
# On local machine:
scp -P <PORT> root@<VAST_IP>:/root/spinet-v2/checkpoints/best_model_biomedclip.pth \
    /Users/kienha/spinet-v2/checkpoints/
scp -P <PORT> root@<VAST_IP>:/root/spinet-v2/experiments/paper_results/logs/biomedclip_train.log \
    /Users/kienha/spinet-v2/experiments/paper_results/logs/
scp -P <PORT> root@<VAST_IP>:/root/spinet-v2/experiments/paper_results/label_embeddings_*.pt \
    /Users/kienha/spinet-v2/experiments/paper_results/
```

Expected: Files copied to local paths. Now evaluation can run locally.

- [ ] **7.E.3: Terminate Vast.ai instance**

Exit tmux, exit SSH, stop + destroy instance on Vast.ai dashboard to avoid continued billing.

- [ ] **7.E.4: Commit log and embeddings locally**

```bash
cd /Users/kienha/spinet-v2 && git add experiments/paper_results/logs/biomedclip_train.log experiments/paper_results/label_embeddings_*.pt
git commit -m "train: BiomedCLIP projection training completed on Vast.ai RTX 4090

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

(Do NOT commit `best_model_biomedclip.pth` — checkpoints are gitignored.)

---

## Task 8: Supervised evaluation on RSNA val

**Files:**
- Create: `evaluate_biomedclip.py`

- [ ] **8.1: Write evaluation script**

Create `/Users/kienha/spinet-v2/evaluate_biomedclip.py`:

```python
"""Evaluate SpineNetCBAMWithBiomedCLIP on RSNA val and perform zero-shot SPIDER evaluation.

Produces per-condition metrics matching experiments/rsna/base_line.txt format,
saved to experiments/paper_results/table2_ablation/ivd_metrics/05_biomedclip.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rsna_preprocessed_dataloader import RSNAPreprocessedDataset
from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP


CONDITIONS = [
    ("spinal_canal", "Spinal Canal Stenosis"),
    ("left_foraminal", "Left Foraminal Narrowing"),
    ("right_foraminal", "Right Foraminal Narrowing"),
]
CLASS_NAMES = ["Normal/Mild", "Moderate", "Severe"]


def collect_predictions(model, loader, device):
    preds = {c: [] for c, _ in CONDITIONS}
    labels = {c: [] for c, _ in CONDITIONS}
    with torch.no_grad():
        for volumes, label_dict in tqdm(loader, desc="Eval"):
            volumes = volumes.unsqueeze(1).to(device)
            out = model(volumes)
            for cond, _ in CONDITIONS:
                p = torch.argmax(out["per_condition"][cond], dim=1).cpu().numpy()
                preds[cond].extend(p.tolist())
                labels[cond].extend(label_dict[cond].numpy().tolist())
    return preds, labels


def format_report(preds, labels) -> str:
    out = ["EVALUATION METRICS", "=" * 70]
    for cond, display in CONDITIONS:
        y_true = np.array(labels[cond])
        y_pred = np.array(preds[cond])
        valid = y_true != -1
        y_true, y_pred = y_true[valid], y_pred[valid]
        if len(y_true) == 0:
            continue
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0,
        )
        out.append(f"\n{display}:")
        out.append(f"  Overall Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        out.append(f"  Total Samples: {len(y_true)}")
        out.append("")
        out.append("  Per-Class Metrics:")
        out.append("  " + "-" * 60)
        out.append(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
        out.append("  " + "-" * 60)
        for i, name in enumerate(CLASS_NAMES):
            out.append(f"  {name:<15} {p[i]:<12.3f} {r[i]:<12.3f} {f1[i]:<12.3f} {int(sup[i])}")
        out.append("  " + "-" * 60)
        out.append(f"  {'Macro Avg':<15} {p.mean():<12.3f} {r.mean():<12.3f} {f1.mean():<12.3f} {int(sup.sum())}")
        out.append("  " + "-" * 60)
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-embeddings", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("rsna_preprocessed"))
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full = RSNAPreprocessedDataset(data_dir=str(args.data_dir), split="train", transform=None)
    patients = full.metadata["study_id"].unique()
    _, val_p = train_test_split(patients, test_size=args.val_split, random_state=args.seed)
    val_idx = full.metadata[full.metadata["study_id"].isin(val_p)].index.tolist()
    loader = DataLoader(
        Subset(full, val_idx),
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )

    model = SpineNetCBAMWithBiomedCLIP(num_labels=9)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.set_label_embeddings(torch.load(args.label_embeddings))
    model.to(device).eval()

    preds, labels = collect_predictions(model, loader, device)
    report = format_report(preds, labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(report)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **8.2: Run evaluation**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 evaluate_biomedclip.py \
    --checkpoint checkpoints/biomedclip/best_model_biomedclip.pth \
    --label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --output experiments/paper_results/table2_ablation/ivd_metrics/05_biomedclip.txt
```

Expected: Prints per-condition metrics matching `01_baseline.txt` format. Severe Recall should be close to CBAM+Focal (~83% for SC, ~30-40% for foraminal) — if much lower, training didn't converge.

- [ ] **8.3: Commit evaluation**

```bash
cd /Users/kienha/spinet-v2 && git add evaluate_biomedclip.py experiments/paper_results/table2_ablation/ivd_metrics/05_biomedclip.txt && git commit -m "eval: BiomedCLIP supervised metrics on RSNA val

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Zero-shot evaluation on SPIDER unseen labels

**Files:**
- Create: `evaluate_biomedclip_zeroshot.py`

- [ ] **9.1: Write zero-shot eval script**

Create `/Users/kienha/spinet-v2/evaluate_biomedclip_zeroshot.py`:

```python
"""Zero-shot evaluation: predict SPIDER labels using BiomedCLIP without training on them.

Loads trained SpineNetCBAMWithBiomedCLIP, appends SPIDER label embeddings, runs
inference on SPIDER validation set.

Metric: argmax over SPIDER labels only (ignore RSNA labels for this experiment).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

from spider_dataloader import SPIDERDataset
from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP


# SPIDER prompt index layout (from build_spider_unseen_prompts in label_prompts.py):
# 0-4: Pfirrmann grade 1-5
# 5: disc without herniation, 6: with herniation
# 7: no spondylolisthesis, 8: with spondylolisthesis
SPIDER_LABEL_OFFSETS = {
    "pfirrmann": (0, 5),
    "disc_herniation": (5, 7),
    "spondylolisthesis": (7, 9),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rsna-label-embeddings", type=Path, required=True)
    parser.add_argument("--spider-label-embeddings", type=Path, required=True)
    parser.add_argument("--spider-data-dir", type=Path, default=Path("spider"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RSNA-trained model.
    model = SpineNetCBAMWithBiomedCLIP(num_labels=9)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.set_label_embeddings(torch.load(args.rsna_label_embeddings))
    # Append SPIDER label embeddings — zero-shot, no retraining.
    model.add_new_labels(torch.load(args.spider_label_embeddings))
    model.to(device).eval()
    print(f"Total labels after adding SPIDER: {model.num_labels}")

    ds = SPIDERDataset(data_dir=str(args.spider_data_dir), split="training", modality="t2")
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2, pin_memory=True)

    all_preds = {task: [] for task in SPIDER_LABEL_OFFSETS}
    all_labels = {task: [] for task in SPIDER_LABEL_OFFSETS}

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Zero-shot"):
            volumes = volumes.unsqueeze(1).to(device)
            out = model(volumes)
            logits = out["logits"]  # [B, 9 + 9 = 18]

            for task, (lo, hi) in SPIDER_LABEL_OFFSETS.items():
                # Only consider SPIDER label range: [9 + lo, 9 + hi).
                task_logits = logits[:, 9 + lo : 9 + hi]
                preds = torch.argmax(task_logits, dim=1).cpu().numpy()
                all_preds[task].extend(preds.tolist())
                all_labels[task].extend(labels[task].numpy().tolist())

    lines = ["ZERO-SHOT EVALUATION ON SPIDER", "=" * 70]
    for task, (lo, hi) in SPIDER_LABEL_OFFSETS.items():
        y_true = np.array(all_labels[task])
        y_pred = np.array(all_preds[task])
        acc = accuracy_score(y_true, y_pred)
        num_classes = hi - lo
        p, r, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0,
        )
        lines.append(f"\n{task}:")
        lines.append(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        lines.append(f"  Macro F1: {f1.mean():.4f}")
        for i in range(num_classes):
            lines.append(f"  Class {i}: P={p[i]:.3f} R={r[i]:.3f} F1={f1[i]:.3f} Support={int(sup[i])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **9.2: Run zero-shot evaluation**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 evaluate_biomedclip_zeroshot.py \
    --checkpoint checkpoints/biomedclip/best_model_biomedclip.pth \
    --rsna-label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --spider-label-embeddings experiments/paper_results/label_embeddings_spider.pt \
    --output experiments/paper_results/spider_zeroshot.txt
```

Expected: Produces metrics for pfirrmann, herniation, spondy. Even low accuracy (30-50%) is a meaningful result since labels were never trained.

- [ ] **9.3: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add evaluate_biomedclip_zeroshot.py experiments/paper_results/spider_zeroshot.txt && git commit -m "eval: zero-shot SPIDER evaluation using BiomedCLIP text embeddings

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Consolidate results into paper document

**Files:**
- Create: `experiments/paper_results/biomedclip_results.md`

- [ ] **10.1: Write results report**

Create `/Users/kienha/spinet-v2/experiments/paper_results/biomedclip_results.md`:

```markdown
# CBAM + Focal Loss + BiomedCLIP Integration — Results

**Training setup:**
- Backbone: 3D ResNet34 + CBAM, frozen (loaded from `best_model_attention.pth`)
- Text encoder: BiomedCLIP (frozen), PubMedBERT_256-vit_base_patch16_224
- Trainable: image_projection (~262K params) + logit_scale (1 param)
- Loss: Cross-entropy over cosine similarity (equivalent to InfoNCE)
- Optimizer: AdamW, lr=1e-3, wd=1e-4, batch=32, 30 epochs

## Supervised — RSNA val set (per-IVV)

Fill in from `table2_ablation/ivd_metrics/05_biomedclip.txt`:

| Condition | Overall Acc | Severe Recall | Severe F1 | Macro F1 |
|---|---|---|---|---|
| Spinal Canal | XX.XX% | XX.X% | XX.X% | XX.X% |
| Left Foraminal | XX.XX% | XX.X% | XX.X% | XX.X% |
| Right Foraminal | XX.XX% | XX.X% | XX.X% | XX.X% |

**Comparison with baseline + CBAM+Focal (Variant D from Phase 2):**
Expected: within ±3% of Variant D Macro F1, demonstrating that replacing fixed
Linear heads with BiomedCLIP-aligned embedding does not hurt supervised performance.

## Zero-shot — SPIDER unseen labels

Fill from `spider_zeroshot.txt`:

| Task | Num classes | Accuracy | Macro F1 |
|---|---|---|---|
| Pfirrmann grade | 5 | XX.XX% | XX.X% |
| Disc herniation | 2 | XX.XX% | XX.X% |
| Spondylolisthesis | 2 | XX.XX% | XX.X% |

**Interpretation:** Random chance is 20% (pfirrmann) or 50% (binary). Any
significant performance above chance demonstrates zero-shot transfer via
BiomedCLIP text alignment. Key finding to report honestly.

## Ablation narrative

Table comparing 4 variants on Spinal Canal Severe F1:
| Variant | Severe Recall | Severe F1 | Label-space flexibility |
|---|---|---|---|
| Baseline (ResNet34 + CE) | 31.2% | 40.8% | ❌ fixed |
| + CBAM + Focal | 83.5% | 51.2% | ❌ fixed |
| + CBAM + Focal + BiomedCLIP (ours) | XX.X% | XX.X% | ✅ **supports zero-shot** |

## Reproduction

```bash
python3 build_label_embeddings.py --label-set rsna --output experiments/paper_results/label_embeddings_rsna.pt
python3 build_label_embeddings.py --label-set spider --output experiments/paper_results/label_embeddings_spider.pt
python3 train_biomedclip.py --cbam-checkpoint checkpoints/best_model_attention.pth --label-embeddings experiments/paper_results/label_embeddings_rsna.pt
python3 evaluate_biomedclip.py --checkpoint checkpoints/biomedclip/best_model_biomedclip.pth --label-embeddings experiments/paper_results/label_embeddings_rsna.pt --output experiments/paper_results/table2_ablation/ivd_metrics/05_biomedclip.txt
python3 evaluate_biomedclip_zeroshot.py --checkpoint checkpoints/biomedclip/best_model_biomedclip.pth --rsna-label-embeddings experiments/paper_results/label_embeddings_rsna.pt --spider-label-embeddings experiments/paper_results/label_embeddings_spider.pt --output experiments/paper_results/spider_zeroshot.txt
```

## Limitations

- Single seed (reproducibility weak — add seeds 123, 2024 if time allows).
- BiomedCLIP is pretrained on 2D PubMed images. Our 3D IVV volumes are flattened
  via CBAM backbone, which may lose 3D context. Results should be interpreted
  accordingly.
- SPIDER zero-shot accuracy depends heavily on prompt phrasing. Alternative
  prompts may yield different numbers.
```

- [ ] **10.2: Fill in real numbers from the output files**

```bash
cd /Users/kienha/spinet-v2 && cat experiments/paper_results/table2_ablation/ivd_metrics/05_biomedclip.txt experiments/paper_results/spider_zeroshot.txt
```

Paste numbers into the template above, replacing `XX.XX%` placeholders.

- [ ] **10.3: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add experiments/paper_results/biomedclip_results.md && git commit -m "docs: consolidate BiomedCLIP results report

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11 (OPTIONAL): Few-shot fine-tune fallback

Only do this if zero-shot SPIDER accuracy is below 35% on any task (random chance territory). Demonstrates that with 10-20 samples per new label, projection can adapt quickly.

**Files:**
- Create: `finetune_projection_few_shot.py`

- [ ] **11.1: Implement few-shot script**

Create `/Users/kienha/spinet-v2/finetune_projection_few_shot.py`:

```python
"""Few-shot fine-tune: adapt projection to SPIDER with ≤20 samples per label.

Keeps CBAM + BiomedCLIP frozen, only fine-tunes image_projection for ~10 epochs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm

from spider_dataloader import SPIDERDataset
from spinenet.models.grading_biomedclip import SpineNetCBAMWithBiomedCLIP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rsna-label-embeddings", type=Path, required=True)
    parser.add_argument("--spider-label-embeddings", type=Path, required=True)
    parser.add_argument("--spider-data-dir", type=Path, default=Path("spider"))
    parser.add_argument("--n-per-label", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpineNetCBAMWithBiomedCLIP(num_labels=9)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.set_label_embeddings(torch.load(args.rsna_label_embeddings))
    model.add_new_labels(torch.load(args.spider_label_embeddings))
    model.to(device)

    # Freeze everything except projection.
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("image_projection") or name.endswith("logit_scale")

    ds = SPIDERDataset(data_dir=str(args.spider_data_dir), split="training", modality="t2")

    # Sample n_per_label examples per SPIDER task.
    # For simplicity, sample globally — refinement left for real experiment.
    indices = torch.randperm(len(ds))[: args.n_per_label * 9].tolist()
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=8, shuffle=True)

    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        n = 0
        for volumes, labels in loader:
            volumes = volumes.unsqueeze(1).to(device)
            # Construct global labels across SPIDER label offsets (indices 9..17).
            # Heuristic: just train on pfirrmann head for this demo.
            pfirrmann = labels["pfirrmann"].to(device)  # [B], values 0..4
            target = pfirrmann + 9  # Map to global label index

            out = model(volumes)
            loss = F.cross_entropy(out["logits"], target, ignore_index=-1)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
            n += 1
        print(f"Epoch {epoch+1}: loss={running/max(n,1):.4f}")

    # Save fine-tuned checkpoint.
    torch.save({"model_state_dict": model.state_dict()}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **11.2: Run few-shot experiment**

```bash
cd /Users/kienha/spinet-v2 && python3 finetune_projection_few_shot.py \
    --checkpoint checkpoints/biomedclip/best_model_biomedclip.pth \
    --rsna-label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --spider-label-embeddings experiments/paper_results/label_embeddings_spider.pt \
    --n-per-label 20 --epochs 10 \
    --output checkpoints/biomedclip/few_shot_projection.pth
```

- [ ] **11.3: Re-run zero-shot eval with few-shot checkpoint**

```bash
python3 evaluate_biomedclip_zeroshot.py \
    --checkpoint checkpoints/biomedclip/few_shot_projection.pth \
    --rsna-label-embeddings experiments/paper_results/label_embeddings_rsna.pt \
    --spider-label-embeddings experiments/paper_results/label_embeddings_spider.pt \
    --output experiments/paper_results/spider_fewshot.txt
```

Expected: accuracy higher than pure zero-shot, demonstrating "if zero-shot fails, few-shot works".

- [ ] **11.4: Commit**

```bash
cd /Users/kienha/spinet-v2 && git add finetune_projection_few_shot.py experiments/paper_results/spider_fewshot.txt && git commit -m "feat: few-shot fine-tune fallback for SPIDER labels

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Final commit and push

- [ ] **12.1: Run all tests one last time**

```bash
cd /Users/kienha/spinet-v2 && source spinenet-venv/bin/activate && export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m pytest tests/test_label_prompts.py tests/test_grading_biomedclip.py -v
```

Expected: All tests pass.

- [ ] **12.2: Verify git status is clean**

```bash
cd /Users/kienha/spinet-v2 && git status --short
```

Expected: No modified or untracked files.

- [ ] **12.3: Push branch**

```bash
cd /Users/kienha/spinet-v2 && git push -u origin biomedclip-integration
```

Expected: Branch pushed, GitHub offers PR creation link.

---

## Definition of Done

Plan is complete when:

- [ ] All Tasks 1-10 complete and committed
- [ ] `experiments/paper_results/biomedclip_results.md` has real numbers (no `XX.XX%` placeholders)
- [ ] Supervised RSNA val performance is within ±3% of CBAM+Focal Variant D on Macro F1 (main sanity check)
- [ ] Zero-shot SPIDER evaluation has numbers (even if modest — honesty > hiding)
- [ ] All unit tests pass
- [ ] Branch pushed to GitHub
- [ ] Task 11 (few-shot) complete IF zero-shot result is below 35% on any task

## Known risks

- **BiomedCLIP download fails**: authenticate with `huggingface-cli login`, or download manually from https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Training does not converge**: likely logit_scale diverges. Initialize at `1/0.07` and clamp at `exp(100)` (already in code). If still diverges, lower LR to 5e-4.
- **Zero-shot accuracy ~ random chance**: prompt engineering may help. Try rewording prompts in `label_prompts.py` to match BiomedCLIP training distribution more closely ("lumbar MRI" vs "magnetic resonance image of lumbar spine"). Retry after adjusting.
- **Out of GPU memory at batch 32**: drop to batch 16, adjust LR if needed. RTX 4090 24GB should comfortably fit; if not, check other processes on GPU.
- **RSNAPreprocessedDataset label keys differ from RSNA_CONDITIONS**: inspect `rsna_preprocessed_dataloader.py` __getitem__ and adjust training loop mapping.

## What happens if worst case

If by end of 2 weeks the results are not publishable:
1. Fall back to Option A from `ADVISOR_FEEDBACK_LABELSPACE.md` — keep CBAM + Focal scope, add this BiomedCLIP work as "Future Work" with the code we produced as evidence.
2. The architecture is still a contribution even if numbers are mediocre — demonstrates feasibility for spine MRI.
