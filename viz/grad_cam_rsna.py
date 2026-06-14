"""
Grad-CAM visualization on RSNA Severe cases.

Generates a 3x3 panel comparing:
    rows    = 3 hand-picked Severe cases (Spinal canal / L-foram / R-foram)
    columns = Original middle slice / CBAM-only Grad-CAM / Hybrid Grad-CAM

The two heatmap columns differ even though both share the same frozen CBAM
backbone weights, because gradients of the target class logit flow through
different downstream paths:
    - CBAM-only: logit = fc(GAP(cbam4))           — direct linear head
    - Hybrid:    logit = scale * (img @ text.T)   — fusion MLP + cosine sim
                 with image = MLP(concat(GAP(cbam4), GAP(BiomedCLIP)))

Output: experiments/v3_20260503/figures/grad_cam_severe.png

Usage:
    python3 viz/grad_cam_rsna.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402
from spinenet.models.grading_hybrid import SpineNetHybrid  # noqa: E402


SEVERE_CLASS = 2

# Hand-picked val-set samples where exactly ONE condition is Severe.
# (Selected from val split with random_state=42, test_size=0.2.)
CASES = [
    {
        "filepath": "145324113/3203860945_l4_l5.npy",
        "condition": "spinal_canal",
        "title": "Spinal canal Severe (L4-L5)",
    },
    {
        "filepath": "416521027/665641441_l5_s1.npy",
        "condition": "left_foraminal",
        "title": "Left foraminal Severe (L5-S1)",
    },
    {
        "filepath": "169651375/2625052739_l5_s1.npy",
        "condition": "right_foraminal",
        "title": "Right foraminal Severe (L5-S1)",
    },
]

# Same prompts as train_rsna_hybrid.py.
RSNA_PROMPTS = {
    "spinal_canal": [
        "normal or mild spinal canal stenosis",
        "moderate spinal canal stenosis",
        "severe spinal canal stenosis",
    ],
    "left_foraminal": [
        "normal or mild left neural foraminal narrowing",
        "moderate left neural foraminal narrowing",
        "severe left neural foraminal narrowing",
    ],
    "right_foraminal": [
        "normal or mild right neural foraminal narrowing",
        "moderate right neural foraminal narrowing",
        "severe right neural foraminal narrowing",
    ],
}


class GradCAM3D:
    """Manual Grad-CAM on a 3D conv layer.

    Hooks forward + backward on the target layer, computes the standard
    Grad-CAM weighted-sum heatmap over channels, returns a 3D heatmap that
    matches the target layer's spatial shape (D, H, W).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)  # [1, 3]
        score = logits[0, target_class]
        score.backward()

        acts = self.activations.detach()  # [1, C, D, H, W]
        grads = self.gradients            # [1, C, D, H, W]
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)  # [D, H, W]
        cam = torch.relu(cam)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.cpu().numpy()

    def close(self):
        for h in self._hooks:
            h.remove()


class CBAMConditionWrapper(nn.Module):
    """Expose 3-class logits for one condition from CBAM-only model."""

    def __init__(self, cbam_model: nn.Module, condition: str):
        super().__init__()
        self.cbam_model = cbam_model
        self.condition = condition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam_model(x)[self.condition]


class HybridConditionWrapper(nn.Module):
    """Expose 3-class logits for one condition from Hybrid model."""

    def __init__(self, hybrid_model: nn.Module, text_embs: torch.Tensor):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.text_embs = text_embs  # [3, 512] L2-normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.hybrid_model(x, self.text_embs)
        return logits


def load_cbam_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def load_hybrid_model(
    cbam_ckpt: Path, hybrid_ckpt: Path, device: torch.device
) -> nn.Module:
    model = SpineNetHybrid(
        cbam_checkpoint_path=str(cbam_ckpt),
        biomedclip_device=str(device),
        slice_strategy="static",
    )
    state = torch.load(hybrid_ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    # Hybrid slim ckpt only contains trainable params (proj + slice_pool +
    # logit_scale). CBAM weights come from the cbam ckpt loaded above.
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Hybrid load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device)
    model.eval()
    # Ensure CBAM grads flow even though the backbone is frozen for training.
    for p in model.cbam.parameters():
        p.requires_grad_(True)
    return model


def upsample_cam(cam: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Upsample [D, h, w] heatmap to [D, H, W] using bilinear per slice."""
    D, h, w = cam.shape
    H, W = target_hw
    t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)  # [1,1,D,h,w]
    t = F.interpolate(t, size=(D, H, W), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def overlay_heatmap(ax, slice_img: np.ndarray, cam_2d: np.ndarray, title: str,
                    fontsize: int = 17):
    ax.imshow(slice_img, cmap="gray")
    ax.imshow(cam_2d, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cbam-ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/v3_20260503/cbam_best.pth",
    )
    ap.add_argument(
        "--hybrid-ckpt",
        type=Path,
        default=REPO_ROOT / "checkpoints/v3_20260503/hybrid_best.pth",
    )
    ap.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "rsna_preprocessed"
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "experiments/v3_20260503/figures/grad_cam_severe.png",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--canal-only",
        action="store_true",
        help="Render only the spinal-canal Severe case (one row, larger fonts) "
        "as the single representative example for the paper.",
    )
    args = ap.parse_args()

    cases = CASES[:1] if args.canal_only else CASES

    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"CBAM ckpt: {args.cbam_ckpt}")
    print(f"Hybrid ckpt: {args.hybrid_ckpt}")

    print("\n[1/4] Loading CBAM-only model...")
    cbam_model = load_cbam_model(args.cbam_ckpt, device)

    print("\n[2/4] Loading Hybrid model (will download BiomedCLIP if first run)...")
    hybrid_model = load_hybrid_model(args.cbam_ckpt, args.hybrid_ckpt, device)

    print("\n[3/4] Pre-computing text embeddings for each condition...")
    text_dbs = {}
    with torch.no_grad():
        for cond, prompts in RSNA_PROMPTS.items():
            text_dbs[cond] = hybrid_model.encode_text(prompts).to(device)
            print(f"  {cond}: {text_dbs[cond].shape}")

    print("\n[4/4] Computing Grad-CAM for each case...")
    fs = 15 if args.canal_only else 17
    fig, axes = plt.subplots(
        len(cases), 3, figsize=(13, 4.4 if args.canal_only else 11),
        squeeze=False,
    )

    for row, case in enumerate(cases):
        cond = case["condition"]
        vol_path = args.data_dir / "volumes" / case["filepath"]
        print(f"\n  Case {row + 1}: {case['title']}")
        print(f"    {vol_path}")
        vol_np = np.load(vol_path).astype(np.float32)  # [9, 112, 224]
        vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)
        vol.requires_grad_(False)

        # CBAM-only Grad-CAM.
        cbam_wrap = CBAMConditionWrapper(cbam_model, cond)
        cam_cbam = GradCAM3D(cbam_wrap, cbam_model.cbam4)
        heat_cbam = cam_cbam(vol, SEVERE_CLASS)  # [9, 7, 14]
        cam_cbam.close()

        # Hybrid Grad-CAM.
        hybrid_wrap = HybridConditionWrapper(hybrid_model, text_dbs[cond])
        cam_hybrid = GradCAM3D(hybrid_wrap, hybrid_model.cbam.cbam4)
        heat_hybrid = cam_hybrid(vol, SEVERE_CLASS)  # [9, 7, 14]
        cam_hybrid.close()

        # Predicted Severe probability for each model (for caption).
        with torch.no_grad():
            cbam_logits = cbam_model(vol)[cond]
            cbam_prob = torch.softmax(cbam_logits, -1)[0, SEVERE_CLASS].item()
            hybrid_logits, _ = hybrid_model(vol, text_dbs[cond])
            hybrid_prob = torch.softmax(hybrid_logits, -1)[0, SEVERE_CLASS].item()

        # Upsample to slice resolution; take middle slice (index 4 of 9).
        cam_cbam_full = upsample_cam(heat_cbam, (112, 224))
        cam_hybrid_full = upsample_cam(heat_hybrid, (112, 224))
        mid_slice = vol_np[4]
        cam_cbam_2d = cam_cbam_full[4]
        cam_hybrid_2d = cam_hybrid_full[4]

        ax_orig = axes[row, 0]
        ax_orig.imshow(mid_slice, cmap="gray")
        ax_orig.set_title(case["title"], fontsize=fs)
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        overlay_heatmap(
            axes[row, 1],
            mid_slice,
            cam_cbam_2d,
            f"CBAM-only (Severe p={cbam_prob:.2f})",
            fontsize=fs,
        )
        overlay_heatmap(
            axes[row, 2],
            mid_slice,
            cam_hybrid_2d,
            f"Hybrid (Severe p={hybrid_prob:.2f})",
            fontsize=fs,
        )

        if row == 0:
            axes[row, 0].set_ylabel("Original (slice 4/9)", fontsize=fs)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n[done] saved {args.output}")


if __name__ == "__main__":
    main()
