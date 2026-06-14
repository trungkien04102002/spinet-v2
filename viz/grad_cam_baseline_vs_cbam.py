"""
Grad-CAM: Baseline (no CBAM) vs CBAM on an RSNA Severe case.

Produces a 3-panel figure: Original | Baseline (no CBAM) | CBAM, for the
spinal-canal Severe case (mid-sagittal, the cleanest to localize).

Key differences vs viz/grad_cam_rsna.py:
  - Compares BASELINE (GradingModelBaseline, no attention) against CBAM,
    instead of CBAM-only vs Hybrid.
  - Targets layer3 / cbam3 (14x28) instead of layer4 (7x14) for sharper
    localization.
  - Per-pixel alpha overlay (only hot regions tinted) so the heatmap sits
    on the lesion instead of washing the whole slice.

Usage:
    python3 viz/grad_cam_baseline_vs_cbam.py \
        --baseline-ckpt checkpoints/rsna/best_model.pth \
        --cbam-ckpt checkpoints/rsna/best_model_attention.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from spinenet.models.grading_baseline import GradingModelBaseline  # noqa: E402
from spinenet.models.grading_attention import GradingModelWithCBAM  # noqa: E402

SEVERE_CLASS = 2

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


class GradCAM3D:
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

    def __call__(self, input_tensor, target_class):
        self.model.zero_grad()
        logits = self.model(input_tensor)  # [1, 3]
        score = logits[0, target_class]
        score.backward()
        acts = self.activations.detach()      # [1, C, D, H, W]
        grads = self.gradients                # [1, C, D, H, W]
        weights = grads.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)  # [D, H, W]
        cam = torch.relu(cam)
        m = cam.max()
        if m > 0:
            cam = cam / m
        return cam.cpu().numpy()

    def close(self):
        for h in self._hooks:
            h.remove()


class CondWrapper(nn.Module):
    def __init__(self, model, condition):
        super().__init__()
        self.model = model
        self.condition = condition

    def forward(self, x):
        return self.model(x)[self.condition]


def load_baseline(ckpt, device):
    model = GradingModelBaseline(format="rsna")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Baseline load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def load_cbam(ckpt, device):
    model = GradingModelWithCBAM(format="rsna", use_cbam=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  CBAM load: {len(missing)} missing, {len(unexpected)} unexpected")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def upsample(cam, hw):
    D, h, w = cam.shape
    H, W = hw
    t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(D, H, W), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()


def overlay(ax, slice_img, cam_2d, title, fs):
    ax.imshow(slice_img, cmap="gray")
    # Smooth + per-pixel alpha: only hot regions tinted.
    cam_s = gaussian_filter(cam_2d, sigma=4)
    if cam_s.max() > 0:
        cam_s = cam_s / cam_s.max()
    alpha = np.clip((cam_s - 0.35) / 0.65, 0, 1) * 0.7  # threshold weak response
    ax.imshow(cam_s, cmap="jet", alpha=alpha)
    ax.set_title(title, fontsize=fs)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", type=Path,
                    default=REPO_ROOT / "checkpoints/rsna/best_model.pth")
    ap.add_argument("--cbam-ckpt", type=Path,
                    default=REPO_ROOT / "checkpoints/rsna/best_model_attention.pth")
    ap.add_argument("--data-dir", type=Path,
                    default=REPO_ROOT / "rsna_preprocessed")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "Master_Thesis_Presentation_Kien/image/grad_cam_baseline_vs_cbam.png")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--canal-only", action="store_true")
    ap.add_argument("--slice", type=int, default=4, help="display slice index (0-8)")
    ap.add_argument("--no-original", action="store_true",
                    help="only 2 panels: Without CBAM | With CBAM")
    args = ap.parse_args()

    cases = CASES[:1] if args.canal_only else CASES
    device = torch.device(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading baseline (no CBAM)...")
    baseline = load_baseline(args.baseline_ckpt, device)
    print("[2/3] Loading CBAM...")
    cbam = load_cbam(args.cbam_ckpt, device)

    print("[3/3] Computing Grad-CAM...")
    fs = 16
    ncols = 2 if args.no_original else 3
    fig, axes = plt.subplots(len(cases), ncols,
                             figsize=(9 if args.no_original else 12,
                                      4.2 if args.canal_only else 11),
                             squeeze=False)

    for row, case in enumerate(cases):
        cond = case["condition"]
        vol_np = np.load(args.data_dir / "volumes" / case["filepath"]).astype(np.float32)
        vol = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)

        cam_b = GradCAM3D(CondWrapper(baseline, cond), baseline.layer3)
        heat_b = cam_b(vol, SEVERE_CLASS)
        cam_b.close()

        cam_c = GradCAM3D(CondWrapper(cbam, cond), cbam.cbam3)
        heat_c = cam_c(vol, SEVERE_CLASS)
        cam_c.close()

        with torch.no_grad():
            pb = torch.softmax(baseline(vol)[cond], -1)[0, SEVERE_CLASS].item()
            pc = torch.softmax(cbam(vol)[cond], -1)[0, SEVERE_CLASS].item()

        hb = upsample(heat_b, (112, 224))
        hc = upsample(heat_c, (112, 224))
        # Report which slice carries most CAM energy (sanity check).
        print(f"  {case['title']}: baseline argmax-slice={hb.sum((1,2)).argmax()}, "
              f"cbam argmax-slice={hc.sum((1,2)).argmax()}, p_base={pb:.2f}, p_cbam={pc:.2f}")

        sl = args.slice
        mid = vol_np[sl]
        if args.no_original:
            overlay(axes[row, 0], mid, hb[sl], f"Without CBAM (Severe p={pb:.2f})", fs)
            overlay(axes[row, 1], mid, hc[sl], f"With CBAM (Severe p={pc:.2f})", fs)
        else:
            axes[row, 0].imshow(mid, cmap="gray")
            axes[row, 0].set_title(case["title"], fontsize=fs)
            axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
            overlay(axes[row, 1], mid, hb[sl], f"Without CBAM (Severe p={pb:.2f})", fs)
            overlay(axes[row, 2], mid, hc[sl], f"With CBAM (Severe p={pc:.2f})", fs)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[done] saved {args.output}")


if __name__ == "__main__":
    main()
