#!/usr/bin/env python3
"""Saliency methods for the CBAM 3D grading model, chosen for distinct mechanisms.

Four methods across three mechanisms (see docs/LVTN_phase3/XAI_SURVEY.md):

  grad_cam    Selvaraju et al. 2017. Channel weight = spatially averaged
              gradient. For this model's head (GAP -> single Linear) Draelos &
              Carin prove this equals CAM and HiResCAM at the last conv layer.
  hires_cam   Draelos & Carin 2020. Element-wise grad * activation, no spatial
              averaging. Included to VERIFY that equivalence numerically rather
              than assert it; it is not expected to add information here.
  layer_cam   Jiang et al. 2021. Per-location gradient weighting, which is what
              lets it survive attachment to a shallow layer where plain
              Grad-CAM collapses (43.62% -> 8.87% loc1 in their Table I).
  occlusion   Zeiler & Fergus 2014, coarse grid. The only one here that measures
              the actual score change instead of a gradient proxy, which matters
              because Zhang et al. 2024 show gradient maps can stay stable while
              the model's AUC falls 8.6%.

All return a volume of the same (D, H, W) as the chosen feature map (or the
input, for occlusion), non-negative, normalised to [0, 1].

The model's head is `layer4 -> cbam4 -> avgpool -> fc_<condition>`, one
Linear(512, 3) per condition, verified against spinenet/models/grading_attention.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

# Hookable stages, coarsest last. Spatial sizes are for a (9, 112, 224) input.
STAGES = {
    "layer1": "64ch,  9x28x56",
    "layer2": "128ch, 9x28x56",
    "layer3": "256ch, 9x14x28",
    "layer4": "512ch, 9x7x14",
    "cbam4": "512ch, 9x7x14  (what avgpool actually sees)",
}
METHODS = ("grad_cam", "hires_cam", "layer_cam", "occlusion")


class _Tap:
    """Capture a module's output activation and the gradient flowing back."""

    def __init__(self, module):
        self.activation = None
        self.gradient = None
        self._h = [module.register_forward_hook(self._fwd)]

    def _fwd(self, _m, _inp, out):
        self.activation = out
        # retain_grad works because the tapped tensor is non-leaf and we keep a
        # reference to it; a full backward hook would fire per-input instead.
        out.retain_grad()

    def close(self):
        for h in self._h:
            h.remove()


def _normalise(vol):
    vol = np.maximum(vol, 0.0)
    peak = vol.max()
    return vol / peak if peak > 0 else vol


def _score(model, volume, condition, target_class):
    """The scalar being explained: the target class logit for one condition."""
    logits = model(volume)[condition]
    return logits[0, target_class]


def cam(model, volume, condition, target_class, method="grad_cam",
        stage="cbam4"):
    """Grad-CAM / HiResCAM / Layer-CAM at `stage`.

    The three differ only in how the gradient is turned into a weight:
      grad_cam   w_c = mean_over_space(grad_c),  map = relu(sum_c w_c * A_c)
      hires_cam  map = relu(sum_c grad_c * A_c)          (no averaging)
      layer_cam  map = relu(sum_c relu(grad_c) * A_c)    (per-location, gated)
    """
    if method not in ("grad_cam", "hires_cam", "layer_cam"):
        raise ValueError(f"{method} is not a CAM variant")
    module = dict(model.named_children())[stage]
    tap = _Tap(module)
    try:
        model.zero_grad(set_to_none=True)
        s = _score(model, volume, condition, target_class)
        s.backward()
        act = tap.activation.detach()[0]          # [C, D, H, W]
        grad = tap.activation.grad.detach()[0]    # [C, D, H, W]
    finally:
        tap.close()

    if method == "grad_cam":
        w = grad.mean(dim=(1, 2, 3), keepdim=True)
        m = (w * act).sum(0)
    elif method == "hires_cam":
        m = (grad * act).sum(0)
    else:  # layer_cam
        m = (grad.clamp(min=0) * act).sum(0)
    return _normalise(F.relu(m).cpu().numpy())


@torch.no_grad()
def occlusion(model, volume, condition, target_class, patch=(3, 28, 56),
              stride=None):
    """Coarse-grid occlusion. Cost is the number of grid positions, so the grid
    stays coarse on purpose: He et al. 2025 occlude anatomical regions rather
    than sweeping a dense window, ~247 forwards for a whole brain volume.

    The occluding value is the volume's own mean rather than zero. On MRI a
    zeroed voxel is itself an abnormality, so zeroing measures the model's
    reaction to a black hole as much as to lost evidence (Rong et al. 2022).
    """
    pd_, ph, pw = patch
    sd, sh, sw = stride or patch
    _, _, D, H, W = volume.shape
    fill = float(volume.mean())

    base = _score(model, volume, condition, target_class).item()
    heat = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for d in range(0, D, sd):
        for h in range(0, H, sh):
            for w in range(0, W, sw):
                v = volume.clone()
                v[:, :, d:d + pd_, h:h + ph, w:w + pw] = fill
                drop = base - _score(model, v, condition, target_class).item()
                heat[d:d + pd_, h:h + ph, w:w + pw] += drop
                count[d:d + pd_, h:h + ph, w:w + pw] += 1
    return _normalise(heat / np.maximum(count, 1))


def n_occlusion_passes(shape=(9, 112, 224), patch=(3, 28, 56), stride=None):
    sd, sh, sw = stride or patch
    d, h, w = shape
    return (len(range(0, d, sd)) * len(range(0, h, sh))
            * len(range(0, w, sw))) + 1


def upsample_to_input(vol, out_shape=(9, 112, 224)):
    """Trilinear upsample a coarse CAM to input resolution for overlaying."""
    t = torch.from_numpy(vol)[None, None].float()
    up = F.interpolate(t, size=out_shape, mode="trilinear", align_corners=False)
    return up[0, 0].numpy()


def peak_location(vol):
    """(d, h, w) of the maximum, for point-localisation scoring."""
    return np.unravel_index(int(np.argmax(vol)), vol.shape)


if __name__ == "__main__":
    print("stages:")
    for k, v in STAGES.items():
        print(f"  {k:8s} {v}")
    print(f"\nmethods: {', '.join(METHODS)}")
    print(f"occlusion passes at default grid: {n_occlusion_passes()}")
    print(f"  at (3,14,28) patch:              {n_occlusion_passes(patch=(3,14,28))}")
