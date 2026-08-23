"""
Grading model reimplementing the CORE architecture of the RSNA-2024
2nd-place solution (brendanartley), adapted as a drop-in replacement for
``spinenet.models.grading_baseline.GradingModelBaseline`` so its numbers are
directly comparable to our Table 1.

Reference (read-only, not imported): experiments/sota_comparison/external/
brendanartley/src/models/cnn25d_sagittal_3heads.py and src/models/attn.py.
That repo's real pipeline is per-vertebra / per-condition with 2.5D crops
sampled at inference time from full series; we don't have that raw-DICOM
harness here. Instead we reuse the exact SAME core idea on top of OUR
existing per-IVD (9, 112, 224) tensors, which are already the shared input
representation used by every model in this codebase:

    per-frame 2D CNN encoder (ResNet18, one forward per one of the 9
    sagittal slices) -> BiLSTM over the 9-frame sequence -> attention
    pooling (collapses the sequence to one vector, same formulation as
    external/brendanartley/src/models/attn.py::Attention) -> 3 independent
    linear classification heads (spinal_canal / left_foraminal /
    right_foraminal, 3 classes each).

Interface contract (must match GradingModelBaseline exactly):
    Input:  [B, 1, 9, 112, 224]
    Output (format='rsna'): dict with keys
        'spinal_canal':    [B, 3]
        'left_foraminal':  [B, 3]
        'right_foraminal': [B, 3]
"""

import torch
import torch.nn as nn

try:
    import torchvision

    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_TORCHVISION = False


class _CompactCNN2D(nn.Module):
    """Fallback 2D per-frame encoder used only if torchvision is missing.

    No pretrained weights, no network access required. Roughly a small
    5-stage conv stack producing a fixed-size embedding per frame.
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.out_dim = out_dim

        def block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.stem = block(3, 32, stride=2)     # 112x224 -> 56x112
        self.stage1 = block(32, 64, stride=2)  # -> 28x56
        self.stage2 = block(64, 128, stride=2)  # -> 14x28
        self.stage3 = block(128, 256, stride=2)  # -> 7x14
        self.stage4 = block(256, out_dim, stride=2)  # -> 4x7
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)  # [N, out_dim]


def _build_frame_encoder():
    """Return (encoder_module, emb_dim). Prefers torchvision resnet18
    (weights=None -> no download), falls back to a compact custom CNN.
    """
    if _HAS_TORCHVISION:
        from torchvision.models import resnet18

        backbone = resnet18(weights=None)
        emb_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        return backbone, emb_dim
    return _CompactCNN2D(out_dim=512), 512


class AttentionPool(nn.Module):
    """Attention pooling over the sequence dim.

    Same formulation as external/brendanartley/src/models/attn.py::Attention
    (a 2-layer tanh-MLP scorer + softmax-weighted sum over the sequence).
    """

    def __init__(self, emb_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(attn_dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, x):
        # x: [B, T, D]
        a = self.attn(x)              # [B, T, 1]
        a = torch.softmax(a, dim=1)   # attend over the 9 frames
        pooled = torch.sum(x * a, dim=1)  # [B, D]
        return pooled


class GradingModelBrendanartley(nn.Module):
    """Drop-in replacement for GradingModelBaseline using the
    brendanartley-style 2D-CNN -> BiLSTM -> attention-pooling architecture.

    Input:  [B, 1, 9, 112, 224]
    Output (format='rsna'): dict of 3 logit tensors, each [B, 3].
    """

    def __init__(
        self,
        format: str = "rsna",
        lstm_hidden: int = 256,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        if format != "rsna":
            raise ValueError(
                f"GradingModelBrendanartley only supports format='rsna', got {format!r}"
            )
        self.format = format

        self.frame_encoder, emb_dim = _build_frame_encoder()
        self.emb_dim = emb_dim

        # BiLSTM over the 9-frame sequence. bidirectional -> hidden_dim per
        # direction is lstm_hidden, concatenated output is 2*lstm_hidden.
        # Keep output width == emb_dim so the attention pool + heads below
        # don't need a separate projection (mirrors AttentionLSTM in
        # external/brendanartley/src/models/attn.py where rnn output width
        # == emb_dim).
        assert emb_dim % 2 == 0, "emb_dim must be even for BiLSTM half-split"
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.attn_pool = AttentionPool(emb_dim)
        self.fc_dropout = nn.Dropout(fc_dropout)

        self.fc_spinal_canal = nn.Linear(emb_dim, 3)
        self.fc_left_foraminal = nn.Linear(emb_dim, 3)
        self.fc_right_foraminal = nn.Linear(emb_dim, 3)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, 9, 112, 224]
        Returns:
            dict with keys spinal_canal / left_foraminal / right_foraminal,
            each [B, 3].
        """
        b, c, t, h, w = x.shape
        assert c == 1, f"expected single-channel input, got {c} channels"

        # Treat the 9 sagittal slices as a frame sequence for a 2D encoder.
        # [B, 1, T, H, W] -> [B*T, 1, H, W] -> repeat to 3ch for resnet18.
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        frames = frames.repeat(1, 3, 1, 1)  # [B*T, 3, H, W]

        feats = self.frame_encoder(frames)      # [B*T, emb_dim]
        feats = feats.view(b, t, self.emb_dim)  # [B, T, emb_dim]

        # BiLSTM over the sequence.
        seq, _ = self.rnn(feats)     # [B, T, emb_dim]
        seq = self.layer_norm(seq)

        # Attention pooling -> single vector per sample.
        pooled = self.attn_pool(seq)  # [B, emb_dim]
        pooled = self.fc_dropout(pooled)

        out_spinal_canal = self.fc_spinal_canal(pooled)
        out_left_foraminal = self.fc_left_foraminal(pooled)
        out_right_foraminal = self.fc_right_foraminal(pooled)

        return {
            "spinal_canal": out_spinal_canal,
            "left_foraminal": out_left_foraminal,
            "right_foraminal": out_right_foraminal,
        }

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the per-frame CNN encoder only (mirrors
        GradingModelBaseline.freeze_backbone — heads + LSTM + attention stay
        trainable so a linear-probe-style "Phase 1" run is still possible).
        """
        for param in self.frame_encoder.parameters():
            param.requires_grad = not freeze


# Test code
if __name__ == "__main__":
    print("Testing GradingModelBrendanartley...\n")
    print(f"torchvision available: {_HAS_TORCHVISION}")

    model = GradingModelBrendanartley(format="rsna")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    batch_size = 2
    x = torch.randn(batch_size, 1, 9, 112, 224)
    print(f"Input shape: {x.shape}")

    outputs = model(x)
    print(f"Output type: {type(outputs)} (dict)")
    print(f"Output keys: {list(outputs.keys())}")
    for key, value in outputs.items():
        assert value.shape == (batch_size, 3), (
            f"Output {key} shape mismatch! Expected ({batch_size}, 3), got {value.shape}"
        )
        print(f"  {key}: {value.shape} OK")

    print("\nAll tests passed!")
