"""
Grading model implementing a Transformer-encoder paradigm as Baseline 2 for
the external SOTA comparison, deliberately DIFFERENT from Baseline 1
(``grading_brendanartley.py``, a recurrent CNN->BiLSTM->attention-pool
design). Same drop-in-replacement contract as
``spinenet.models.grading_baseline.GradingModelBaseline`` so numbers are
directly comparable to our Table 1.

Architecture: per-slice 2D CNN encoder (ResNet18, one forward per one of the
9 sagittal slices) produces 9 slice "tokens" -> prepend a learnable [CLS]
token -> add learnable positional embeddings -> small nn.TransformerEncoder
(self-attention over the 10-token sequence, batch_first=True) -> take the
CLS output -> 3 independent linear classification heads (spinal_canal /
left_foraminal / right_foraminal, 3 classes each).

This mirrors the ViT-style "tokenize + CLS + positional embedding +
self-attention" paradigm, applied to a short sequence of per-slice
embeddings instead of image patches -- a non-recurrent alternative to
Baseline 1's BiLSTM, at a similar parameter scale, and 24GB-GPU-friendly.

Interface contract (must match GradingModelBaseline / GradingModelBrendanartley):
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
    5-stage conv stack producing a fixed-size embedding per frame. Identical
    to the fallback used in grading_brendanartley.py (kept in sync so the
    two baselines are comparable even without torchvision).
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


class GradingModelTransformer(nn.Module):
    """Drop-in replacement for GradingModelBaseline using a
    tokenize-slices + [CLS] + positional-embedding + TransformerEncoder
    paradigm (non-recurrent alternative to the BiLSTM baseline).

    Input:  [B, 1, 9, 112, 224]
    Output (format='rsna'): dict of 3 logit tensors, each [B, 3].
    """

    NUM_SLICES = 9

    def __init__(
        self,
        format: str = "rsna",
        num_layers: int = 3,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        if format != "rsna":
            raise ValueError(
                f"GradingModelTransformer only supports format='rsna', got {format!r}"
            )
        self.format = format

        self.frame_encoder, emb_dim = _build_frame_encoder()
        self.emb_dim = emb_dim

        # Learnable [CLS] token, prepended to the 9 slice tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings for the (1 CLS + 9 slice) tokens.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_SLICES + 1, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)

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
        assert t == self.NUM_SLICES, (
            f"expected {self.NUM_SLICES} sagittal slices, got {t}"
        )

        # Treat the 9 sagittal slices as tokens for a 2D encoder.
        # [B, 1, T, H, W] -> [B*T, 1, H, W] -> repeat to 3ch for resnet18.
        frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w)
        frames = frames.repeat(1, 3, 1, 1)  # [B*T, 3, H, W]

        feats = self.frame_encoder(frames)      # [B*T, emb_dim]
        tokens = feats.view(b, t, self.emb_dim)  # [B, T, emb_dim]

        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, emb_dim]
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, T+1, emb_dim]
        tokens = tokens + self.pos_embed

        encoded = self.transformer(tokens)  # [B, T+1, emb_dim]
        cls_out = encoded[:, 0]             # [B, emb_dim] (CLS token output)
        cls_out = self.norm(cls_out)
        cls_out = self.fc_dropout(cls_out)

        out_spinal_canal = self.fc_spinal_canal(cls_out)
        out_left_foraminal = self.fc_left_foraminal(cls_out)
        out_right_foraminal = self.fc_right_foraminal(cls_out)

        return {
            "spinal_canal": out_spinal_canal,
            "left_foraminal": out_left_foraminal,
            "right_foraminal": out_right_foraminal,
        }

    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze the per-frame CNN encoder only (mirrors
        GradingModelBaseline.freeze_backbone / GradingModelBrendanartley
        .freeze_backbone — heads + CLS/pos embeddings + transformer stay
        trainable so a linear-probe-style "Phase 1" run is still possible).
        """
        for param in self.frame_encoder.parameters():
            param.requires_grad = not freeze


# Test code
if __name__ == "__main__":
    print("Testing GradingModelTransformer...\n")
    print(f"torchvision available: {_HAS_TORCHVISION}")

    model = GradingModelTransformer(format="rsna")
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
