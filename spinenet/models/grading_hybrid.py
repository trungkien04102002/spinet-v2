"""
Hybrid Grading Model: CBAM 3D + BiomedCLIP 2D + Cosine Similarity Classification.

Architecture:
    Volume (B, 1, 9, 112, 224)
        |-- CBAM 3D ResNet34 (frozen, from RSNA checkpoint) --> feat_cbam [B, 512]
        |-- Slice select (3 center) --> per-slice BiomedCLIP 2D (frozen) --> [B, 3, 512]
        |                                                                   |
        |                                                            Attention pool
        |                                                                   |
        |                                                              feat_bmc [B, 512]
        |
        Concat -> [B, 1024] -> MLP projection (TRAIN) -> image_emb [B, 512]

    Label texts -> BiomedCLIP text encoder (frozen) -> text_embs [N, 512]

    logits = logit_scale * image_emb @ text_embs.T  -> [B, N]

Trainable parameters:
    - Attention pool weights (~1K)
    - Image projection MLP (~500K)
    - logit_scale (1)

Frozen parameters:
    - CBAM 3D backbone (~22M)
    - BiomedCLIP image encoder (~86M)
    - BiomedCLIP text encoder + text projection (~110M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from spinenet.models.grading_attention import GradingModelWithCBAM
from spinenet.models.biomedclip_wrapper import BiomedCLIPWrapper


class SliceAttentionPool(nn.Module):
    """
    Soft attention pooling over K slices.
    Learns per-slice importance weights, returns weighted sum.
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)

    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: [B, K, D] - K slice features per batch item
        Returns:
            [B, D] - attention-weighted sum
        """
        scores = self.scorer(slice_features).squeeze(-1)  # [B, K]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, K, 1]
        pooled = (slice_features * weights).sum(dim=1)  # [B, D]
        return pooled


class SpineNetHybrid(nn.Module):
    """
    Hybrid CBAM + BiomedCLIP model for spine MRI grading with zero-shot capability.

    Args:
        cbam_checkpoint_path: Path to trained CBAM checkpoint (best_model_attention.pth)
        biomedclip_device: Device for BiomedCLIP (separate from main device for memory)
        slice_strategy: 'static' (3 center slices) or 'dynamic' (cosine similarity filter)
        dynamic_keep_ratio: Fraction of slices to keep when slice_strategy='dynamic' (default 0.33)
    """

    def __init__(
        self,
        cbam_checkpoint_path: str,
        biomedclip_device: str = "cuda",
        slice_strategy: str = "static",
        dynamic_keep_ratio: float = 0.33,
        projection_hidden: int = 768,
        embed_dim: int = 512,
    ):
        super().__init__()

        # ---- Frozen CBAM backbone ----
        self.cbam = GradingModelWithCBAM(format="rsna", use_cbam=True)
        ckpt = torch.load(cbam_checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.cbam.load_state_dict(state, strict=False)
        for p in self.cbam.parameters():
            p.requires_grad = False
        self.cbam.eval()

        # ---- Frozen BiomedCLIP ----
        self.biomedclip = BiomedCLIPWrapper(device=biomedclip_device)

        # ---- Slice strategy ----
        assert slice_strategy in ("static", "dynamic"), f"Unknown slice_strategy: {slice_strategy}"
        self.slice_strategy = slice_strategy
        self.dynamic_keep_ratio = dynamic_keep_ratio

        # ---- Trainable: attention pool ----
        self.slice_pool = SliceAttentionPool(dim=embed_dim)

        # ---- Trainable: image projection MLP ----
        cbam_dim = embed_dim  # CBAM outputs 512
        bmc_dim = embed_dim  # BiomedCLIP outputs 512
        self.image_projection = nn.Sequential(
            nn.Linear(cbam_dim + bmc_dim, projection_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_hidden, embed_dim),
        )

        # ---- Trainable: logit scale (CLIP-style) ----
        # CLIP convention: parameter is in log-space, exp() gives temperature.
        # init log(1/0.07) ≈ 2.659, so exp(2.659) ≈ 14.3 (CLIP default temperature).
        # Init as 1/0.07 (=14.28) was a bug: exp(14.28) ≈ 1.6M, gets clamped to
        # 100 from step 1, dead gradient. See OpenAI CLIP / OpenCLIP / HF convention.
        import math
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    # ----------------- Slice encoding -----------------

    def _select_slices_static(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Pick 3 center slices.
        volume: [B, 1, 9, 112, 224]
        returns: [B, 3, 112, 224]
        """
        return volume[:, 0, 3:6, :, :]

    @torch.no_grad()
    def _encode_slices_via_biomedclip(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Encode K grayscale slices per volume via BiomedCLIP image encoder.

        Args:
            slices: [B, K, H, W] grayscale in [0, 1]

        Returns:
            [B, K, 512]
        """
        B, K, H, W = slices.shape
        flat = slices.reshape(B * K, H, W)
        rgb_list = []
        for i in range(B * K):
            rgb_list.append(self.biomedclip.preprocess_slice(flat[i]))
        rgb_batch = torch.stack(rgb_list, dim=0).to(slices.device)  # [B*K, 3, 224, 224]
        embs = self.biomedclip.encode_image(rgb_batch)  # [B*K, 512]
        return embs.reshape(B, K, -1)

    @torch.no_grad()
    def _select_slices_dynamic(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Select slices via cosine similarity filtering (Lian et al. 2026 inspired).

        Encode all 9 slices, compute average similarity to neighbors, keep top
        `dynamic_keep_ratio * 9` slices with LOWEST similarity (most unique).

        Args:
            volume: [B, 1, 9, 112, 224]

        Returns:
            [B, K, 112, 224] selected slices, where K = round(0.33 * 9) = 3
        """
        B = volume.shape[0]
        all_slices = volume[:, 0, :, :, :]  # [B, 9, 112, 224]
        all_features = self._encode_slices_via_biomedclip(all_slices)  # [B, 9, 512]

        # Pairwise cosine sim within each batch item
        sim_matrix = torch.bmm(all_features, all_features.transpose(1, 2))  # [B, 9, 9]
        avg_sim = sim_matrix.mean(dim=-1)  # [B, 9] - avg sim to all slices

        K = max(1, round(self.dynamic_keep_ratio * 9))
        # Lowest avg_sim = most unique
        _, keep_idx = torch.topk(avg_sim, K, dim=-1, largest=False)  # [B, K]
        keep_idx_sorted, _ = torch.sort(keep_idx, dim=-1)  # preserve depth order

        # Gather slices
        idx_expand = keep_idx_sorted.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 112, 224)
        selected = torch.gather(all_slices, 1, idx_expand)  # [B, K, 112, 224]
        return selected

    # ----------------- Encoders -----------------

    def encode_image(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Compute image embedding for batch of volumes.

        Args:
            volume: [B, 1, 9, 112, 224]

        Returns:
            image_emb: [B, 512] L2-normalized
        """
        # CBAM 3D path. If any CBAM parameter still requires gradients (e.g.
        # SPIDER full fine-tune via --unfreeze-cbam), we keep autograd active so
        # gradients can flow into the backbone. Otherwise we save memory + time
        # by running under no_grad.
        cbam_trainable = any(p.requires_grad for p in self.cbam.parameters())
        if cbam_trainable:
            feat_cbam = self.cbam.encode(volume)
        else:
            with torch.no_grad():
                feat_cbam = self.cbam.encode(volume)  # [B, 512]

        # BiomedCLIP 2D path (frozen) + slice attention pool (trainable)
        if self.slice_strategy == "static":
            slices = self._select_slices_static(volume)
        else:  # dynamic
            slices = self._select_slices_dynamic(volume)

        slice_features = self._encode_slices_via_biomedclip(slices)  # [B, K, 512]
        feat_bmc = self.slice_pool(slice_features)  # [B, 512]

        # Fuse
        concat = torch.cat([feat_cbam, feat_bmc], dim=-1)  # [B, 1024]
        image_emb = self.image_projection(concat)  # [B, 512]
        image_emb = F.normalize(image_emb, dim=-1)
        return image_emb

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode list of text labels via frozen BiomedCLIP text encoder.

        Returns:
            [N, 512] L2-normalized
        """
        return self.biomedclip.encode_text(texts)

    # ----------------- Forward -----------------

    def forward(
        self,
        volume: torch.Tensor,
        text_embs: torch.Tensor,
    ) -> tuple:
        """
        Args:
            volume: [B, 1, 9, 112, 224]
            text_embs: [N, 512] pre-computed text embeddings (L2-normalized)

        Returns:
            logits: [B, N]
            image_emb: [B, 512]
        """
        image_emb = self.encode_image(volume)
        logits = self.logit_scale.exp().clamp(max=100.0) * (image_emb @ text_embs.T)
        return logits, image_emb

    def trainable_params(self):
        """Yield only trainable parameters (projection, attention pool, logit_scale)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield p


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grading_hybrid.py <cbam_checkpoint_path>")
        sys.exit(1)

    cbam_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building Hybrid model...")
    model = SpineNetHybrid(
        cbam_checkpoint_path=cbam_path,
        biomedclip_device=device,
        slice_strategy="static",
    ).to(device)

    print(f"\nDevice: {device}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    # Forward test
    print("\nForward test...")
    volume = torch.rand(2, 1, 9, 112, 224).to(device)
    text_embs = model.encode_text([
        "normal spinal canal",
        "moderate spinal canal stenosis",
        "severe spinal canal stenosis",
    ])
    print(f"Text embs shape: {text_embs.shape}")

    logits, image_emb = model(volume, text_embs)
    print(f"Logits shape: {logits.shape} (expect [2, 3])")
    print(f"Image emb shape: {image_emb.shape} (expect [2, 512])")
    print(f"Image emb L2 norms: {image_emb.norm(dim=-1)}")

    print("\nAll tests passed.")
