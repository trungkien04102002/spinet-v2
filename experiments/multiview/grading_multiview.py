"""
GradingMultiView — T2 + per-side-T1 fusion model for RSNA lumbar grading.

Reuses the existing 3D-ResNet34(+CBAM) encoder from
`spinenet.models.grading_attention.GradingModelWithCBAM` as TWO independent
branches:
  - `encoder_t2`  — Sagittal T2 midline crop (one embedding per sample).
  - `encoder_t1`  — Sagittal T1 foraminal crop, applied TWICE per sample
    (once on the LEFT crop, once on the RIGHT crop) — the SAME weights are
    reused for both sides, since it's the same modality/anatomy just
    imaged on a different lateral slice. See reconciliation note below.

Reconciled 2026-07-22 against the actual sibling-task output
(`experiments/f1_improvement/prep_t1_crops.py`): T1 foraminal crops are
PER-SIDE — left/right neural foraminal narrowing live on DIFFERENT T1
slices, so there is no single midline T1 crop the way there is for T2.
Consequently:
  - `spinal_canal` uses **T2 only** (no T1 input at all — there's no
    canal-relevant T1 crop in this design).
  - `left_foraminal` fuses (T2, T1-LEFT).
  - `right_foraminal` fuses (T2, T1-RIGHT).

Output contract (same as the single-view models):
    forward(t2_vol, t1_left_vol, t1_right_vol) -> {
        'spinal_canal':    (B, 3),   # T2-only head, no fusion
        'left_foraminal':  (B, 3),   # fusion(T2, T1-left)
        'right_foraminal': (B, 3),   # fusion(T2, T1-right)
    }

Two fusion modes (apply only to the two foraminal heads; spinal_canal is
always a plain linear head on the T2 embedding):
    fusion='concat' — flat concat of [emb_t2 ; emb_t1_side] (1024-dim) ->
        per-condition linear head. This is the Week-1/2 baseline
        (MULTIVIEW_RESEARCH.md §(2).1 "late fusion").

    fusion='gated' — per-condition GMU-style gate (MULTIVIEW_RESEARCH.md §3):
        a_{c,s} = MLP([e_c ; emb_s])         # e_c = learnable per-condition query
        w_{c,s} = softmax_s(a_{c,s})         # softmax over sequences {T2, T1-side}
        fused_c = sum_s w_{c,s} * emb_s
        logits_c = head_c(fused_c)
    Gate bias is initialized toward the clinical prior (T1-heavy for both
    foraminal conditions — see MULTIVIEW_RESEARCH.md (a) table) but stays
    learnable (plain nn.Parameter, no detach/freeze).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from spinenet.models.grading_attention import GradingModelWithCBAM


CONDITIONS = ["spinal_canal", "left_foraminal", "right_foraminal"]
# Only these two conditions have a T1 counterpart (per-side) and are
# actually fused/gated. spinal_canal is T2-only — see module docstring.
FORAMINAL_CONDITIONS = ["left_foraminal", "right_foraminal"]
CONDITION_SIDE = {"left_foraminal": "left", "right_foraminal": "right"}

SEQUENCES = ["t2", "t1"]  # fixed order used everywhere below (per foraminal head)

# Clinical prior used only to INITIALIZE the gated-fusion gate bias (kept
# learnable afterwards) — see MULTIVIEW_RESEARCH.md (a) table: neural
# foraminal narrowing (both sides) is graded primarily on Sagittal T1.
# Values are pre-softmax logit offsets (T2 first, T1 second) — a modest
# nudge, not a hard gate. (spinal_canal has no gate — T2-only head.)
_GATE_PRIOR_LOGITS = {
    "left_foraminal": [0.0, 1.0],     # T1 leader
    "right_foraminal": [0.0, 1.0],    # T1 leader
}


class GMUConditionGate(nn.Module):
    """Per-condition GMU-style gate over a fixed set of sequence embeddings.

    a_{c,s} = MLP([e_c ; emb_s])
    w_{c,s} = softmax_s(a_{c,s})
    fused_c = sum_s w_{c,s} * emb_s
    """

    def __init__(
        self,
        embed_dim: int,
        condition: str,
        num_sequences: int = 2,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.condition = condition
        self.num_sequences = num_sequences

        # Learnable per-condition query embedding e_c.
        self.query = nn.Parameter(torch.randn(embed_dim) * 0.02)

        # Small MLP scoring [e_c ; emb_s] -> scalar attention logit a_{c,s}.
        # Shared across sequences (same MLP applied per-sequence, per GMU).
        self.score_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize the score MLP's output bias per-sequence via a
        # separate learnable additive bias (one scalar per sequence),
        # nudged toward the clinical prior but still trainable.
        prior = _GATE_PRIOR_LOGITS.get(condition, [0.0] * num_sequences)
        assert len(prior) == num_sequences
        self.prior_bias = nn.Parameter(torch.tensor(prior, dtype=torch.float32))

    def forward(self, embeddings: List[torch.Tensor]):
        """
        Args:
            embeddings: list of (B, embed_dim) tensors, one per sequence,
                in SEQUENCES order (t2, t1[-side]).

        Returns:
            fused: (B, embed_dim)
            weights: (B, num_sequences) softmax weights (for visualization
                of the learned leader/supporter split).
        """
        assert len(embeddings) == self.num_sequences
        batch_size = embeddings[0].shape[0]
        query = self.query.unsqueeze(0).expand(batch_size, -1)  # (B, D)

        scores = []
        for s, emb_s in enumerate(embeddings):
            paired = torch.cat([query, emb_s], dim=1)  # (B, 2D)
            a_s = self.score_mlp(paired).squeeze(-1)  # (B,)
            a_s = a_s + self.prior_bias[s]
            scores.append(a_s)
        scores = torch.stack(scores, dim=1)  # (B, num_sequences)
        weights = torch.softmax(scores, dim=1)  # (B, num_sequences)

        stacked = torch.stack(embeddings, dim=1)  # (B, num_sequences, D)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B, D)

        return fused, weights


class GradingMultiView(nn.Module):
    """T2 (+ per-side T1) multi-view grading model.

    Args:
        fusion: 'concat' (flat late fusion) or 'gated' (GMU leader/supporter)
            — applies to the two foraminal heads only; spinal_canal is
            always a plain T2-only linear head (see module docstring).
        use_cbam: passed through to both encoders.
        embed_dim: encoder output dim (512 for the ResNet34 backbone used
            here — do not change unless the backbone changes).
    """

    def __init__(
        self,
        fusion: str = "concat",
        use_cbam: bool = True,
        cbam_reduction: int = 16,
        embed_dim: int = 512,
        gate_hidden_dim: int = 128,
    ):
        super().__init__()
        if fusion not in ("concat", "gated"):
            raise ValueError(f"Unknown fusion mode: {fusion!r} (use 'concat' or 'gated')")

        self.fusion = fusion
        self.embed_dim = embed_dim

        # === ENCODERS ===
        # Reuse the existing CBAM/ResNet34 3D backbone as-is (format='rsna'
        # only matters for its own heads, which we never call — we use
        # `.encode()` to get the pre-head 512-dim embedding).
        # encoder_t2: T2 midline crop (spinal_canal + supporter for both
        #             foraminal heads).
        # encoder_t1: T1 foraminal crop, applied to LEFT and RIGHT crops
        #             separately (same weights both times — same modality,
        #             just a different lateral slice per side).
        self.encoder_t2 = GradingModelWithCBAM(
            format="rsna", use_cbam=use_cbam, cbam_reduction=cbam_reduction
        )
        self.encoder_t1 = GradingModelWithCBAM(
            format="rsna", use_cbam=use_cbam, cbam_reduction=cbam_reduction
        )

        # === HEADS ===
        # spinal_canal: T2-only, no fusion, same for both fusion modes.
        self.head_spinal_canal = nn.Linear(embed_dim, 3)

        if fusion == "concat":
            fused_dim = embed_dim * 2
            self.heads = nn.ModuleDict(
                {cond: nn.Linear(fused_dim, 3) for cond in FORAMINAL_CONDITIONS}
            )
            self.gates = None
        else:  # fusion == "gated"
            self.gates = nn.ModuleDict(
                {
                    cond: GMUConditionGate(
                        embed_dim=embed_dim,
                        condition=cond,
                        num_sequences=len(SEQUENCES),
                        hidden_dim=gate_hidden_dim,
                    )
                    for cond in FORAMINAL_CONDITIONS
                }
            )
            self.heads = nn.ModuleDict(
                {cond: nn.Linear(embed_dim, 3) for cond in FORAMINAL_CONDITIONS}
            )

        # Populated on the most recent forward() call when fusion='gated' —
        # {condition: (B, num_sequences) tensor}, ONLY for the two foraminal
        # conditions (spinal_canal has no gate — T2-only) — exposed so
        # callers can visualize the learned leader/supporter split without
        # re-running the gate.
        self.last_gate_weights: Optional[Dict[str, torch.Tensor]] = None

    def load_pretrained_backbones(self, weights_dir: str, verbose: bool = True):
        """Load the shared pretrained 3D-ResNet34 backbone into BOTH
        branches (T2 and T1 start from the same pretrained weights; they
        diverge during fine-tuning)."""
        self.encoder_t2.load_pretrained_backbone(weights_dir, verbose=verbose)
        self.encoder_t1.load_pretrained_backbone(weights_dir, verbose=verbose)

    def load_trained_cbam_encoders(self, checkpoint_path: str, verbose: bool = True):
        """Warm-start BOTH encoders from a trained CBAM checkpoint (continue
        from the existing model rather than starting from the generic backbone)."""
        self.encoder_t2.load_trained_cbam(checkpoint_path, verbose=verbose)
        self.encoder_t1.load_trained_cbam(checkpoint_path, verbose=verbose)

    def freeze_backbones(self, freeze: bool = True):
        self.encoder_t2.freeze_backbone(freeze=freeze)
        self.encoder_t1.freeze_backbone(freeze=freeze)

    def forward(
        self,
        t2_vol: torch.Tensor,
        t1_left_vol: torch.Tensor,
        t1_right_vol: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            t2_vol:       (B, 1, 9, 112, 224) — Sagittal T2 midline crop
            t1_left_vol:  (B, 1, 9, 112, 224) — Sagittal T1 LEFT crop
            t1_right_vol: (B, 1, 9, 112, 224) — Sagittal T1 RIGHT crop

        Returns:
            dict {spinal_canal, left_foraminal, right_foraminal} each (B, 3)
        """
        emb_t2 = self.encoder_t2.encode(t2_vol)              # (B, embed_dim)
        emb_t1_left = self.encoder_t1.encode(t1_left_vol)    # (B, embed_dim)
        emb_t1_right = self.encoder_t1.encode(t1_right_vol)  # (B, embed_dim)

        emb_by_side = {"left": emb_t1_left, "right": emb_t1_right}

        outputs = {}
        # spinal_canal: T2-only, no fusion.
        outputs["spinal_canal"] = self.head_spinal_canal(emb_t2)

        gate_weights = {}
        for cond in FORAMINAL_CONDITIONS:
            emb_t1_side = emb_by_side[CONDITION_SIDE[cond]]

            if self.fusion == "concat":
                fused_c = torch.cat([emb_t2, emb_t1_side], dim=1)  # (B, 2*embed_dim)
                outputs[cond] = self.heads[cond](fused_c)
            else:  # gated
                fused_c, weights_c = self.gates[cond]([emb_t2, emb_t1_side])
                outputs[cond] = self.heads[cond](fused_c)
                gate_weights[cond] = weights_c

        self.last_gate_weights = gate_weights if self.fusion == "gated" else None

        return outputs

    def get_gate_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """Per-condition softmax weights over [T2, T1-side] from the last
        forward call (fusion='gated' only) — for the leader/supporter viz.
        Keys are ONLY the two foraminal conditions (spinal_canal has no
        gate — T2-only head). Returns None for fusion='concat'.
        """
        return self.last_gate_weights


# Self-test
if __name__ == "__main__":
    print("Testing GradingMultiView (both fusion modes)...")

    batch_size = 2
    t2 = torch.randn(batch_size, 1, 9, 112, 224)
    t1_left = torch.randn(batch_size, 1, 9, 112, 224)
    t1_right = torch.randn(batch_size, 1, 9, 112, 224)

    for fusion in ("concat", "gated"):
        print(f"\n--- fusion={fusion} ---")
        model = GradingMultiView(fusion=fusion)
        model.eval()
        with torch.no_grad():
            outputs = model(t2, t1_left, t1_right)

        for cond in CONDITIONS:
            assert outputs[cond].shape == (batch_size, 3), (
                f"{fusion}: {cond} shape mismatch: {outputs[cond].shape}"
            )
            print(f"  {cond}: {outputs[cond].shape} OK")

        gw = model.get_gate_weights()
        if fusion == "gated":
            assert gw is not None
            assert set(gw.keys()) == set(FORAMINAL_CONDITIONS)
            for cond in FORAMINAL_CONDITIONS:
                assert gw[cond].shape == (batch_size, len(SEQUENCES))
                sums = gw[cond].sum(dim=1)
                assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
                print(f"  gate[{cond}] weights (T2, T1-{CONDITION_SIDE[cond]}): {gw[cond][0].tolist()}")
        else:
            assert gw is None

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total params: {total_params:,}")

    print("\nAll self-tests passed!")
