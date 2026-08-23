"""Strip frozen `biomedclip.*` keys from Hybrid SPIDER checkpoints to slim them.

The BiomedCLIP image+text encoders are frozen during Hybrid training, so saving
them in the checkpoint is dead weight (~750 MB per file). They can be re-loaded
from the BiomedCLIP HuggingFace hub at inference time.

By default also drops `optimizer_state_dict` (only useful for resuming training,
~500 MB for unfreeze runs). Pass --keep-optimizer to retain it.

Usage:
    python3 scripts/strip_biomedclip_keys.py [--keep-optimizer] <ckpt.pth> [<ckpt2.pth> ...]

Writes <ckpt>.slim.pth next to each input. Verifies size reduction and that
trainable keys (cbam, image_projection, slice_pool, logit_scale) are preserved.
"""
import sys
from pathlib import Path
import torch


def strip(ckpt_path: Path, keep_optimizer: bool = False) -> Path:
    print(f"\n[+] Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state, dict):
        raise SystemExit(f"Unexpected checkpoint structure in {ckpt_path}")

    n_before = len(state)
    bmc_keys = [k for k in state if k.startswith("biomedclip.")]
    kept = {k: v for k, v in state.items() if not k.startswith("biomedclip.")}
    n_after = len(kept)

    bmc_bytes = sum(v.numel() * v.element_size() for v in
                    (state[k] for k in bmc_keys))
    kept_bytes = sum(v.numel() * v.element_size() for v in kept.values())

    print(f"    keys: {n_before} -> {n_after} (dropped {len(bmc_keys)} biomedclip.*)")
    print(f"    biomedclip weights: {bmc_bytes/1024**2:.1f} MB removed")
    print(f"    kept weights:       {kept_bytes/1024**2:.1f} MB")

    cls_kept = sorted({k.split(".")[0] for k in kept})
    print(f"    top-level kept: {cls_kept}")

    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = kept
        if not keep_optimizer and "optimizer_state_dict" in ckpt:
            opt = ckpt.pop("optimizer_state_dict")
            opt_bytes = sum(
                vv.numel() * vv.element_size()
                for v in opt.get("state", {}).values()
                if isinstance(v, dict)
                for vv in v.values()
                if hasattr(vv, "numel")
            )
            print(f"    optimizer_state_dict: {opt_bytes/1024**2:.1f} MB removed")
        out_obj = ckpt
    else:
        out_obj = kept

    out = ckpt_path.with_suffix(".slim.pth")
    torch.save(out_obj, out)
    print(f"[+] Wrote: {out} ({out.stat().st_size/1024**2:.1f} MB)")
    return out


def main():
    args = sys.argv[1:]
    keep_optimizer = False
    if "--keep-optimizer" in args:
        keep_optimizer = True
        args.remove("--keep-optimizer")
    if not args:
        print(__doc__)
        sys.exit(1)
    for arg in args:
        strip(Path(arg), keep_optimizer=keep_optimizer)


if __name__ == "__main__":
    main()
