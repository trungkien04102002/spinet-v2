#!/usr/bin/env python3
"""Average the trainable weights of several Hybrid checkpoints from one run.

Standard SWA: checkpoints from a single run with no LR restarts sit in the same
loss basin, so averaging their weights is meaningful. Only the trainable head is
stored in these checkpoints (the CBAM backbone and BiomedCLIP are frozen and
loaded separately), so this averages exactly what was learned.

Writes a checkpoint in the same format, so dump_logits.py consumes it unchanged.
"""
import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    states, metas = [], []
    for p in args.inputs:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        states.append(sd)
        metas.append((Path(p).name, ck.get("epoch")))
        print(f"  loaded {Path(p).name}  epoch={ck.get('epoch')}  tensors={len(sd)}")

    keys = set(states[0])
    for sd in states[1:]:
        if set(sd) != keys:
            raise SystemExit("checkpoints have different parameter sets; refusing to average")

    avg = {}
    for k in states[0]:
        vals = [sd[k] for sd in states]
        if vals[0].is_floating_point():
            avg[k] = torch.stack([v.float() for v in vals], 0).mean(0).to(vals[0].dtype)
        else:
            # Integer buffers cannot be meaningfully averaged; they must agree.
            if not all(torch.equal(vals[0], v) for v in vals[1:]):
                raise SystemExit(f"non-float tensor {k} differs across checkpoints")
            avg[k] = vals[0]

    # Report how far apart the inputs were, so a no-op average is visible.
    ref = states[0]
    drift = sum(float((ref[k].float() - avg[k].float()).norm()) for k in ref
                if ref[k].is_floating_point())
    print(f"  L2 drift of checkpoint 1 from the average: {drift:.4f}")

    torch.save({"model_state_dict": avg,
                "swa_sources": metas,
                "epoch": "swa"}, args.output)
    print(f"  saved {args.output}")


if __name__ == "__main__":
    main()
