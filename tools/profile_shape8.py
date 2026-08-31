"""
M2 -- real torch.profiler trace for shape #8 (the lowest-ratio shape,
d_model=1024, currently routed to v_router2's fp16 amp + Triton AddNorm
route). Answers "what's actually slow" with a measured per-op/per-kernel
CUDA time breakdown instead of Roofline inference from opt_ms + GFLOP counts
-- which is all that's existed in this repo until now (see TODO.md M2).

Usage: on a CUDA node,
    python tools/profile_shape8.py [--candidate PATH] [--out FILE]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from bench_harness import OFFICIAL_SHAPES  # noqa: E402
import torch_transformer_benchmark as ttb  # noqa: E402
from torch_transformer_benchmark import TransformerConfig, BaselineTransformer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default=os.path.join(_ROOT, "candidates", "v_router2.py"))
    p.add_argument("--shape", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--active", type=int, default=20)
    args = p.parse_args()

    device = torch.device("cuda")
    cfg_dict = OFFICIAL_SHAPES[args.shape]
    config = TransformerConfig(**cfg_dict)

    spec = importlib.util.spec_from_file_location("cand", args.candidate)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    baseline = BaselineTransformer(config).to(device)
    model = cand.UserOptimizedTransformer(config).to(device)
    custom_copy = getattr(cand, "copy_model_weights", None)
    strict = bool(getattr(cand, "STRICT_WEIGHT_COPY", True))
    if callable(custom_copy):
        custom_copy(baseline, model)
    else:
        ttb.copy_model_weights(baseline, model, strict=strict)
    model.eval()

    x = torch.randn(cfg_dict["batch_size"], cfg_dict["seq_len"], cfg_dict["d_model"], device=device)
    mask = torch.ones(cfg_dict["batch_size"], cfg_dict["seq_len"], dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            model(x, mask)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=False) as prof:
            for _ in range(args.active):
                model(x, mask)
            torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(f"=== Shape #{args.shape} ({cfg_dict}) -- candidate: {args.candidate} ===")
    print(table)

    if args.out:
        with open(args.out, "w") as f:
            f.write(table)
        prof.export_chrome_trace(args.out + ".chrometrace.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
