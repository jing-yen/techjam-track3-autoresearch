"""
M2 -- generalized version of profile_shape8.py: real torch.profiler trace for
any shape (default: the two still-open ones from TODO.md's M2 entry, #1 and
#13 -- #8 is already done). Same protocol as profile_shape8.py, parametrized.

Usage: python tools/profile_shapes.py --shapes 1,13 [--candidate PATH]
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


def profile_one(cand, shape_id: int, device, warmup: int = 10, active: int = 20):
    cfg_dict = OFFICIAL_SHAPES[shape_id]
    config = TransformerConfig(**cfg_dict)

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
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True, with_stack=False) as prof:
            for _ in range(active):
                model(x, mask)
            torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)
    print(f"=== Shape #{shape_id} ({cfg_dict}) -- candidate: {cand.__name__} ===")
    print(table)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default=os.path.join(_ROOT, "candidates", "v_router2.py"))
    p.add_argument("--shapes", default="1,13")
    args = p.parse_args()

    device = torch.device("cuda")
    spec = importlib.util.spec_from_file_location("cand", args.candidate)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    for sid in [int(s) for s in args.shapes.split(",")]:
        profile_one(cand, sid, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
