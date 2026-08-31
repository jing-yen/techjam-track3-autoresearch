"""
T17 root-cause check -- why did applying T7+T15's AddNorm kernel to the
fused route regress shapes #9/#10/#12 by ~30%? The TODO/journal entry's
"best guess" (fused_add_layernorm's .contiguous() calls adding copy
overhead) is likely WRONG on reflection: Tensor.contiguous() no-ops (no
copy) when the tensor is already contiguous, and every call site here
(Linear/GELU outputs, and fused_add_layernorm's own view() of a
torch.empty_like 2D allocation) should already be contiguous by
construction. Real profiler data, not another guess.

Profiles shape #9 (batch=64, seq=128, d=128, heads=1 -- the biggest
regression) under both the CURRENT v_router2 (fused route, no AddNorm) and
T17 (v_triton_addnorm_fused, AddNorm at both boundaries) side by side.
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


def profile_one(candidate_path, shape_id, device, warmup=15, active=30):
    cfg_dict = OFFICIAL_SHAPES[shape_id]
    config = TransformerConfig(**cfg_dict)

    spec = importlib.util.spec_from_file_location("cand", candidate_path)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    baseline = BaselineTransformer(config).to(device)
    model = cand.UserOptimizedTransformer(config).to(device)
    custom_copy = getattr(cand, "copy_model_weights", None)
    if callable(custom_copy):
        custom_copy(baseline, model)
    else:
        ttb.copy_model_weights(baseline, model, strict=bool(getattr(cand, "STRICT_WEIGHT_COPY", True)))
    model.eval()

    x = torch.randn(cfg_dict["batch_size"], cfg_dict["seq_len"], cfg_dict["d_model"], device=device)
    mask = torch.ones(cfg_dict["batch_size"], cfg_dict["seq_len"], dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, with_stack=False) as prof:
            for _ in range(active):
                model(x, mask)
            torch.cuda.synchronize()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
    print(f"=== Shape #{shape_id} -- candidate: {candidate_path} ===")
    print(table)
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, default=9)
    args = p.parse_args()
    device = torch.device("cuda")
    profile_one(os.path.join(_ROOT, "candidates", "v_router2.py"), args.shape, device)
    profile_one(os.path.join(_ROOT, "candidates", "v_triton_addnorm_fused.py"), args.shape, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
