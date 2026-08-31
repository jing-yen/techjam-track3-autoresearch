"""
K2' -- is GELU actually landing fused into a Triton epilogue on the
compile/reduce routes, or staying a separate ATen pointwise kernel after
whichever GEMM backend wins? K4's diagnostic run already showed Inductor's
own autotune competition picks ATen's `bias_addmm` over the best Triton GEMM
candidate for our GEMM shapes (128x128x128: bias_addmm 0.0123ms vs Triton's
best 0.0143ms, ~17% slower) -- so the raw-GEMM half of K2' is already
answered from that data alone, no new GPU time needed. This script settles
the other half: whether GELU shows up as its own kernel in the generated
code (meaning a hand-fused Triton epilogue could still save a launch) or is
already being fused into something.

Uses TORCH_LOGS=output_code (via torch._logging.set_logs) to dump Inductor's
actual generated Triton/C++ code and greps it for gelu/erf.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from bench_harness import OFFICIAL_SHAPES  # noqa: E402
import torch_transformer_benchmark as ttb  # noqa: E402
from torch_transformer_benchmark import TransformerConfig, BaselineTransformer  # noqa: E402


def main() -> int:
    device = torch.device("cuda")

    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.DEBUG)
    logging.getLogger("torch._inductor.graph").addHandler(handler)
    logging.getLogger("torch._inductor.codecache").addHandler(handler)
    logging.getLogger("torch._inductor").addHandler(handler)
    torch._logging.set_logs(output_code=True)

    cand_path = os.path.join(_ROOT, "candidates", "v_router2.py")
    spec = importlib.util.spec_from_file_location("cand", cand_path)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    # shape 1: (64,128,128,4) -> "compile" route. Small enough for a fast,
    # readable dump.
    cfg_dict = OFFICIAL_SHAPES[1]
    config = TransformerConfig(**cfg_dict)
    baseline = BaselineTransformer(config).to(device)
    model = cand.UserOptimizedTransformer(config).to(device)
    ttb.copy_model_weights(baseline, model, strict=True)
    model.eval()

    x = torch.randn(cfg_dict["batch_size"], cfg_dict["seq_len"], cfg_dict["d_model"], device=device)
    mask = torch.ones(cfg_dict["batch_size"], cfg_dict["seq_len"], dtype=torch.bool, device=device)

    with torch.no_grad():
        for _ in range(5):
            model(x, mask)
        torch.cuda.synchronize()

    text = log_buf.getvalue()
    print(f"=== captured {len(text)} chars of output_code log ===")

    lines = text.splitlines()
    gelu_lines = [ln for ln in lines if "gelu" in ln.lower() or "erf" in ln.lower()]
    kernel_def_lines = [ln for ln in lines if ln.strip().startswith("def triton_") or ln.strip().startswith("async_compile.triton")]

    print(f"\n=== {len(kernel_def_lines)} triton kernel definitions found ===")
    for ln in kernel_def_lines[:40]:
        print(ln)

    print(f"\n=== {len(gelu_lines)} lines mentioning gelu/erf ===")
    for ln in gelu_lines[:40]:
        print(ln)

    # Print 10 lines of context around each gelu/erf hit so we can see
    # whether it's inside a kernel body (fused) or a standalone aten call.
    print("\n=== context around gelu/erf hits ===")
    for i, ln in enumerate(lines):
        if "gelu" in ln.lower() or "erf" in ln.lower():
            start = max(0, i - 3)
            end = min(len(lines), i + 3)
            print(f"--- hit at line {i} ---")
            for j in range(start, end):
                print(lines[j])
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
