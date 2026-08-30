#!/usr/bin/env python3
"""
B2 -- log which SDPA backend actually fires, per official shape.

For each shape, builds representative q/k/v tensors (matching best.py's
SDPASelfAttention: [B, H, S, head_dim], causal, no padding) and forces each
of FLASH / EFFICIENT_ATTENTION / CUDNN_ATTENTION / MATH via
torch.nn.attention.sdpa_kernel, recording which ones the shape is even
eligible for.

CUDNN_ATTENTION added after the fact (a deep-research literature review
flagged that PyTorch 2.5+ ships a cuDNN-backed SDPA path we had never
probed -- confirmed present in our torch 2.10.0+cu128 via the SDPBackend
enum). This is a genuinely distinct kernel from flash/mem-efficient, not
just a synonym, so its eligibility and relative speed are worth knowing
even though our production candidates never called it explicitly (PyTorch's
own dispatcher already tries every available backend and silently falls
back, so if cudnn were faster and eligible it would already be firing by
default -- this probe makes that visible instead of assumed).

Because we cannot be certain of this torch version's exact internal
priority ordering among 4 backends (undocumented and has changed across
releases), "fires" below is reported per our own fixed *check* order
(flash, mem_efficient, cudnn, math) as a best-effort proxy, same caveat as
before B2 added cudnn -- treat "eligible" as authoritative, "fires" as an
estimate.

Usage: PYTHON bench_harness.py's PYTHON, on a CUDA node:
    python tools/probe_sdpa_backends.py [--dtype float32|float16] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from bench_harness import OFFICIAL_SHAPES  # noqa: E402


def eligible_backends(shape_cfg: dict, device: torch.device, dtype: torch.dtype) -> dict:
    b, s = shape_cfg["batch_size"], shape_cfg["seq_len"]
    d, h = shape_cfg["d_model"], shape_cfg["num_heads"]
    head_dim = d // h
    q = torch.randn(b, h, s, head_dim, device=device, dtype=dtype)
    k = torch.randn(b, h, s, head_dim, device=device, dtype=dtype)
    v = torch.randn(b, h, s, head_dim, device=device, dtype=dtype)

    result = {}
    for name, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, k, v, is_causal=True)
            result[name] = True
        except RuntimeError as e:
            result[name] = False
            result[f"{name}_error"] = str(e)[:200]
        except torch.OutOfMemoryError as e:  # huge shapes (e.g. #14)
            result[name] = None
            result[f"{name}_error"] = f"OOM: {str(e)[:150]}"
            torch.cuda.empty_cache()
    del q, k, v
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--shapes", default="all", help="'all' or comma list of official shape ids")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    if args.shapes == "all":
        ids = sorted(OFFICIAL_SHAPES.keys())
    else:
        ids = [int(t) for t in args.shapes.split(",") if t.strip()]

    rows = []
    for sid in ids:
        cfg = OFFICIAL_SHAPES[sid]
        head_dim = cfg["d_model"] // cfg["num_heads"]
        try:
            elig = eligible_backends(cfg, device, dtype)
            fires = next((n for n in ("flash", "mem_efficient", "cudnn", "math") if elig.get(n)), "none")
        except Exception as e:  # noqa: BLE001
            elig, fires = {"error": f"{type(e).__name__}: {e}"}, "error"
        rows.append({
            "shape_id": sid, "config": cfg, "head_dim": head_dim,
            "fires": fires, "eligible": elig,
        })
        print(f"shape {sid:>2}  head_dim={head_dim:>4}  fires={fires:<14}  "
              f"flash={elig.get('flash')}  mem_eff={elig.get('mem_efficient')}  "
              f"cudnn={elig.get('cudnn')}  math={elig.get('math')}")

    out = {"device": device.type, "dtype": args.dtype, "torch": torch.__version__, "rows": rows}
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
