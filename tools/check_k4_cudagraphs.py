"""
K4 -- audit whether v_router2's compile/reduce-overhead routes actually get
static parameter addresses under CUDA Graph Trees, and whether running
several compile/reduce-routed shapes back-to-back in one process (exactly
what a real 13-shape sweep does, and exactly what S2 -- iter 21's unexplained
"reduce wins in isolation, loses in the full sweep" revert -- looks like)
causes any cudagraph pool skip/recapture/static-input-mismatch behavior.

Method: torch._logging.set_logs(cudagraphs=True, recompiles=True) gives
PyTorch's own internal diagnostic for exactly this question -- authoritative,
not inferred. Runs shapes in two orders to isolate cross-shape interaction:
  (a) each compile/reduce shape alone in its own process (baseline)
  (b) all of them back-to-back in one process (S2's actual repro shape)
Any "skipping cudagraphs" / "static input" / "cudagraph tree" warning that
appears only in (b) is the smoking gun for a real cross-instance pool
interaction, not just a hypothetical.
"""
from __future__ import annotations

import argparse
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


def run_shape(cand, shape_id: int, device, log_buf: io.StringIO, warmup: int = 8, active: int = 5):
    cfg_dict = OFFICIAL_SHAPES[shape_id]
    config = TransformerConfig(**cfg_dict)
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

    marker = f">>> SHAPE {shape_id} route={model._impl_name if hasattr(model, '_impl_name') else '?'}"
    print(marker)
    log_buf.write(marker + "\n")
    with torch.no_grad():
        for i in range(warmup):
            model(x, mask)
        torch.cuda.synchronize()
        for i in range(active):
            model(x, mask)
        torch.cuda.synchronize()
    print(f"<<< SHAPE {shape_id} done")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default=os.path.join(_ROOT, "candidates", "v_router2.py"))
    p.add_argument("--shapes", default="1,2,3,4,5,7", help="compile/reduce-routed shapes, sweep order")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda")
    shapes = [int(s) for s in args.shapes.split(",")]

    # Redirect the cudagraphs/recompiles logger output into a buffer we can
    # grep, in addition to letting it print live.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.DEBUG)
    for name in ("torch._inductor.cudagraph_trees", "torch._dynamo", "torch._inductor.compile_fx"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)

    torch._logging.set_logs(cudagraphs=True, recompiles=True, graph_breaks=True)

    spec = importlib.util.spec_from_file_location("cand", args.candidate)
    cand = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cand)

    print(f"=== K4: {len(shapes)} shapes back-to-back in ONE process: {shapes} ===")
    for sid in shapes:
        run_shape(cand, sid, device, log_buf)

    text = log_buf.getvalue()
    print("\n=== captured cudagraph/dynamo log lines (filtered) ===")
    hits = [ln for ln in text.splitlines() if any(
        kw in ln.lower() for kw in ("cudagraph", "static", "skip", "recompil", "pool", "graph break"))]
    for ln in hits:
        print(ln)
    print(f"\n=== {len(hits)} matching lines out of {len(text.splitlines())} total captured ===")

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
