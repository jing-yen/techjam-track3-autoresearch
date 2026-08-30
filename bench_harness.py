#!/usr/bin/env python3
"""
JSON-emitting benchmark harness for the autoresearch swarm.

Wraps the organizer's ``torch_transformer_benchmark.py`` (imported read-only,
kept pristine as the reference) and:

  1. Dynamically loads a *candidate* module that defines
     ``UserOptimizedTransformer(BaselineTransformer)``.
  2. For each requested shape, copies the baseline weights into the candidate,
     checks per-element correctness (the exact rule from the benchmark:
     ``abs<=atol OR rel<=rtol``, every element must pass), and measures
     latency/speedup with the benchmark's own CUDA-event timing.
  3. Emits a single machine-readable JSON object to stdout (and optionally to
     ``--out``) that the runner / Workflow parses.

Design notes
------------
* We *reuse* the organizer's ``BaselineTransformer``, ``compare_outputs``,
  ``generate_random_case``, ``benchmark_once``, ``warmup_model`` and
  ``copy_model_weights`` rather than reimplementing them, so our reported
  numbers use the same code path the organizer scores against.
* Baseline OOM on the huge shape (#14, seq=100k) is expected on any single GPU
  (~40 GB/head). We catch it: the shape is marked ``baseline_oom``, correctness
  is skipped for it (no reference available), and we still time the candidate
  (which can run via memory-efficient SDPA) so "runs at all" is reportable.

Candidate module contract
--------------------------
A candidate file must define::

    from torch_transformer_benchmark import BaselineTransformer
    class UserOptimizedTransformer(BaselineTransformer):
        def forward(self, x, valid_token_mask=None): ...

Optional module-level knobs:
    STRICT_WEIGHT_COPY = True   # set False if you renamed/fused params
    def copy_model_weights(baseline, optimized): ...   # custom weight mapping
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import traceback
from typing import Dict, List, Optional

import torch

# Make "from torch_transformer_benchmark import ..." resolve from this dir,
# both for us and for the candidate module we load.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch_transformer_benchmark as ttb  # noqa: E402  (organizer reference)


# ---------------------------------------------------------------------------
# Shape catalog: the 14 official shapes (PDF Appendix "Test Shapes") plus a few
# tiny dev shapes that run fast on a laptop CPU/MPS for plumbing checks.
# Fields: batch_size, seq_len, d_model, num_heads, ffn_dim, num_layers, causal
# ("QKV Dim" in the PDF == d_model.)
# ---------------------------------------------------------------------------
OFFICIAL_SHAPES: Dict[int, dict] = {
    1:  dict(batch_size=64,    seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    2:  dict(batch_size=1,     seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    3:  dict(batch_size=4,     seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    4:  dict(batch_size=16,    seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    5:  dict(batch_size=128,   seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    6:  dict(batch_size=10000, seq_len=128,    d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    7:  dict(batch_size=64,    seq_len=128,    d_model=32,   num_heads=4,  ffn_dim=32,   num_layers=4, causal=True),
    8:  dict(batch_size=64,    seq_len=128,    d_model=1024, num_heads=4,  ffn_dim=1024, num_layers=4, causal=True),
    9:  dict(batch_size=64,    seq_len=128,    d_model=128,  num_heads=1,  ffn_dim=128,  num_layers=4, causal=True),
    10: dict(batch_size=64,    seq_len=128,    d_model=128,  num_heads=2,  ffn_dim=128,  num_layers=4, causal=True),
    11: dict(batch_size=64,    seq_len=128,    d_model=128,  num_heads=16, ffn_dim=128,  num_layers=4, causal=True),
    12: dict(batch_size=64,    seq_len=32,     d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    13: dict(batch_size=64,    seq_len=1024,   d_model=128,  num_heads=4,  ffn_dim=128,  num_layers=4, causal=True),
    14: dict(batch_size=32,    seq_len=100000, d_model=1024, num_heads=16, ffn_dim=1024, num_layers=2, causal=True),
}

# Tiny shapes for laptop plumbing checks (ids are negative so they never clash
# with official ids). Fast on CPU/MPS.
DEV_SHAPES: Dict[int, dict] = {
    -1: dict(batch_size=2, seq_len=16, d_model=64, num_heads=4, ffn_dim=64,  num_layers=2, causal=True),
    -2: dict(batch_size=3, seq_len=24, d_model=32, num_heads=2, ffn_dim=48,  num_layers=1, causal=False),
    -3: dict(batch_size=1, seq_len=8,  d_model=16, num_heads=1, ffn_dim=16,  num_layers=1, causal=True),
}


def resolve_device(name: str) -> torch.device:
    """auto -> cuda, else mps, else cpu. Explicit names honored."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_candidate(path: str):
    """Import a candidate module from a file path and return its
    UserOptimizedTransformer class + optional knobs."""
    spec = importlib.util.spec_from_file_location("candidate_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise -> caught by caller per-shape
    if not hasattr(module, "UserOptimizedTransformer"):
        raise AttributeError(f"{path} does not define UserOptimizedTransformer")
    return module


def _median(vals: List[float]) -> float:
    return statistics.median(vals) if vals else float("nan")


def _geomean(vals: List[float]) -> float:
    vals = [v for v in vals if v > 0 and math.isfinite(v)]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def eval_one_shape(
    shape_id: int,
    cfg_dict: dict,
    module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    accuracy_trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict:
    """Evaluate one shape. Never raises: all failures are captured into the
    returned record so one bad shape can't abort the sweep.

    ``module`` is a candidate module already imported by the caller (B9: the
    candidate is loaded ONCE for the whole sweep, not per shape, so a
    ``torch.compile`` cache built on shape N survives into shape N+1)."""
    rec: dict = {
        "shape_id": shape_id,
        "config": cfg_dict,
        "status": "ok",
        "passed": None,
        "max_abs": None,
        "max_rel": None,
        "failed_elems": None,
        "total_elems": None,
        "baseline_ms": None,
        "opt_ms": None,
        "speedup": None,
        "error": None,
    }
    try:
        config = ttb.TransformerConfig(**cfg_dict)
        config.validate()

        strict = bool(getattr(module, "STRICT_WEIGHT_COPY", True))
        custom_copy = getattr(module, "copy_model_weights", None)

        baseline = ttb.BaselineTransformer(config)
        optimized = module.UserOptimizedTransformer(config)
        if callable(custom_copy):
            custom_copy(baseline, optimized)
        else:
            ttb.copy_model_weights(baseline, optimized, strict=strict)

        baseline = baseline.to(device=device, dtype=dtype).eval()
        optimized = optimized.to(device=device, dtype=dtype).eval()

        # ---- correctness ----
        baseline_oom = False
        max_abs = 0.0
        max_rel = 0.0
        failed = 0
        total = 0
        all_passed = True
        with torch.inference_mode():
            for trial in range(accuracy_trials):
                x, mask = ttb.generate_random_case(
                    config=config, device=device, dtype=dtype,
                    seed=seed + trial, padding_ratio=padding_ratio,
                    input_scale=input_scale,
                )
                try:
                    ref = baseline(x, mask)
                except RuntimeError as e:
                    if _is_oom(e):
                        baseline_oom = True
                        break
                    raise
                cand = optimized(x, mask)
                result = ttb.compare_outputs(ref, cand, rtol=rtol, atol=atol)
                all_passed &= result.passed
                max_abs = max(max_abs, result.max_abs_error)
                max_rel = max(max_rel, result.max_relative_error)
                failed += result.failed_elements
                total += result.total_elements

        if baseline_oom:
            rec["status"] = "baseline_oom"
            rec["passed"] = None  # no reference available -> correctness N/A
        else:
            rec["passed"] = bool(all_passed)
            rec["max_abs"] = max_abs
            rec["max_rel"] = max_rel
            rec["failed_elems"] = failed
            rec["total_elems"] = total
            if not all_passed:
                rec["status"] = "correctness_fail"

        # ---- timing (skip if correctness already failed, unless caller wants it) ----
        # Mirrors the organizer's benchmark_models() exactly (B8): one fixed
        # input, warm up both models once, then alternate measurement order
        # per round (baseline-then-optimized on even rounds, reversed on odd)
        # to cancel thermal/clock-order bias instead of timing in two
        # sequential blocks.
        if rec["status"] in ("ok", "baseline_oom"):
            x, mask = ttb.generate_random_case(
                config=config, device=device, dtype=dtype,
                seed=seed + 100000, padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            ttb.warmup_model(optimized, x, mask, warmup, device)
            if not baseline_oom:
                try:
                    ttb.warmup_model(baseline, x, mask, warmup, device)
                except RuntimeError as e:
                    if _is_oom(e):
                        baseline_oom = True
                        rec["status"] = "baseline_oom"
                    else:
                        raise

            opt_samples: List[float] = []
            base_samples: List[float] = []
            try:
                for round_index in range(rounds):
                    if round_index % 2 == 0:
                        if not baseline_oom:
                            base_samples.extend(ttb.benchmark_once(baseline, x, mask, repeats, device))
                        opt_samples.extend(ttb.benchmark_once(optimized, x, mask, repeats, device))
                    else:
                        opt_samples.extend(ttb.benchmark_once(optimized, x, mask, repeats, device))
                        if not baseline_oom:
                            base_samples.extend(ttb.benchmark_once(baseline, x, mask, repeats, device))
            except RuntimeError as e:
                if _is_oom(e):
                    baseline_oom = True
                    rec["status"] = "baseline_oom"
                    base_samples = []
                else:
                    raise

            rec["opt_ms"] = _median(opt_samples)
            if not baseline_oom and base_samples:
                rec["baseline_ms"] = _median(base_samples)
                if rec["opt_ms"] and rec["opt_ms"] > 0:
                    rec["speedup"] = rec["baseline_ms"] / rec["opt_ms"]
    except Exception as e:  # noqa: BLE001 - capture everything into the record
        rec["status"] = "candidate_error"
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    return rec


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda oom" in msg or "mps" in msg and "memory" in msg


def select_shapes(spec: str) -> Dict[int, dict]:
    if spec == "dev":
        return dict(DEV_SHAPES)
    if spec == "all":
        return dict(OFFICIAL_SHAPES)
    if spec == "official-safe":
        # Shape #6 was measured correct on A100-40 (S4, iter 15) and uses only
        # ~6.7 GB. Shape #14 remains infeasible in fp32 on an 80 GB GPU.
        return {k: v for k, v in OFFICIAL_SHAPES.items() if k != 14}
    ids = [int(tok) for tok in spec.split(",") if tok.strip()]
    catalog = {**OFFICIAL_SHAPES, **DEV_SHAPES}
    out = {}
    for i in ids:
        if i not in catalog:
            raise SystemExit(f"unknown shape id {i}")
        out[i] = catalog[i]
    return out


def run(args: argparse.Namespace) -> dict:
    device = resolve_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    shapes = select_shapes(args.shapes)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    # B9: load the candidate ONCE for the whole sweep. Previously this ran
    # inside the per-shape loop, re-importing (and thus re-JIT-compiling,
    # for torch.compile candidates) 12x per sweep.
    try:
        module = load_candidate(args.candidate)
        load_error = None
    except Exception as e:  # noqa: BLE001
        module = None
        load_error = f"{type(e).__name__}: {e}"

    per_shape: List[dict] = []
    for sid, cfg in shapes.items():
        if module is None:
            per_shape.append({
                "shape_id": sid, "config": cfg, "status": "candidate_error",
                "passed": None, "max_abs": None, "max_rel": None,
                "failed_elems": None, "total_elems": None,
                "baseline_ms": None, "opt_ms": None, "speedup": None,
                "error": load_error,
            })
            continue
        rec = eval_one_shape(
            sid, cfg, module, device, dtype,
            accuracy_trials=args.accuracy_trials, seed=args.seed,
            padding_ratio=args.padding_ratio, input_scale=args.input_scale,
            rtol=args.rtol, atol=args.atol,
            warmup=args.warmup, repeats=args.repeats, rounds=args.rounds,
        )
        per_shape.append(rec)

    # Aggregate. The correctness GATE = every runnable shape passed
    # (baseline_oom shapes are excluded from the gate since they have no ref).
    runnable = [r for r in per_shape if r["status"] not in ("baseline_oom",)]
    correctness_passed = bool(runnable) and all(r["passed"] for r in runnable)
    speedups = [r["speedup"] for r in per_shape if isinstance(r.get("speedup"), (int, float)) and r["speedup"] and math.isfinite(r["speedup"])]

    errors = [f"shape {r['shape_id']}: {r['error']}" for r in per_shape if r["status"] == "candidate_error"]

    return {
        "candidate": os.path.abspath(args.candidate),
        "device": device.type,
        "dtype": args.dtype,
        "torch": torch.__version__,
        "rtol": args.rtol,
        "atol": args.atol,
        "shapes_spec": args.shapes,
        "correctness_passed": correctness_passed,
        "median_speedup": _median(speedups),
        "geomean_speedup": _geomean(speedups),
        "n_shapes": len(per_shape),
        "n_speedups": len(speedups),
        "per_shape": per_shape,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JSON-emitting benchmark harness")
    p.add_argument("--candidate", required=True, help="path to candidate .py defining UserOptimizedTransformer")
    p.add_argument("--shapes", default="dev", help="'dev' | 'all' | 'official-safe' | comma list of ids")
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0|mps")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--padding-ratio", type=float, default=0.0)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--accuracy-trials", type=int, default=3)
    p.add_argument("--rtol", type=float, default=0.02)
    p.add_argument("--atol", type=float, default=0.002)
    p.add_argument("--seed", type=int, default=1234)
    # Match the organizer's official protocol defaults exactly (B8):
    # torch_transformer_benchmark.py --warmup/--repeats/--benchmark-rounds.
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    p.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default=None, help="also write JSON to this path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    # exit code: 0 if correct, 2 if not (mirrors the organizer benchmark)
    return 0 if result["correctness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
