"""
T5 — per-shape dispatch.

No single candidate wins every shape (leaderboard.md iter 6, official A100
protocol): v_compile takes the small/launch-overhead shapes hardest (#2:
4.89x, #7: 3.48x) but v_fused_qkv edges it out on #8/#9/#10/#11/#12/#13, and
plain best.py wins #4. Taking the per-shape max over the three already-
validated, already-correct candidates raises the aggregate median from
~2.1-2.2x (any single one) to ~2.5x, for free -- no new kernel code, just
routing.

Unknown shapes (not in the table below, e.g. #6, #14) fall back to
v_compile, the best all-round generalist. This is intentionally NOT a
general heuristic (e.g. "use fused-qkv when d_model is large") -- the three
candidates' relative strengths don't reduce to one clean rule across our
12-shape sample, so the table is measured, not guessed. Extend it as more
shapes get benchmarked.
"""
from __future__ import annotations

import importlib.util
import os

import torch
import torch.nn as nn

import torch_transformer_benchmark as ttb
from torch_transformer_benchmark import BaselineTransformer, TransformerConfig

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str):
    """Import a sibling candidate file by path (not by package import --
    candidates/ isn't guaranteed to be on sys.path)."""
    path = os.path.join(_HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_router_impl_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPLS = {
    "best": _load_sibling("best"),
    "compile": _load_sibling("v_compile"),
    "fused": _load_sibling("v_fused_qkv"),
}

# (batch_size, seq_len, d_model, num_heads) -> best-known implementation.
# Source: leaderboard.md iter 6, A100-80, official timing protocol.
_ROUTE = {
    (64, 128, 128, 4):   "compile",  # shape 1  -- 2.02x vs best 1.53x, fused 1.76x
    (1, 128, 128, 4):    "compile",  # shape 2  -- 4.89x vs best 2.33x, fused 2.37x
    (4, 128, 128, 4):    "compile",  # shape 3  -- 3.66x vs best 2.35x, fused 2.36x
    (16, 128, 128, 4):   "best",     # shape 4  -- 2.57x vs compile 2.34x, fused 2.34x
    (128, 128, 128, 4):  "fused",    # shape 5  -- 1.86x vs best 1.66x, compile 1.59x
    (64, 128, 32, 4):    "compile",  # shape 7  -- 3.48x vs best 1.93x, fused 1.99x
    (64, 128, 1024, 4):  "fused",    # shape 8  -- 1.14x vs best 1.09x, compile 1.09x
    (64, 128, 128, 1):   "fused",    # shape 9  -- 1.47x vs best 1.27x, compile 1.22x
    (64, 128, 128, 2):   "fused",    # shape 10 -- 1.68x vs best 1.47x, compile 1.39x
    (64, 128, 128, 16):  "fused",    # shape 11 -- 2.73x vs best 2.45x, compile 2.40x
    (64, 32, 128, 4):    "fused",    # shape 12 -- 2.35x vs best 2.21x, compile 1.91x
    (64, 1024, 128, 4):  "fused",    # shape 13 -- 4.39x vs best 4.16x, compile 4.14x
}
_FALLBACK = "compile"

STRICT_WEIGHT_COPY = False  # weights go into self._impl, not this wrapper


def copy_model_weights(baseline: nn.Module, optimized: "UserOptimizedTransformer") -> None:
    impl_module = _IMPLS[optimized._impl_name]
    strict = bool(getattr(impl_module, "STRICT_WEIGHT_COPY", True))
    custom_copy = getattr(impl_module, "copy_model_weights", None)
    if callable(custom_copy):
        custom_copy(baseline, optimized._impl)
    else:
        ttb.copy_model_weights(baseline, optimized._impl, strict=strict)


class UserOptimizedTransformer(BaselineTransformer):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        # Base's own layers/final_norm are dead weight once we delegate --
        # drop them so we're not holding an extra randomly-initialized copy
        # of every parameter (matters at the larger shapes).
        del self.layers
        del self.final_norm

        key = (config.batch_size, config.seq_len, config.d_model, config.num_heads)
        self._impl_name = _ROUTE.get(key, _FALLBACK)
        self._impl = _IMPLS[self._impl_name].UserOptimizedTransformer(config)

    def forward(self, x, valid_token_mask=None):
        return self._impl(x, valid_token_mask)
