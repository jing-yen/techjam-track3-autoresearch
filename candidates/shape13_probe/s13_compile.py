"""Shape-13 explicit-route probe: forces UserOptimizedTransformer to
_IMPLS["compile"] regardless of the real router table, to compare all six
implementations head-to-head on shape 13 specifically (currently unrouted,
falls to the "compile" fallback by default -- never explicitly compared)."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "v_router2_autotuned", os.path.join(os.path.dirname(__file__), "..", "v_router2_autotuned.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

STRICT_WEIGHT_COPY = False

class UserOptimizedTransformer(_mod.BaselineTransformer):
    def __init__(self, config):
        super().__init__(config)
        del self.layers
        del self.final_norm
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        self._impl_name = "compile"
        self._impl = _mod._IMPLS["compile"](config)

    def forward(self, x, valid_token_mask=None):
        return self._impl(x, valid_token_mask)

def copy_model_weights(baseline, optimized):
    if optimized._impl_name in ("fused", "fusedcg"):
        _mod._fused_copy(baseline, optimized._impl)
    else:
        import torch_transformer_benchmark as ttb
        ttb.copy_model_weights(baseline, optimized._impl, strict=True)
