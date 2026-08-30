"""Test fixture: a candidate whose forward always raises, to verify the harness
captures the error into the per-shape record instead of crashing the sweep."""
from torch_transformer_benchmark import BaselineTransformer


class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None):
        raise RuntimeError("intentional failure for testing")
