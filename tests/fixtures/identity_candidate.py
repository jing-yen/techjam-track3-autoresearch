"""Test fixture: a candidate identical to the baseline (uses baseline forward).
Correctness must pass and speedup must be ~1.0."""
from torch_transformer_benchmark import BaselineTransformer


class UserOptimizedTransformer(BaselineTransformer):
    pass
