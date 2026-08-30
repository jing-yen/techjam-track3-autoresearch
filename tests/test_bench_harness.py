"""Unit tests for bench_harness. Runs on CPU with tiny dev shapes.

Run under pytest, or standalone: `python tests/test_bench_harness.py`.
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import bench_harness as bh  # noqa: E402

IDENTITY = os.path.join(HERE, "fixtures", "identity_candidate.py")
SDPA = os.path.join(ROOT, "candidates", "best.py")
BROKEN = os.path.join(HERE, "fixtures", "broken_candidate.py")


def _args(candidate, shapes="dev", padding_ratio=0.0):
    return argparse.Namespace(
        candidate=candidate, shapes=shapes, device="cpu", dtype="float32",
        padding_ratio=padding_ratio, input_scale=1.0, accuracy_trials=3, rtol=0.02,
        atol=0.002, seed=1234, warmup=1, repeats=3, rounds=1,
        matmul_precision="high", allow_tf32=True, out=None,
    )


REQUIRED_KEYS = {
    "candidate", "device", "dtype", "torch", "rtol", "atol", "correctness_passed",
    "median_speedup", "geomean_speedup", "n_shapes", "n_speedups", "per_shape", "errors",
}


def test_identity_passes_and_schema():
    d = bh.run(_args(IDENTITY))
    assert REQUIRED_KEYS <= set(d), f"missing keys: {REQUIRED_KEYS - set(d)}"
    assert d["correctness_passed"] is True
    assert all(r["passed"] for r in d["per_shape"])
    assert math.isfinite(d["median_speedup"])


def test_sdpa_correct_no_padding():
    d = bh.run(_args(SDPA))
    assert d["correctness_passed"] is True
    for r in d["per_shape"]:
        assert r["passed"], f"shape {r['shape_id']} failed: {r}"
        assert r["max_abs"] <= 0.002 + 1e-9 or r["max_rel"] <= 0.02 + 1e-9


def test_sdpa_correct_with_padding():
    d = bh.run(_args(SDPA, padding_ratio=0.4))
    assert d["correctness_passed"] is True, d["per_shape"]


def test_broken_candidate_is_captured_not_raised():
    # A candidate that raises inside forward must be recorded as candidate_error,
    # never crash the sweep.
    d = bh.run(_args(BROKEN))
    assert d["correctness_passed"] is False
    assert any(r["status"] == "candidate_error" for r in d["per_shape"])
    assert d["errors"]


def test_unknown_shape_id_raises():
    try:
        bh.select_shapes("999")
    except SystemExit:
        return
    raise AssertionError("expected SystemExit for unknown shape id")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
