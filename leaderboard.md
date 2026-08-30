# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/v_compile.py` (SDPA + `torch.compile(mode="max-autotune")`) |
| node_id | `v_compile` |
| correctness | ✅ A100-80 (xgph1), `official-safe` (12/12 shapes), float32, max_abs = 0.0 exact |
| median speedup | **2.71x** |
| geomean speedup | **2.53x** |
| dtype | float32 |
| device | NUS SoC cluster, A100-80 PCIe (`gpu:a100-80:1`, node xgph1) |
| updated by | opus-1 (iter 5) |

**Caveat (see TODO.md B8/B9, still open):** these numbers use the harness's
current timing protocol (warmup 5, repeats 20, rounds 1, sequential blocks) —
noisier and order-biased vs. the official protocol (warmup 20, repeats 100,
rounds 3, alternating). `v_compile` is additionally penalized by B9
(candidate reloaded per shape, discarding the `torch.compile` cache 12x) — its
true steady-state speedup is likely higher than shown here. Do not treat these
as final; re-measure after B8+B9 land.

Runner-up: `candidates/v_fused_qkv.py` — median 2.39x, geomean 2.15x, 12/12
correct.
Also verified: `candidates/best.py` (seed) — median 2.17x, geomean 1.95x,
12/12 correct.

Shape #14 not yet attempted on GPU (B5: ~85 GB fp32 activations, expected to
OOM even on A100-80 — documented limitation, not a blocker per TODO.md).

## Per-shape speedups — `v_compile.py`, A100-80, `official-safe`

| shape | passed | baseline_ms | opt_ms | speedup |
|--|--|--|--|--|
| 1 | ✅ | 2.624 | 1.306 | 2.01x |
| 2 | ✅ | 1.875 | 0.375 | 5.00x |
| 3 | ✅ | 1.890 | 0.454 | 4.17x |
| 4 | ✅ | 1.854 | 0.697 | 2.66x |
| 5 | ✅ | 4.653 | 2.982 | 1.56x |
| 7 | ✅ | 1.880 | 0.558 | 3.37x |
| 8 | ✅ | 30.082 | 27.156 | 1.11x |
| 9 | ✅ | 1.994 | 1.199 | 1.66x |
| 10 | ✅ | 2.339 | 1.210 | 1.93x |
| 11 | ✅ | 5.030 | 1.700 | 2.96x |
| 12 | ✅ | 1.865 | 0.675 | 2.76x |
| 13 | ✅ | 61.500 | 15.513 | 3.96x |
