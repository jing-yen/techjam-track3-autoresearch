# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/v_compile.py` (SDPA + `torch.compile(mode="max-autotune")`) |
| node_id | `v_compile` |
| correctness | ✅ A100-80 (xgph1), `official-safe` (12/12 shapes), float32, max_abs ~1e-6 (fp32 non-associativity, 4 orders under the 0.002 gate) |
| median speedup | **2.18x** |
| geomean speedup | **2.25x** |
| dtype | float32 |
| device | NUS SoC cluster, A100-80 PCIe (`gpu:a100-80:1`, node xgph1) |
| protocol | **official** — warmup=20, repeats=100, rounds=3, alternating baseline/optimized order (matches `torch_transformer_benchmark.py:benchmark_models`); candidate loaded once per sweep (B8+B9 fixed) |
| updated by | opus-1 (iter 6) |

**These are the final, honest numbers** — B8 (official timing protocol) and B9
(candidate reloaded per shape, discarding the `torch.compile` cache) from
TODO.md are both fixed as of this run. An earlier pass (iter 5, noisy
short-warmup protocol) reported `v_compile` at 2.71x median / 2.53x geomean —
that number was inflated by measurement noise, not real; use the numbers on
this page, not that one, in the report.

Lead is narrow: `v_compile` (2.25x geomean) barely beats `v_fused_qkv`
(2.09x) and `best`/seed (1.96x). All three are legitimate, fully-correct
>2x speedups over the baseline; the compile variant's edge is real but
modest, not the 2.7x it first looked like.

Runner-up: `candidates/v_fused_qkv.py` — median 2.16x, geomean 2.09x, 12/12
correct.
Also verified: `candidates/best.py` (seed) — median 2.07x, geomean 1.96x,
12/12 correct.

## Shape #14 — confirmed infeasible on A100-80 (documented limitation)

Attempted all three candidates on the full A100-80 (batch=32, seq=100000,
d_model=1024, num_heads=16). Two distinct, real failure modes — this
matches B5's math exactly, with real numbers instead of estimates:

- **`best.py` / `v_fused_qkv.py`** (eager SDPA): both OOM during warmup,
  *before* even reaching the baseline comparison —
  `CUDA out of memory. Tried to allocate 12.21 GiB. GPU has 79.25 GiB total,
  73.85 GiB already in use, 5.40 GiB free.` This is **not an attention-kernel
  problem** — FlashAttention/mem-efficient SDPA avoids the O(S²) score matrix
  entirely, but the plain `[B,S,D]` activations at this scale (32 × 100,000 ×
  1024, times ~7 live tensors per the organizer's own forward pass) already
  exceed 80GB on their own.
- **`v_compile.py`** (SDPA + `torch.compile(mode="max-autotune")`): never
  reached the OOM point — Slurm killed it at the 15-minute time limit while
  still autotuning. Each of the model's few distinct large matmuls
  (3.2M × 1024 @ 1024×1024) took `torch.compile` **~100s per candidate
  kernel × ~15 candidates ≈ 200-300s to autotune**, and there are several
  such matmuls in the model. `max-autotune` on a shape this large is
  impractical on its own compile-time cost, independent of the memory
  ceiling above.

**Conclusion:** infeasible in fp32 on any single GPU we have access to, for
two independent reasons. Not pursuing fp16/H100 or batch-chunking — out of
scope per TODO.md B5/rubric §3.3 (no "production-ready" gold-plating
expected). Documented here as the required §3.5 limitations reflection.

## Per-shape speedups — `v_compile.py`, A100-80, `official-safe`, official protocol

| shape | passed | baseline_ms | opt_ms | speedup |
|--|--|--|--|--|
| 1 | ✅ | 2.625 | 1.298 | 2.02x |
| 2 | ✅ | 1.808 | 0.370 | 4.89x |
| 3 | ✅ | 1.816 | 0.496 | 3.66x |
| 4 | ✅ | 1.782 | 0.761 | 2.34x |
| 5 | ✅ | 4.648 | 2.915 | 1.59x |
| 7 | ✅ | 1.817 | 0.522 | 3.48x |
| 8 | ✅ | 30.079 | 27.670 | 1.09x |
| 9 | ✅ | 2.000 | 1.634 | 1.22x |
| 10 | ✅ | 2.331 | 1.680 | 1.39x |
| 11 | ✅ | 5.072 | 2.113 | 2.40x |
| 12 | ✅ | 1.767 | 0.925 | 1.91x |
| 13 | ✅ | 62.169 | 15.019 | 4.14x |
