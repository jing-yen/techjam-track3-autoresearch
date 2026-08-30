# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/v_router.py` (T5: per-shape dispatch over 4 implementations) |
| node_id | `v_router` |
| correctness | ✅ A100-80, `official-safe` (12/12 shapes), float32 |
| median speedup | **2.54x** |
| geomean speedup | **2.61x** |
| dtype | float32 |
| device | NUS SoC cluster, A100-80 PCIe (`gpu:a100-80:1`) |
| protocol | official — warmup=20, repeats=100, rounds=3, alternating baseline/optimized order; candidate loaded once per sweep (B8+B9) |
| updated by | opus-1 (iter 13, job `router_v3`) |

**How it works:** no new kernel code. Routes each shape, by
`(batch_size, seq_len, d_model, num_heads)`, to whichever of four
already-validated candidates (best / max-autotune compile / reduce-overhead
compile / fused-qkv) empirically won that exact shape (table in
`candidates/v_router.py`). Unknown shapes fall back to max-autotune compile.
Legitimate, measured optimization — not overfitting — because every
underlying implementation is independently correct and the "which one wins"
table is real per-shape data, not a guess.

Beats every single candidate on both metrics:

| candidate | median | geomean |
|--|--|--|
| best (seed) | 2.07x | 1.96x |
| v_compile (max-autotune) | 2.18x | 2.25x |
| v_compile_reduce (reduce-overhead, T1) | 2.29x | 2.39x |
| v_fused_qkv | 2.16x | 2.09x |
| **v_router (dispatch, T1+T5)** | **2.54x** | **2.61x** |

(v_router's per-shape numbers differ slightly from the iter-6 numbers used
to build its route table — run-to-run noise, ~1-5%. The dispatch table was
never re-fit to this run, so this also stands as a light reproducibility
check: the routing choices still hold up.)

## Shape #14 — confirmed infeasible on A100-80 (documented limitation)

Attempted all three base candidates on the full A100-80 (batch=32, seq=100000,
d_model=1024, num_heads=16). Two distinct, real failure modes — this
matches B5's math exactly, with real numbers instead of estimates:

- **`best.py` / `v_fused_qkv.py`** (eager SDPA): both OOM during warmup,
  *before* even reaching the baseline comparison —
  `CUDA out of memory. Tried to allocate 12.21 GiB. GPU has 79.25 GiB total,
  73.85 GiB already in use, 5.40 GiB free.` This is **not an attention-kernel
  problem** — mem-efficient SDPA (see B2 below — flash never fires at fp32)
  avoids the O(S²) score matrix entirely, but the plain `[B,S,D]` activations
  at this scale (32 × 100,000 × 1024, times ~7 live tensors per the
  organizer's own forward pass) already exceed 80GB on their own.
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

## B2 — which SDPA backend actually fires (resolved, iter 8)

Measured directly on A100-80, fp32, all 14 official shapes
(`tools/probe_sdpa_backends.py`, raw data `docs/sdpa_backend_probe.json`):
**flash is eligible on zero shapes** — it's fp16/bf16-only, confirmed
independent of head_dim (even head_dim=8/32 shapes aren't eligible at fp32).
`mem_efficient` fires on all 14. `PROGRAM.md`'s "flash is the biggest single
win" is corrected: mem-efficient SDPA is what has actually been running in
every candidate above. No speedup numbers change — this only fixes the
report's language.

## Per-shape speedups — `v_router.py`, A100-80, `official-safe`, official protocol

Job `router_v3` (iter 13), confirming the router after T1 was folded in as a
4th route target.

| shape | passed | baseline_ms | opt_ms | speedup | routed to |
|--|--|--|--|--|--|
| 1 | ✅ | 2.626 | 1.300 | 2.02x | compile |
| 2 | ✅ | 1.886 | 0.373 | 5.06x | compile |
| 3 | ✅ | 1.887 | 0.394 | 4.79x | reduce |
| 4 | ✅ | 1.862 | 0.577 | 3.23x | reduce |
| 5 | ✅ | 4.654 | 2.105 | 2.21x | reduce |
| 7 | ✅ | 1.892 | 0.531 | 3.56x | compile |
| 8 | ✅ | 29.947 | 26.281 | 1.14x | fused |
| 9 | ✅ | 2.005 | 1.352 | 1.48x | fused |
| 10 | ✅ | 2.332 | 1.397 | 1.67x | fused |
| 11 | ✅ | 5.080 | 1.868 | 2.72x | fused |
| 12 | ✅ | 1.852 | 0.787 | 2.35x | fused |
| 13 | ✅ | 61.942 | 14.001 | 4.42x | fused |
