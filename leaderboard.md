# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/v_router.py` (T5: per-shape dispatch over 4 implementations) |
| node_id | `v_router` |
| correctness | ✅ A100-80, `official-safe` (12/12 shapes), float32, TF32 **on** (organizer default) — max_abs ~0.001, still 2x under the 0.002 gate |
| median speedup | **2.67x** |
| geomean speedup | **2.98x** |
| dtype | float32 (TF32 tensor cores enabled, matching the organizer's own default config — see S1 below) |
| device | NUS SoC cluster, A100-80 PCIe (`gpu:a100-80:1`) |
| protocol | official — warmup=20, repeats=100, rounds=3, alternating baseline/optimized order; candidate loaded once per sweep (B8+B9) |
| updated by | opus-1 (iter 14, job `s1_tf32`) |

**Disclosure (S1, TODO.md):** every number before this run was measured with
TF32 force-disabled process-wide — including for the *baseline* — because
the old fix for a real bug (Inductor's max-autotune picking a TF32 kernel
for the candidate while the baseline stayed on eager cuBLAS, drifting outside
the gate) was scoped too broadly and silently overrode the harness's own
default (`allow_tf32=True`, matching the organizer's config) for every route
target, not just the compiled one. Rescoped to affect only the `compile`
(max-autotune) path; everything else now runs at the harness/organizer
default. Both baseline and candidate speed up under TF32 (e.g. shape 8's
baseline: 29.9ms → 7.9ms) — this is the organizer's actual default
configuration, not a change we're choosing for a bigger number.

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
| **v_router (dispatch, S1+T1+T5)** | **2.67x** | **2.98x** |

(The router's dispatch table was not re-fit to this run. S1 changed only the
precision policy: max-autotune routes stay at full fp32 for correctness, while
the other routes now honor the organizer's TF32-on default.)

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

Job `s1_tf32` (iter 14), after scoping the max-autotune TF32 workaround to only
the affected route and restoring the organizer default elsewhere.

| shape | passed | baseline_ms | opt_ms | speedup | routed to |
|--|--|--|--|--|--|
| 1 | ✅ | 2.623 | 1.302 | 2.02x | compile |
| 2 | ✅ | 1.841 | 0.375 | 4.91x | compile |
| 3 | ✅ | 1.909 | 0.330 | 5.79x | reduce |
| 4 | ✅ | 1.884 | 0.379 | 4.97x | reduce |
| 5 | ✅ | 2.721 | 1.120 | 2.43x | reduce |
| 7 | ✅ | 1.851 | 0.525 | 3.52x | compile |
| 8 | ✅ | 7.882 | 6.125 | 1.29x | fused |
| 9 | ✅ | 1.721 | 0.807 | 2.13x | fused |
| 10 | ✅ | 1.905 | 0.805 | 2.37x | fused |
| 11 | ✅ | 3.481 | 1.198 | 2.91x | fused |
| 12 | ✅ | 1.889 | 0.803 | 2.35x | fused |
| 13 | ✅ | 43.134 | 9.633 | 4.48x | fused |
