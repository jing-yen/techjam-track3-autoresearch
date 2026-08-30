# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/v_router2.py` (T5 dispatch + B10 mask-cache + T6 AMP on 6/8/13) |
| node_id | `v_router2` |
| correctness | ✅ A100-80, `official-safe` (13/13 shapes incl. shape 6), float32, TF32 **on** (organizer default) for non-compile routes — worst max_abs 0.00168, atol 0.002 |
| median speedup | **2.89x** |
| geomean speedup | **3.58x** |
| dtype | float32 base, with `torch.autocast(fp16)` on shapes #6/#8/#13 specifically (T6) |
| device | NUS SoC cluster, A100-80 PCIe (`gpu:a100-80:1`) |
| protocol | official — warmup=20, repeats=100, rounds=3, alternating baseline/optimized order; candidate loaded once per sweep (B8+B9) |
| updated by | opus-1 (iter 21, job `step4_confirm_a80`) |

**On top of S1's TF32 disclosure below:** `v_router2` adds three more
confirmed improvements over the S1-era `v_router`:
- **B10** (mask-cache): removes a per-forward GPU→CPU `.all()` sync by
  caching the mask classification (keyed on tensor identity+version+shape).
  Alone: geomean 2.92x → 3.30x (A100-40).
- **T6** (fp16 `autocast` on shapes #6/#8/#13 only): confirmed via a full
  13-shape sweep to be the *only* shapes where fp16 beats the fp32 route —
  #13 4.48x → 10.50x, #8 1.28x → 1.68x, #6 2.81x → 2.89x.
- **S2 attempted-and-reverted** (see journal iter 20-21): routing shapes
  #9/#10 to `reduce` looked good isolated (3.65x/3.98x) but *regressed*
  inside the full sweep (1.81x/2.01x, worse than `fused`'s in-sweep
  2.09x/2.33x) — likely `torch.compile`/CUDA-graph interaction between the
  5 shapes now compiling `reduce` in one process. Reverted; kept on `fused`.
  Real lesson: validate route changes in the full sweep, not pairwise.

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

## Shape #14 — UPDATE (S5, iter 22): now runs via batch-chunking

**Superseded below.** The original three eager/compiled candidates all
failed (kept below for the record — real, useful evidence of *why* naive
implementations don't work). But `candidates/v_chunked14.py` (S5: exact
batch-chunking, `chunk_size=4` — B=32 is 32 independent sequences, nothing
couples them, so splitting into groups and concatenating changes nothing
mathematically) **completes a full forward pass on real A100-80 hardware in
~74.6s** (job `777316`, reduced timing protocol — the full official 320-call
protocol didn't finish within a 30-min Slurm limit). No correctness
comparison is possible directly (the baseline needs 18.6 TB to materialize
its score matrix — B5's original finding still holds for the *reference*),
but the chunking mechanism itself is GPU-confirmed exact on shapes #8/#13
(job `s5_validate_a80`: max_abs 0.0009-0.0011, well within the 0.002 gate),
which do have references. Report: "runs, ~74.6s/forward, mechanism proven
exact where a reference exists, unverifiable directly on #14 itself."
~75s/forward is slow relative to the other shapes' millisecond scale;
CUDA-stream pipelining across the 8 sequential chunks is the natural next
lever if that becomes a real requirement — not yet pursued.

## Shape #14 — original attempt: eager/compiled candidates all failed

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

**Original conclusion (since revised by S5 above):** these three specific
implementations are infeasible on any single GPU, for two independent
reasons — eager SDPA's plain activation memory, and max-autotune's
compile-time cost. That conclusion still holds *for these three
candidates*. It does not generalize to "shape 14 is infeasible" — S5's
batch-chunked candidate above shows the actual ceiling was activation
memory specifically, and that a targeted ~20-line fix clears it.

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
