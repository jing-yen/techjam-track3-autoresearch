# LOG — Running experiment narrative

Human-readable mirror of `journal.jsonl`. Newest entries appended at the bottom.
Each entry: iteration, agent, direction, hypothesis, result, decision.

---

### iter 0 · agent `seed` · direction `sdpa` · **new-best (root)**

**Hypothesis.** Replace the baseline's explicit `[B,H,S,S]` attention with
`F.scaled_dot_product_attention`, preserving exact GELU, fp32-stable softmax,
causal + key-padding masking, and padded-query output zeroing. This should be the
strongest single win on GPU and is the only path that can run shape #14
(seq=100k), where the baseline OOMs.

**Result.** Correctness verified locally on CPU across dev shapes and
`official-safe` (all 12 non-extreme official shapes), with and without padding —
`max_abs ≈ 1e-6`, gate passes on every element. GPU speedup **pending first
cluster run**.

**Decision.** Root of the search tree → `candidates/best.py`.

---

### iter 1-3 · agent `seed` · MPS validation + pre-built variants (pre-cluster)

Before cluster access, validated the pipeline on this Mac's Apple GPU (MPS) and
pre-built the next two playbook candidates so the first cluster batch measures
several at once.

- **SDPA seed on MPS** — correct on every shape; median speedup 1.06 (geomean
  1.02). Attention-heavy shapes gain (11: 1.17×, 13: 1.11×, 8: 1.10×); tiny
  shapes slightly regress (1: 0.89×, 9: 0.84×) because MPS has no op-fusion and
  no FlashAttention. Read: on CUDA the attention wins should widen and
  `torch.compile` should fix the tiny-shape regression.
- **`v_compile.py`** (SDPA + `torch.compile`) — CPU-correct (max_abs ~1e-6).
  Real speedup pending A100 (compile's payoff is on CUDA).
- **`v_fused_qkv.py`** (SDPA + fused `Linear(d,3d)`, non-strict weight copy) —
  correct with/without padding (max_abs ~1e-6); MPS median 1.11×, and edges the
  seed on the tiny shape (1: 0.96× vs 0.89×). Also validates the non-strict
  weight-copy path end-to-end.

**Next:** benchmark all three on A100-80 → first honest numbers.

---

### iter 4 · agent `opus-1` · direction `correctness-fix` · **B1 fixed**

The review layer (Codex) found **B1**: `generate_random_case` returns an
**all-True** mask (never `None`) at `padding_ratio<=0`, so the seed always built
the additive `[B,1,S,S]` mask — disqualifying SDPA's flash backend on every shape
and OOMing shape #14 (~1.2 TB mask). Verified against
`torch_transformer_benchmark.py:255-259`.

**Fix** (best.py + both variants): detect no-padding once (single sync), collapse
the all-True mask to `None` so attention uses `is_causal` (flash/mem-efficient),
keep the additive path only for real padding. Correct at padding 0.0 and 0.3
(max_abs 1.4e-6); instrumented run confirms the all-True mask now routes to
`is_causal` (additive:0, causal:1). Also corrected three false claims in
PROGRAM.md (mask-never-None, #14 is ~18.6 TB not 40 GB/head, TF32 already on).

The multi-agent loop worked: research layer found a real bug, implementation
layer fixed and verified it. Real flash confirmation + #14 feasibility pending A100.

---

### iter 5 · agent `opus-1` · direction `compile` · **new-best: `v_compile`** — first real A100 numbers

Got PyTorch working on the SoC cluster (existing `~/flood_env` install; the
login-node import failure was a login-node memory limit, not a broken
install — it imports fine on GPU compute nodes) and ran all three candidates
on a real A100-80 for the first time. Along the way, fixed two problems this
surfaced that CPU/MPS testing couldn't:

- **GPU-only correctness bug (not in TODO.md, found here):** Inductor's
  `max-autotune` autotunes TF32 GEMM kernels for fp32 inputs by default,
  drifting up to ~0.005 from the baseline's full-fp32 matmuls — above the
  0.002 atol on 9/12 shapes. Fixed by pinning
  `allow_tf32=False` / `float32_matmul_precision("highest")` in all three
  candidates so correctness doesn't depend on the ambient global flag.
- **Cluster infra:** node `xgpj0` fails to `dlopen` `libtorch_global_deps.so`
  even in complete isolation (confirmed: not a race, not quota — file is
  present and correct-size, other `a100-80` nodes import cleanly). Excluded
  it via a new generic `--exclude` passthrough in `runner.py`/`cluster.config.json`.

**Result.** All three candidates pass correctness on `official-safe`
(12/12 shapes, max_abs = 0.0 exact) on A100-80 (node xgph1):

| candidate | median | geomean |
|--|--|--|
| best (seed) | 2.17x | 1.95x |
| **v_compile** | **2.71x** | **2.53x** |
| v_fused_qkv | 2.39x | 2.15x |

**Decision.** `v_compile` → new leaderboard best.

**Caveat — do not over-read these numbers yet.** TODO.md's B8 (harness
timing protocol doesn't match the official warmup/repeats/rounds spec) and B9
(candidate reloaded per shape, discarding `torch.compile`'s cache 12x — this
specifically penalizes `v_compile`) are both still open. Re-measure once they
land; `v_compile`'s real steady-state speedup is likely higher than shown.
Shape #14 (B5) not yet attempted on GPU.

---

### iter 6 · agent `opus-1` · direction `measurement-fix` · **final numbers (B8+B9 fixed)**

Fixed both caveats from iter 5 in `bench_harness.py`: B9 (candidate now
loaded once per sweep, not re-imported per shape) and B8 (timing now mirrors
`torch_transformer_benchmark.py:benchmark_models()` exactly — warmup=20,
repeats=100, rounds=3, alternating baseline/optimized order per round).
9/9 local tests + CPU smoke tests pass. Re-ran all three candidates on
A100-80 (node xgph1, job 775208).

**Result — these are the final, defensible numbers:**

| candidate | median | geomean |
|--|--|--|
| best (seed) | 2.07x | 1.96x |
| **v_compile** | **2.18x** | **2.25x** |
| v_fused_qkv | 2.16x | 2.09x |

All 12/12 `official-safe` shapes still pass on all three (max_abs ~1e-6,
normal fp32 non-associativity — nowhere near the 0.002 gate).

**This corrects iter 5's headline number.** The earlier "v_compile: 2.71x
median / 2.53x geomean" was measured with the noisy short-warmup protocol and
is **not real** — do not cite it. The honest number is that `v_compile`
still leads, but only narrowly: 2.25x geomean vs. 2.09x (`v_fused_qkv`) and
1.96x (seed). All three are legitimate >2x, fully-correct speedups; the
compile variant's edge over a plain SDPA swap is real but modest, not the
gap the first pass suggested.

**Decision.** Leaderboard updated to iter 6's numbers (`v_compile` stays
best, geomean-ranked). Next: shape #14 OOM-confirmation run (documented
limitation, not expected to pass — see B5), then consider T5 (per-shape
dispatch) now that the honest per-shape table shows where the real headroom
is (#8 d=1024 barely moves at 1.09x; #2/#13 already near their ceiling).

---

### iter 7 · agent `opus-1` · direction `shape14-confirmation` · **B5 confirmed with real data**

Ran shape #14 (batch=32, seq=100000, d_model=1024, num_heads=16) on A100-80
for all three candidates, to replace B5's estimated math with real evidence.

**Result — two independent, confirmed failure modes:**
- `best.py` and `v_fused_qkv.py` both OOM during the optimized model's
  warmup, before even reaching the baseline comparison: `Tried to allocate
  12.21 GiB. GPU has 79.25 GiB total, 73.85 GiB already in use, 5.40 GiB
  free.` This is activation memory, not an attention-algorithm problem —
  both candidates already use flash/mem-efficient SDPA, which avoids the
  O(S²) score matrix entirely. The plain `[B,S,D]` activations alone (32 ×
  100,000 × 1024, ~7 live tensors) exceed 80GB.
- `v_compile.py` never got that far — Slurm killed it at the 15-minute
  time limit while still inside `max-autotune`'s kernel search. Each large
  matmul (3.2M×1024 @ 1024×1024) took ~100s per candidate kernel × ~15
  candidates to autotune, and the model has several such matmuls.
  `max-autotune` is separately impractical here on compile-time cost alone.

**Decision.** Confirmed infeasible in fp32 on any GPU available, for two
independent reasons. Documented in `leaderboard.md` with the real error
traces — this is the §3.5 limitations reflection, not a gap to keep
chasing. Not pursuing fp16/H100/batch-chunking (out of scope per TODO.md B5
and rubric §3.3's "no gold-plating" note).

---

### iter 8 · agent `opus-1` · direction `measurement` · **B2 resolved: flash never fires at fp32**

Wrote `tools/probe_sdpa_backends.py` to settle B2: force each SDPA backend
(flash / mem-efficient / math) per official shape via
`torch.nn.attention.sdpa_kernel` on A100-80, fp32, and record which are
eligible. Real dispatch always picks the highest-priority eligible one, so
this directly answers "what actually fires" without needing to trace kernel
launches.

**Result: flash is eligible on zero of the 14 official shapes.** `mem_efficient`
fires on all 14 — including the small-head-dim shapes (#7 head_dim=8,
#1/#12/#13 head_dim=32) that B7's table predicted would be flash-eligible by
head_dim alone. Confirms flash is fp16/bf16-only *unconditionally* at this
dtype, independent of head_dim. Raw data: `docs/sdpa_backend_probe.json`.

**This settles an inaccurate claim.** `PROGRAM.md:61` calls flash "the
biggest single win" — false as measured. Mem-efficient SDPA is what has
actually been running in every candidate, in every benchmark run in this
repo so far. No existing speedup number changes (mem-efficient was already
firing under the hood); this only corrects what the report should call it.

**Decision.** B2 closed. TODO.md updated with the real finding.

---

### iter 9 · agent `opus-1` · direction `dispatch` · **new-best: `v_router` (T5)**

Built `candidates/v_router.py`: routes each shape to whichever of
`best`/`v_compile`/`v_fused_qkv` empirically won it in iter 6's official-
protocol run, by `(batch_size, seq_len, d_model, num_heads)`. Zero new
kernel code — just dispatch over three already-correct, already-measured
implementations.

**Hit a real bug on the first attempt.** v1 dynamically imported the sibling
candidate files by a `__file__`-relative path. Passed locally (CPU, same
directory) but failed all 12 shapes on the cluster with
`FileNotFoundError: .../router_official/best.py` — `runner.py`'s ssh mode
scp's each candidate alone into a per-job temp directory, so `__file__`
never points at `candidates/`. Every other pre-built candidate in this repo
already avoids this by being a self-contained snapshot (see v_compile.py's
own docstring) — v_router.py now follows the same convention, inlining all
three implementations. Verified the fix by reproducing the exact scp
scenario locally (copied the file into an isolated temp dir, ran the
harness against that copy) before spending cluster time again.

**Result.** Beats every single candidate, on both metrics:

| candidate | median | geomean |
|--|--|--|
| best (seed) | 2.07x | 1.96x |
| v_compile | 2.18x | 2.25x |
| v_fused_qkv | 2.16x | 2.09x |
| **v_router** | **2.27x** | **2.47x** |

12/12 `official-safe` shapes correct. Per-shape speedups differ ~1-5% from
the iter-6 numbers the route table was built from (normal run-to-run noise)
but every routing choice still held — a light reproducibility check on the
underlying data, not just the router itself.

**Decision.** `v_router` → new leaderboard best.

---

### iter 10-12 · agent `opus-1` · T1 confirmed + folded, T6 in progress (unverified)

**T1** (`candidates/v_compile_reduce.py`, reduce-overhead compile): confirmed
on A100-80, official protocol, 12/12 correct — median 2.29x, geomean 2.39x
standalone. Beats every existing candidate on shapes 3 (4.83x), 4 (3.24x),
5 (2.19x). Folded into `v_router.py` as a 4th route target for those three
shapes (commit `7a10116`).

**Router re-confirmation — pending, not verified.** Submitted a rerun
(`router_v2`) to confirm the T1-updated router's real aggregate number, but
the SSH ControlMaster connection dropped before results could be pulled.
The Slurm job itself is unaffected by an SSH disconnect and likely
completed, but **do not cite a new median/geomean for `v_router` until this
is re-checked** — `leaderboard.md` still shows the last *confirmed* number
(2.27x/2.47x, pre-T1).

**T6** (`candidates/v_amp.py`, fp16 via `torch.autocast`): built as a more
careful follow-up after the naive blanket-fp16 cast failed correctness on
11/12 shapes (max_abs 0.006-0.009). Only CPU-smoke-tested so far (no CUDA
autocast path on CPU, falls back to fp32 — same math as `best.py`). **Not
yet run on GPU.** Motivating finding still stands regardless of outcome:
`tools/probe_sdpa_backends.py --dtype float16` showed flash SDPA eligible
on all 14 official shapes (vs. zero at fp32), correcting B7's head_dim-cap
assumption for this torch/CUDA version.

**Next when work resumes:** reconnect SSH, check `squeue`/`.runs/router_v2`
and `.runs/` for v_amp GPU results, verify before writing any new number.

---

### iter 13 · agent `opus-1` · **router re-confirmed: 2.54x median / 2.61x geomean**

SSH reconnected (the earlier disconnect was a dead ControlMaster socket —
re-established with `ssh -fN xlogin`). The iter-11 `router_v2` job never
actually reached the cluster (the ssh call failed before sbatch submission).
Resubmitted clean as `router_v3`.

**Confirmed, real numbers:** 12/12 `official-safe` shapes correct, median
**2.54x**, geomean **2.61x** — up from 2.27x/2.47x pre-T1. `leaderboard.md`
updated. This is now the number to cite.

---

### iter 14 · agent `opus-1` · direction `precision-scope` · **new best: 2.67x median / 2.98x geomean**

**Hypothesis.** The previous import-time TF32 disable was process-global and
silently forced both the baseline and every router target to full fp32, even
though only Inductor's max-autotune path had shown asymmetric TF32-kernel
correctness drift. Scope the full-precision workaround to `compile` only and
restore the organizer's TF32-on defaults for eager fused/reduce routes.

**Result.** Job `s1_tf32` on A100-80, official protocol: all 12/12
`official-safe` shapes pass. Median **2.667x**, geomean **2.977x**; worst
max_abs is **0.001135**, still below the 0.002 absolute gate. Shape #8's
baseline/optimized times fell from 29.95/26.28 ms to 7.88/6.13 ms. The gain is
from matching the organizer's published default, not selecting a favorable
non-default benchmark mode.

**Decision.** New best. Keep the scoped precision policy and disclose the
historical measurement correction in the leaderboard/report.
