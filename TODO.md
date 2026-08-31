# TODO — Idea backlog & claim board

Claim an `Open` item by moving it to `In progress` with your agent-id (see
`AGENTS.md` §1). Ranked by expected impact. Add follow-ups freely.

**Every item below carries `file:line` evidence. Do not add an item without it.**
Mark anything unmeasured `UNVERIFIED` and say what would settle it.

---

## What actually scores (§3.6) — read before ranking anything

| Criterion | Weight |
|---|---|
| Technical Execution | 35% |
| Innovation & Problem Insight | 20% |
| Impact & Relevance | 20% |
| Feasibility & Practicality | 15% |
| Presentation & Communication | 10% (final event only) |

**There is no line item for raw speedup.** Nothing in the rubric scores
milliseconds. What scores is well-structured code, thoughtful architecture,
sharpness of problem understanding, and a demo that runs reliably.

§3.2: "The use of AI tools is encouraged so that the participants can implement
different kernels for different input shapes in limited time."
§3.5: "Provide a clear tech report including details on the AI skills/tools used
to get bonus points."

- The autoresearch swarm is **not a detour from the task**. It is the strongest
  available answer to Innovation (20%) and to half of Technical Execution (35%),
  and it is explicitly what the organizers asked for. A measured 1.4x with a
  documented agent loop, a correctness ledger and a per-shape dispatch table
  beats an undocumented 2x.
- Corollary: the §3.5 deliverables are **not overhead to do after the kernel
  work. They are most of the score.** Start them in parallel, now.
- §3.3 puts "production-ready deployment" out of scope. Do not gold-plate.

## READ FIRST — three PROGRAM.md claims are false

Verified against the organizer script on 2026-08-30. Fix these beliefs before
writing code against them.

1. **`valid_token_mask` is NEVER `None`.** At `padding_ratio<=0` (the default,
   `torch_transformer_benchmark.py:614`) `generate_random_case` returns an
   **all-True mask**, not `None` (`torch_transformer_benchmark.py:255-259`).
   Every caller passes it: `:391 :392 :472 :494 :504`,
   `bench_harness.py:197 :203`.
   → `candidates/best.py:71` (`if valid_token_mask is None`) is **dead code**;
   `is_causal=True` at `best.py:74` never executes.
   → `PROGRAM.md:35` and `PROGRAM.md:103` ("this is what makes shape #14
   feasible") are **false as implemented**. See B1.

2. **`~40 GB/head` is wrong** (`PROGRAM.md:103`, `bench_harness.py:25`). 37 GB is
   one `(batch,head)` slice of the score matrix. There are 32x16 = 512 of them →
   **18.6 TB** fp32 total.

3. **TF32 is already enabled for both sides** — `allow_tf32` default True,
   `matmul_precision="high"` (`torch_transformer_benchmark.py:638-645, :684-688`;
   `bench_harness.py:288-291, :346-347`). The baseline already runs fp32 matmuls
   on tensor cores. "Switch to tensor cores" is not an available win, and this is
   why the 0.002/0.02 gate is generous.

---

## MEASURED STATE — 2026-08-31, A100-80 (xgph1), fp32, official protocol

`candidates/v_router.py`, **2.67x median / 2.98x geomean, 12/12 correct**
(journal iter 14, job `s1_tf32`). TF32 is on at the organizer default for all
non-max-autotune routes; the max-autotune route alone stays at full fp32 to
avoid its measured asymmetric-kernel correctness drift.

| # | speedup | opt ms | routed to |
|--|--|--|--|
| 2 | 4.91x | 0.375 | compile |
| 13 | 4.48x | 9.633 | fused |
| 3 | 5.79x | 0.330 | reduce |
| 7 | 3.52x | 0.525 | compile |
| 11 | 2.91x | 1.198 | fused |
| 12 | 2.35x | 0.803 | fused |
| 4 | 4.97x | 0.379 | reduce |
| 1 | 2.02x | 1.302 | compile |
| 5 | 2.43x | 1.120 | reduce |
| 10 | 2.37x | 0.805 | fused |
| 9 | 2.13x | 0.807 | fused |
| 8 | 1.29x | 6.125 | fused |

---

## Done — latest experiment

- **S4 — Run shape #6. Highest expected value per minute in the queue.**
  **DONE by `codex-1`** (iter 15, Slurm array 777056, A100-40). All three
  candidates passed **491,520,000/491,520,000 elements** with max_abs
  1.91e-6. Router/max-autotune won at **2.79x, 125.66 ms**, ahead of
  reduce-overhead (**2.38x, 146.62 ms**) and fused-QKV (**1.92x, 181.15 ms**).
  Shape #6 is now included in `official-safe`; the router has an explicit
  `compile` route. Final aggregate reporting still requires an A100-80 sweep.

## Done — shape 14

- **S5 — CONFIRMED ON REAL A100-80: shape #14 now runs.** (journal iter 22,
  job `777316`). `candidates/v_chunked14.py`: B=32 is 32 **independent**
  sequences; nothing couples them, so processing in groups of `chunk_size=4`
  and concatenating is exact, not approximate (no precision change). This
  **overturns the earlier "confirmed infeasible" conclusion** (iter 7) —
  shape #14 previously OOMed categorically at 73.85 GB/79.25 GB; chunked, it
  completes a full forward pass in **~74.6s** (reduced timing protocol —
  warmup=2/repeats=3/rounds=1, since the full official 320-call protocol
  didn't finish in a 30-min Slurm limit on an earlier attempt with zero
  progress logged).
  *Correctness:* #14 has no reference (baseline needs 18.6 TB per B5), so
  direct comparison is impossible — but the chunking **mechanism** is
  GPU-confirmed exact on shapes #8/#13 (job `s5_validate_a80`: max_abs
  0.0009-0.0011, well within the 0.002 gate), which do have references.
  Report #14 as "runs, ~74.6s/forward, mechanism proven exact on 8/13,
  unverifiable directly against a reference that cannot exist."
  *Next lever if speed matters more:* CUDA-stream pipelining across the 8
  sequential chunks (overlap chunk N's compute with chunk N+1's transfer) —
  not yet pursued, no evidence yet on how much it would help.
  Full analysis incl. why sparse/linear attention is disqualified:
  `docs/research-shape14.md`.

## Open — post-S1 follow-ups

Read `docs/research-sub2x.md` for the full derivation. Summary of the causes,
because three of the four are **not** the same problem:

- **#8 remains the lowest-ratio shape (1.29x), but S1 cut optimized time from
  26.28 ms to 6.13 ms by restoring TF32 on its eager fused-QKV route.** This
  confirms precision was the lever; any further work should compare against
  the TF32 ceiling, not the old fp32 ceiling.
- **#9/#10 are no longer sub-2x** (2.13x/2.37x, ~0.81 ms optimized each), but
  S2 can still cheaply test whether a compiled route is faster.
- **#5 is now 2.43x at 1.12 ms** after TF32 restoration. Do not optimize it
  independently unless a shared variant improves the aggregate.

- **S2 — CLOSED: `fused` confirmed optimal for #9/#10, with a real
  methodology lesson attached.** An isolated pairwise test (just shapes
  9/10, journal iter 20) showed `reduce` clearly beating `fused` (3.65x/
  3.98x vs 2.14x/2.37x) — but tried inside the full 13-shape sweep
  (`v_router2.py`, journal iter 21), it *regressed*: 1.81x/2.01x, worse
  than `fused`'s in-sweep 2.09x/2.33x. Likely `torch.compile`/CUDA-graph
  memory-pool interaction between the 5 shapes that now compile a `reduce`
  instance in the same process — invisible to a 2-shape isolated test.
  Reverted; `fused` stays on #9/#10. **Route decisions must be validated in
  the full deployment sweep, not pairwise** — recorded as a real lesson for
  any future routing change.

- **S3 — #5 and the large-batch regime: document, do not optimize.** Evidence
  above. Falsify: if a variant beats 2.43x on #5 without regressing #1, this
  reasoning was wrong.

## Open — zero GPU cost, do these first

- **D1 — MOSTLY DONE.** `scripts/check_placeholders.sh` is down from 49 to
  **5 remaining `<FILL>`s, all in README.md's team-member-contributions
  table** (`README.md:283-287`) — real names/roles only the team can supply,
  not fillable from any measurement or log. Everything else (environment
  table, results tables, progress chart, per-shape numbers in both
  README.md and TECH_REPORT.md) is filled with real, current data as of
  iter 30 (`v_router2`, 2.99x median / 3.72x geomean). **Still outstanding,
  not started:** the demo video (public YouTube, linked in Devpost) and the
  Devpost description's five named fields — neither is a file in this repo
  to check off; both need a human to actually record/write/submit.

- **B10 — CONFIRMED, isolated from AMP for correct attribution** (job
  `b10_isolate_a40`, journal iter 16). `candidates/codex-b10-step1.py`'s
  `_effective_mask()` caches the `.all()` classification keyed on
  `(id(tensor), data_ptr, _version, shape, stride, storage_offset, device,
  dtype)`, falling back to the uncached path for tensors with no version
  counter (`inference_mode`-allocated, can't safely cache). First call still
  syncs; every later call with the same mask tensor reads only host-visible
  metadata. **Real gain by itself: geomean 2.92x -> 3.30x** (A100-40, mask-cache
  only, no AMP). Confirms the original hypothesis exactly: biggest on the
  tiny shapes where the sync dominated real compute (#2 5.46x->8.07x, #3
  6.29x->10.68x, #4 5.84x->7.93x). Folded into `codex-combined-step4.py`.

- **AI-attribution gap — worth real points, costs nothing.** Commit `1f99f8d`
  carries `Co-Authored-By: Claude Opus 4.8` and a session link. **None of the
  thirteen commits since do**, including every A100 result. §3.5 awards bonus
  points for the AI skills/tools used and git history is the most credible
  evidence, because it is timestamped rather than written for a judge. Add the
  trailer going forward and state the tooling explicitly in `TECH_REPORT.md` §6.

## Open — the measurement that unblocks the rest

- **M1 — CLOSED: falsified, new leaderboard best confirmed on A100-80.**
  `candidates/v_router2.py` (AMP autocast(fp16) on shapes #6/#8/#13 + B10
  mask-cache; S2's route change tried and reverted, see S2 below) scored
  **median 2.89x / geomean 3.58x on A100-80** (job `step4_confirm_a80`,
  journal iter 21), all 13/13 correct, worst max_abs 0.00168 (atol 0.002) —
  up from the router's 2.67x/2.98x. This is now the leaderboard number.

## Open — shape 14

- **B5 — CLOSED with real evidence, no further work.** OOM during optimized-model
  warmup at **73.85 GB of 79.25 GB** (journal iter 7), confirming plain
  activation memory (32x100000x1024, ~7 live tensors), not an attention-algorithm
  limit — both candidates already use mem-efficient SDPA. `v_compile` never even
  reached the OOM point: killed at the 15-minute limit still inside
  max-autotune's kernel search. Goes in the README limitations section as the
  §3.5 reflection. Do not pursue fp16/H100/chunking (rubric §3.3 puts
  production-grade work out of scope).

## Open — optimization, remaining

- **T6 — CONFIRMED ON A100-80** (job `step4_confirm_a80`, journal iter 21).
  `torch.autocast(fp16)` (keeping norms/reductions fp32) applied to shapes
  #6/#8/#13 in `v_router2.py`. Real gain per-shape vs the fp32 router:
  #13 4.48x -> 10.50x, #8 1.28x -> 1.68x, #6 2.81x -> 2.89x, all still
  correct (worst max_abs 0.00168 < 0.002 atol). Confirms the original
  hypothesis: `tools/probe_sdpa_backends.py --dtype float16` showed flash SDPA
  eligible on all 14 shapes vs 0/14 at fp32 — fp16 buys tensor cores *and* the
  flash backend at once. **bf16 autocast tried as an alternative and rejected**
  (journal iter 18): fails correctness on 13/13 shapes, max_abs 0.0095-0.016
  vs atol 0.002 — bf16's 7-bit mantissa is too imprecise despite the safer
  exponent range; fp16 is the only viable reduced-precision path here. A
  full 13-shape AMP sweep (`amp_full_sweep_a40`) is running to check whether
  more shapes than 6/8/13 should route to it — codex only tried the three
  biggest/slowest. **Result (journal iter 19): confirmed optimal as-is.**
  AMP-everywhere is 13/13 correct (max_abs 0.0012-0.0019, safely under gate
  on every shape) but only *wins* per-shape on #6 (4.78x vs router 2.61x),
  #8 (6.10x vs 1.29x), #13 (15.13x vs 4.72x) — exactly the three shapes
  already routed to it. #5 is a statistical tie (2.54x vs 2.51x, not worth
  the risk); every other shape clearly favors its existing route. No further
  AMP routing needed.
- **T4 — folded into B10's `codex-b10-step1.py` mask-cache fix** rather than
  landing standalone; see B10 above.
- **T7 — DONE, integrated and confirmed on the full leaderboard.** (journal
  iter 29-30, jobs `777483`/`router2_triton_confirm2`). Surveyed three
  Triton targets (AddNorm vs fused-FFN-epilogue vs fused-attention+bias);
  AddNorm (residual-add fused into the immediately-following LayerNorm)
  confirmed as the right risk-appropriate first target. Standalone A100-80
  result vs plain LayerNorm: 13/13 correct, geomean 1.88x → 2.02x (+7.4%).
  Integrated into `v_router2.py`'s `best`/`amp` eager routes (kept separate
  from `compile`/`reduce`, which already get their own fusion from
  Inductor). Full-sweep confirmation: **median 2.99x / geomean 3.72x,
  13/13 correct** — now the leaderboard number. Honestly caveated in
  `leaderboard.md`: the untouched `compile`/`reduce`/`fused` routes showed
  up to ±30% run-to-run swing in the same measurement, so most of the delta
  from 2.89x/3.58x is likely cluster/protocol variance, not purely the
  kernel — read the per-shape numbers, not just the aggregate, before
  citing this as "T7 added 0.14x."
  **Caveat carried forward — the kernel itself is UNTUNED, not optimized.**
  `candidates/v_triton_addnorm.py`'s `_fused_add_layernorm_kernel` has no
  `@triton.autotune`, no `num_warps`/`num_stages` configuration, no
  block-size/warp-count sweep — it runs on Triton's bare defaults.
  `BLOCK_SIZE = triton.next_power_of_2(n_cols)` is a correctness
  requirement, not a performance choice. This is a correct, working first
  cut, not a tuned kernel — see T7b below.

- **S8 — OPEN, HIGHER PRIORITY THAN T7b/T9 BELOW: measure the run-to-run
  error bar before claiming any more gains.** (per `docs/research-batch3.md`,
  S7). A sharp finding: the claimed T7 gain (2.89x→2.99x, +3.9% geomean) is
  likely **smaller than the cluster's own run-to-run noise**. One untouched
  route (zero code changes) swung -30% between two confirmed runs of the
  same candidate (shape 7: 5.85x→4.08x) — on a 13-shape geomean, one such
  swing alone is worth ±2.7%. The honestly Triton-attributable gain (AMP
  routes only) is ~+1.6% — smaller than that noise band. **Action:** re-run
  the current champion (`v_router2.py`, unchanged) twice more on
  `official-safe`. Two extra sweeps convert "we suspect ±30% variance" into
  a measured error bar — worth more to Technical Execution than another
  0.1x, and settles whether T7's contribution is real or noise.
  *For the report:* quote 2.99x/3.72x (legitimate official-protocol
  number), but do not narrate "Triton bought us +3.9%" without this
  error bar to back it up.

- **S9 — OPEN, cheap, same priority tier as S8: seed-robustness check on
  shape #8.** (`docs/research-batch3.md`, S7). Three independent error
  sources now stack on shape #8's `amp` route: fp32 SDPA (~1e-6) → Triton
  AddNorm (0.0006-0.0013) → fp16 autocast (0.00176) — **88% of the 0.002
  absolute tolerance**, on a single fixed seed. This is *not* a near-miss
  (the gate is per-element `abs<=0.002` **OR** `rel<=0.02`,
  `torch_transformer_benchmark.py:314-316` — a 0.00176 element passes
  comfortably on the relative arm unless `|ref|` is tiny), but it's a
  trend worth checking before judging does. **Action:** re-run shape #8
  with 2-3 different `--seed` values. If correctness holds across seeds,
  that's a strong, defensible claim for the report; if not, better to
  find out now.

- **T7b — OPEN, LOWER PRIORITY than S8/S9 above (targets a gain smaller
  than the documented noise floor — do S8/S9 first).** Same textbook next step
  Triton's own tutorials use for exactly this kernel shape (a reduction +
  elementwise kernel): wrap `_fused_add_layernorm_kernel` with
  `@triton.autotune(configs=[...], key=["n_cols"])`, sweeping a small grid
  of `num_warps` (e.g. 1/2/4/8) and `num_stages` (e.g. 1-4) — `BLOCK_SIZE`
  is already fixed by correctness so it isn't a free tuning axis here.
  Triton benchmarks each config once per distinct `n_cols` (our `d_model`
  values: 32, 128, 1024) and caches the winner, the same mechanism
  `torch.compile`'s own autotuning uses (see L3's finding on how that
  caching does and doesn't dedupe — check whether the same redundant-
  autotuning-per-occurrence issue applies here too before assuming this is
  free). Low risk: the fusion design is already correctness-validated,
  this only searches over launch configs for the same math.
  *Falsify:* autotuned config matches the current default → no gain, close
  it and say so plainly rather than re-running until something moves.

- **T9 — OPEN, LOWER PRIORITY than S8/S9 above (same reason — see
  `docs/research-batch3.md` S7), scope genuinely uncertain, consider asking
  the organizers.**
  Would calling a standalone SOTA kernel library (flash-attn's own PyPI
  package, xFormers' `memory_efficient_attention`) directly, rather than
  through PyTorch's SDPA dispatcher, be in scope? Checked the actual
  problem statement (`TikTok TechJam 2026 Information Document`): Track 3's
  own §3.1/§3.3 do NOT contain an explicit "any open-source library is
  fine" resource-policy statement (a *different* track's section does have
  one, Track 3's doesn't — don't assume it carries over). §3.1 does list
  PyTorch as one of four equally-sanctioned implementation approaches,
  which is suggestive but not the same as an explicit blanket allowance.
  **Substantive technical point, independent of the scope question:** we
  already confirmed via B2 (`docs/sdpa_backend_probe_cudnn.json`) that
  PyTorch's own SDPA dispatcher already calls into the same *class* of
  kernel these packages provide — `SDPBackend.FLASH_ATTENTION` wraps a
  Dao-AI-Lab-derived flash-attention kernel, `SDPBackend.EFFICIENT_ATTENTION`
  is architecturally what xFormers' memory-efficient attention provides.
  So calling the standalone packages directly likely would NOT unlock a
  fundamentally different kernel — the only way it could help is if our
  bundled PyTorch (2.10.0+cu128) lags behind the latest standalone
  flash-attn/xformers release closely enough to matter, which is unverified
  either way. cuDNN's fused attention (the other "top kernel" candidate) is
  already confirmed dead-end: runtime-disabled in this exact cluster
  environment (B2).
  *Before spending GPU time:* (1) check whether flash-attn/xformers are
  even installable in `~/flood_env` on the cluster, (2) compare their
  bundled kernel version against what ships in torch 2.10.0+cu128's SDPA,
  (3) if genuinely unsure about scope, ask the organizers rather than
  guess — this is the one open question so far without explicit text to
  point to either way.
- **T2, T3 — superseded.** Both landed as `v_compile.py` / `v_fused_qkv.py` and
  are now route targets inside `v_router.py`.

- **M2 — OPEN, do this before adding more kernel candidates blind.** No
  step in this project's actual measurement stack — not `bench_harness.py`,
  not `autoresearch.workflow.js`'s strategist/coder/runner/postmortem loop
  — has ever used `torch.profiler`, Nsight, or any per-op trace. Every
  bottleneck claim so far (including this session's own Roofline estimates
  for shape #8) is derived from `opt_ms` + a GFLOP count, never a measured
  breakdown of where time inside one forward pass actually goes (norm vs
  attention vs FFN vs launch/dispatch overhead vs mask construction).
  `torch.profiler` ships with PyTorch, needs no install, and can export a
  Chrome-trace or a table of per-op CUDA time with one `with
  torch.profiler.profile(...):` block around a few warmed-up forward
  calls. **Action:** profile `v_router2.py` (the eager `best`/`amp` routes,
  where T7's Triton kernel now lives) on 2-3 representative shapes
  (#1 small, #8 GEMM-heavy, #13 long-seq) and report the real top-3
  time-consuming ops per shape. This either confirms the FFN/AddNorm
  intuition behind T10 below, or surfaces something nobody has looked for
  yet — that's the actual point of profiling before choosing the next
  kernel, not after.

- **T10 — OPEN, the deliberately-deferred sibling of T7.** Fused FFN
  epilogue (`Linear → GELU → Linear`). Explicitly considered during T7's
  survey and passed over *for* AddNorm specifically because it's
  higher-risk (a fused GEMM+epilogue Triton kernel has to actually beat
  cuBLAS/Inductor's already-tuned matmul, not just save bandwidth like
  AddNorm does) — not because it lacks headroom. Now that AddNorm has
  landed, validated, and shown the whole "ship a correctness-gated Triton
  kernel here" pattern works, this is the natural second kernel. Shape #8's
  real ceiling is still ambiguous (`docs/research-batch2.md`: could be 57%
  of the TF32 156 TFLOP/s ceiling or 29% of the fp16 312 TFLOP/s ceiling
  depending on which op dominates under `autocast`'s mixed dtype policy) —
  real, unresolved headroom either way, but **M2's profiling result should
  settle whether FFN is actually the bottleneck before starting this**,
  not just Roofline inference.
  *Risk, stated plainly:* this needs real tile-size/pipeline-stage tuning
  to have a chance of beating an already-tuned GEMM — budget for it to take
  longer than T7 did, and for a first attempt to land at parity or worse
  before tuning helps.

- **T11 — OPEN, the actionable follow-through to L3 (which was
  explanatory-only).** L3 confirmed `torch.compile`'s max-autotune does
  NOT deduplicate identical-shaped matmuls across the N unrolled repeated
  layers (24 AUTOTUNE lines collapsed to only 2 distinct shapes, each
  autotuned 12 times, for shape #1's `num_layers=4`) — but never tried the
  fix. Hypothesis: compile only `OptimizedBlock.forward` once (not the
  whole unrolled multi-layer forward), and call the *same* compiled
  callable N times in the layer loop. Since every layer shares identical
  shapes (only the weights differ), this should let Inductor's autotune
  cache genuinely hit rather than re-benchmark per occurrence — cutting
  compile time, and might sidestep S2's diagnosed CUDA-graph-pool
  interaction too (fewer distinct compiled graph instances alive at once
  across a sweep). Directly relevant to whether `v_compile.py`-style
  routes could ever become viable for shape #14 (iter 7's original
  timeout) — not urgent for the current leaderboard, but the cleanest
  remaining lead on *why* compile scales badly with `num_layers`.
  *Falsify:* per-layer compilation doesn't reduce total AUTOTUNE line count
  → the redundancy isn't fixable this way, close it.

- **T12 — OPEN, low priority, marginal expected upside.** Manual
  `torch.cuda.graph()` capture (hand-rolled, not via `torch.compile`'s
  automatic `reduce-overhead` mode). More explicit control over exactly
  what gets captured and replayed, without Inductor's guard/specialization
  machinery in the way — in principle could shave a bit more launch
  overhead off the tiny-batch shapes (#2, #3) than `reduce-overhead`
  already does. Real but likely small: `reduce-overhead` already captures
  the whole compiled region as one CUDA graph, so a hand-rolled version is
  mostly betting that Inductor's own graph boundary is suboptimal, which
  is a narrower claim than it sounds. Lower priority than S8/S9/M2/T10/T11
  above — only worth it if profiling (M2) specifically shows launch
  overhead, not compute, still dominates one of the small shapes after
  everything else lands.

## Literature review cross-check (`~/Downloads/deep-research-report.md`)

A deep-research survey of the transformer-kernel-optimization literature
(FlashAttention family, PagedAttention/FlashInfer, quantization, sparsity,
Triton/CuTe/TileLang, Roofline methodology) was checked against this repo.
Most of it targets **autoregressive decode/serving** (growing KV cache,
many forward calls, request batching) — our task is a **single fixed-shape
prefill-only forward pass** per candidate, so a large fraction doesn't
apply. Below: every distinct idea in the report, and its actual disposition
here, not just "considered."

**Already tried, matches the literature's own conclusion:**
- Flash-style tiled exact attention avoiding the O(S²) score matrix — T0
  seed, confirmed correct/fast from iter 0.
- Fusion (norm/residual/GELU/matmul epilogues) via a graph compiler rather
  than hand-written kernels — `torch.compile` (T1/T2), matches the report's
  own "dispatch among multiple kernels... instead of one universal kernel"
  recommendation for the PyTorch stack specifically.
- Precision-aware Tensor Core usage (TF32) — S1.
- Mixed precision with numerically sensitive ops kept in fp32 — T6/AMP
  (`autocast` keeps LayerNorm/softmax fp32, matches Transformer Engine's
  documented pattern exactly).
- Per-shape/per-regime kernel dispatch instead of one universal kernel — T5
  (`v_router.py`), the report's own stated best practice for PyTorch.
- Informal Roofline/bottleneck classification per shape (TFLOP/s vs peak,
  launch-overhead diagnosis for tiny batches) — `docs/research-sub2x.md`,
  T8.
- **Newly checked and closed this pass:** cuDNN SDPA backend
  (`SDPBackend.CUDNN_ATTENTION`) — the report flagged that PyTorch 2.5+
  ships this as a 4th backend our original B2 probe never tested. Confirmed
  present in torch 2.10.0+cu128; **confirmed unavailable at the environment
  level** ("cuDNN attention has been runtime disabled") on all 14 shapes —
  not a per-shape ineligibility, a build-level gap. `mem_efficient` remains
  the only real fp32 option. Journal iter 23, `docs/sdpa_backend_probe_cudnn.json`.
- **Newly tried and rejected this pass:** approximate (tanh) GELU instead of
  exact (erf) — passes correctness (max_abs 0.0003-0.0005) but is a
  statistical wash on speed (2.083x vs 2.086x median) — GELU is too small a
  fraction of FFN time on our shapes for this to matter. Journal iter 24,
  `candidates/v_gelu_tanh.py`. Not adopted.

**Correctly out of scope, not worth adding:**
- PagedAttention, FlashInfer, KV-cache paging/quantization, split-KV decode
  scheduling — all decode/serving-specific (growing KV cache across many
  forward calls). We run one forward pass per shape; there is no KV cache.
- GQA/MQA — changes the model architecture (fewer KV heads), which breaks
  weight-copy compatibility with the fixed baseline. Disqualified on
  correctness grounds, not effort.
- Weight-only INT4/INT8 (GPTQ/AWQ/Marlin) — these amortize dequantization
  cost over *many* decode steps reusing the same weights; we do a single
  forward pass per shape, so that amortization doesn't apply, and our bf16
  result (7-bit mantissa already fails by 5-8x) is strong evidence a
  coarser format fails worse. Not worth a GPU job to confirm the obvious.
- FP8 — no native FP8 Tensor Cores on A100 (our primary hardware); would
  need H100 (see below).
- Structured/unstructured sparsity (SparseGPT, Sparse-Marlin) — requires
  pruning the baseline's fixed weights, which changes the function being
  computed. Same correctness objection as GQA.
- RoPE/ALiBi fusion, gated-FFN (SwiGLU) horizontal fusion — the baseline
  model has neither (no positional-embedding kernel to fuse; FFN is plain
  `Linear→GELU→Linear`, no separate gate/up projection to fuse together).
- Multi-GPU/distributed kernels (ParallelKittens, collective overlap) — one
  GPU per benchmark job, no cross-device communication exists to overlap.
- CPU-specific backends (oneDNN/AMX) — out of scope, target hardware is
  A100/H100 GPU per PROGRAM.md.

**Genuinely new, not yet tried — added below as open items:**

- **L1 — CLOSED: falsified, documented, not acted on.** (journal iter 26,
  job `l1_h100_confirm`). Ran `v_router2` (A100-80-tuned route table) on
  H100-47, official protocol, 13/13 correct (worst max_abs 0.00182 < 0.002
  atol). **Route rankings do NOT transfer unchanged: H100 gives LOWER
  numbers** with our current routing — median 2.62x / geomean 3.19x, vs
  A100-80's 2.89x/3.58x (e.g. shape 4: 6.50x on A100-80 `reduce` vs 4.57x
  on H100; shape 9: 2.22x `amp` vs 1.68x). Matches the literature review's
  own warning almost exactly. Not re-tuning routes for H100 — A100-80
  remains the canonical leaderboard device (matches primary cluster
  capacity and the existing leaderboard.md/README.md convention) — but
  this is a real, honest cross-architecture finding worth a paragraph in
  the tech report: it shows the router's routing decisions are
  architecture-specific, not universal, which is itself evidence the
  swarm found a genuine hardware-dependent effect rather than overfitting
  one benchmark run.
- **L2 — CLOSED: falsified, pipelining is slower.** (journal iter 27,
  job `777436`, `candidates/v_chunked14_streams.py`). Tested 2 alternating
  CUDA streams across shape #14's 8 sequential chunks. **Result: 79.6s vs
  74.6s sequential — ~6.6% *slower*, not faster.** As the candidate's own
  docstring anticipated: chunks are already GPU-resident (no H2D/D2H
  transfer to hide, unlike the classic "prefetch tile n+1" pattern), and
  each chunk (batch=4, seq=100000) likely already saturates available SM
  occupancy on its own — 2 streams add cross-stream sync/allocator overhead
  without finding real idle capacity to fill. `v_chunked14.py` (sequential)
  remains the better shape-14 implementation; do not adopt the streamed
  variant.
- **L3 — CLOSED: confirmed no dedup, explains iter 7's shape-14 finding,
  no leaderboard action.** (journal iter 25, `tools/l3_autotune_count.py`).
  Instrumented Inductor's autotune log for shape #1 (num_layers=4):
  **24 total AUTOTUNE lines collapse to only 2 distinct GEMM shapes, each
  occurring 12 times** — the identical `addmm(8192×128, 8192×128, 128×128)`
  shape is autotuned separately at every one of its 12 occurrences across
  the unrolled 4-layer graph, not deduplicated. This is the concrete
  mechanism behind iter 7's shape-14 timeout: with 2 layers × several
  large-matmul call sites, each ~100-300s autotune pass multiplies by
  *occurrence count*, not distinct-shape count. Explanatory value for the
  tech report; no leaderboard change, since `v_router2` already avoids
  routing large-batch/large-seq shapes through `compile` for this exact
  class of reason (established before this diagnosis, now backed by a
  mechanism instead of just an observation).

## Done

- **S1 — TF32 scope fixed and measured** (iter 14, job `s1_tf32`). The old
  import-time blanket disable silently forced both baseline and candidate to
  full fp32. The workaround now applies only to the max-autotune route that
  exhibited asymmetric TF32 kernel drift; other routes restore the organizer's
  `allow_tf32=True`, `matmul_precision="high"` defaults. A100-80 official
  sweep: **12/12 correct, 2.67x median / 2.98x geomean**, max_abs 0.001135
  against atol 0.002. Shape #8 optimized time fell 26.28 ms -> 6.13 ms.

- **T1 — `torch.compile(mode="reduce-overhead")`** → `candidates/v_compile_reduce.py`.
  A100-80, official protocol: 12/12 correct, **median 2.29x / geomean 2.39x
  standalone**. Beats every existing candidate on shapes 3/4/5. Folded into
  `v_router.py`'s route table as a 4th target (iter 10-11).
- **B2 — RESOLVED by direct measurement** (journal iter 8, `bc67c11`,
  `docs/sdpa_backend_probe.json`). Forcing each SDPA backend per shape on
  A100-80: **flash eligible on 0/14 shapes at fp32**, mem-efficient on 14/14.
  Every speedup in this repo to date came from mem-efficient SDPA, never flash.
  No number changes; the report must simply name it correctly.
- **B7 — CLOSED, claim twice corrected.** Original claim (sm80 caps flash
  head_dim at 128, so #8 is excluded) was wrong twice over. First, Codex cited
  `sdp_utils.cpp` showing the cap is **256**, not 128. Then the fp16 probe showed
  **flash fires on all 14 shapes including #8 at head_dim 256**. The gating
  variable was never head_dim — it is **dtype**. Recorded because it is a clean
  example of a plausible, well-cited claim surviving one review and still being
  wrong until measured.
- **B8 + B9 — DONE** (`164dd97`). Timing now mirrors
  `torch_transformer_benchmark.py:benchmark_models()` (warmup 20 / repeats 100 /
  rounds 3, alternating order); candidate loaded once per sweep. **This pair
  caught a 24% inflation**: iter 5 reported 2.71x median under the old protocol,
  corrected to 2.18x.
- **T8 — ANSWERED without a run.** Baseline wall-clock is ~1.87 ms for #2 (B=1),
  #3 (B=4), #4 (B=16), #7 and #12 alike. Sixteen times the work, same time — the
  reference is launching, not computing. Speedups track it (#2 5.07x, #3 4.24x).
- **TF32 correctness bug — found by running, not reasoning** (`8010964`).
  Inductor's max-autotune selected TF32 GEMM kernels for fp32 inputs, drifting
  ~0.005 against `atol=0.002` on 9/12 shapes. Invisible on CPU and MPS, which
  have no TF32. See S1 — the *fix* is now itself a research question.
- **T5 — Per-shape dispatch** → `candidates/v_router.py`. Routes each shape to
  whichever of best/v_compile/v_fused_qkv empirically won it (measured, not
  the head_dim/backend-table heuristic originally proposed — the three
  candidates' relative strengths didn't reduce to one clean rule across the
  12-shape sample). Latest A100-80 official-protocol confirmation after S1:
  **2.67x median / 2.98x geomean**, 12/12 correct (iter 14). (agent: opus-1)
- **T0 — SDPA seed** → `candidates/best.py`. Correctness verified on CPU on dev
  shapes + `official-safe`, with and without padding. **Caveat:** the CPU run
  cannot expose B1, B2 or B5, and the claimed shape-#14 capability is false (B1).
  GPU speedup still pending the first cluster run. (agent: seed)

## Resolved by the problem statement (27 Aug 2026 revision)

Read directly from TechJam Track 3 §3.1-3.7, not second-hand.

1. **Tolerance** — `relative error < 0.02, abs error < 0.002` (§3.2). See B4.
2. **Shape table** — §3.7 matches `bench_harness.py:70-85` exactly. All 14 rows,
   causal TRUE throughout. Our catalog is correct, no changes needed.
3. **Evaluation GPU and speed aggregation — THERE IS NO ORGANIZER-RUN BENCHMARK.**
   §3.2: "Optimize & test your codes on your own machine. Different methods may
   be used to optimize the codes depending on the machine (GPU cards) you use."
   §3.4: "You can download 1 of these, and run it on your own machine."
   §3.5: the tech report must state "what the environment is (CPU, GPU, DISK, etc)".
   - We choose the GPU. We run the benchmark. We report the numbers.
   - There is no hidden grader, no fixed target hardware, and no cross-shape
     aggregation formula, because nobody re-runs our code on their machine.
   - The correctness gate is self-administered. Report it honestly and in full,
     including the shapes that fail or cannot run.
4. **Shape #14 is no longer a blocker.** §3.5 requires the README to carry "a
   brief reflection on your solution's limitations and what you would improve
   given more time". A documented, quantified OOM with the memory arithmetic
   (~85 GB fp32 for seven live activations) is a legitimate deliverable, not a
   disqualification. Do NOT burn the remaining hours trying to make it run.

## Still worth getting

- **Webinar recording** — 28 Aug, 3:00-3:45pm, linked from the problem statement.
  45 minutes of organizer Q&A that may carry intent not in the written spec.
- **B0 — confirm our script is the 27 Aug 6:25PM revision.** The statement says
  `torch_transformer_benchmark.py` was updated that day. Our copy was committed
  2026-08-30 16:41 (`e13d295`), after the update, and its argparse defaults
  (`:618-619`) already match §3.2, so it is almost certainly current. Diff it
  against the download link once. 30 seconds, removes a silent total-loss risk.

## Seeded by human

_Humans: drop ideas here as free text — an agent will formalize each into a
proper hypothesis + candidate and treat it as high priority._
