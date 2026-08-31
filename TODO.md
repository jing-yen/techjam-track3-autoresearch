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

## M2 — DONE for all 3 originally-scoped shapes (#1, #8, #13); real,
actionable finding surfaced: T7's AddNorm only covers ONE of two boundaries

(job `778628`, `tools/profile_shapes.py --shapes 1,13`, joining #8's
already-done trace). **Shape #1** (`compile` route, torch.compile+cudagraph):
dominated by GEMMs (`triton_tem_fused_addmm_t_view` 46.5% + `ampere_sgemm`
25.2% cuBLAS = ~72% combined, real compute, expected), flash attention
18.8%. Confirms Inductor's own auto-fusion ALREADY fuses LayerNorm+residual
for compiled routes (`triton_per_fused_add_addmm_native_layer_norm_view_*`,
several kernels, ~5.6% combined) — real evidence (not just inference) that
`v_router2.py`'s existing design comment ("T7 used only by eager routes,
compile/reduce get their own fusion from Inductor") was correct. GELU
shows up as its own small separate pointwise kernel
(`triton_poi_fused_addmm_gelu_view_3`, 1.78%) — same ballpark as #8's 1.92%,
confirms K2's conclusion generalizes: not a big lever anywhere we've measured.
**Shape #13** (`amp` route, S=1024, eager): flash attention 24.65%, fp16
GEMM 19.54%, fp32↔fp16 casting (`aten::to`/`_to_copy`/`copy_`) **17.66%**
— matches #8's 17.1% almost exactly, confirming the casting tax is a
consistent, shape-independent cost of the `autocast` approach, not a
shape-8 fluke. **The new finding:** `aten::native_layer_norm` (UNFUSED)
is **19.21%** of total CUDA time — bigger than T7's own already-fused
AddNorm kernel showing up right next to it at 9.57%. Root cause, traced to
source: `_BestBlockTriton.forward` (`v_router2.py:257-264`) fuses only
`x + attention(norm1(x)) -> norm2`; the OTHER boundary,
`x + ffn_out(...) -> next layer's norm1` (or `final_norm` for the last
layer), stays two separate ops. This is real, unaddressed, and — unlike
K2'/T10's marginal ~2% GELU opportunities — a full **19%** of one shape's
CUDA time, using a kernel T7 already proved correct rather than a new one.

- **T15 — DONE, LANDED, NEW LEADERBOARD BEST.** `candidates/v_triton_addnorm2.py`
  (standalone validation) + integrated into `candidates/v_router2.py` as
  `_BestBlockTriton2`/`_BestTransformer2`, wired to `_AMPTransformer` only
  (shapes #6/#8/#13 — the only route this touches). Extends T7's fused
  AddNorm kernel (unmodified, byte-identical) to the second residual+norm
  boundary: `ffn_out`-add fused into the *next* layer's `norm1`, or into
  `final_norm` for the last layer (T7 only covered attn-out-add into
  `norm2`). Padding handled by deliberately NOT threading a mask through
  the fused kernel — mirrors T7's own already-validated approach — argued
  in the file's docstring AND tested empirically at `--padding-ratio 0.3`
  (not just assumed, since this is a new fusion, not a reuse of T7's exact
  validated pattern). **Confirmed on real GPU, full 13-shape sweep (job
  `778709`, journal iter 42): 13/13 correct, worst max_abs 0.00176
  (unchanged from before). Geomean 3.69x → 3.81x (+3.3%).** All three
  amp-routed shapes improved individually, well above the ±2.7% noise
  floor (S8): #6 +17.2%, #8 +4.2%, #13 +12.5% — matches M2's prediction
  exactly (shape 13 had the biggest measured unfused-boundary cost).
  Median barely moved (2.99x→2.98x, inside noise — dominated by the 10
  untouched compile/reduce/fused shapes). **`leaderboard.md` updated**
  per the guarded-update procedure (pulled, re-checked, still improving).

## K4 — CLOSED: real mechanism found, but it lands in warmup, not the timed
loop, so it does NOT explain S2 (which remains genuinely open)

(job `778517`, `tools/check_k4_cudagraphs.py`, `torch._logging.set_logs
(cudagraphs=True, recompiles=True)` — PyTorch's own internal diagnostic, not
inferred). Ran the compile/reduce-routed shapes (1,2,3,4,5,7) both isolated
(own process each) and back-to-back in one process (mirrors a real sweep).

**Confirmed real, with a precise mechanism (not the one K4 originally
guessed):** isolated runs show **zero** recompiles or allocator checkpoints;
the one-process run shows **4 Dynamo recompiles and 5 CUDA-caching-allocator
checkpoint/restores** for just 6 shapes. Root cause, traced to source:
`_CompileTransformerBase._forward_impl` (`v_router2.py:337`) is one shared
class method, `torch.compile`d fresh per model instance — Dynamo's guard
cache is keyed on that *shared* code object, not per-instance. `v_router2`
also deliberately toggles `torch.backends.cuda.matmul.allow_tf32` per
route (`v_router2.py:520-535`, the S1 fix: `False`/"highest" for `compile`,
`True`/"high" for everything else, both ways every time — intentional,
already documented, not a bug). Since that's a Dynamo-tracked global-state
guard, every transition into or out of a `compile`-routed shape invalidates
prior cache entries and forces a fresh compile — confirmed directly in the
log: shape 3's first call was recompiled specifically because "GLOBAL_STATE
changed: allow_tf32" against both of shapes 1 and 2's cached entries.

**Why this does NOT explain S2:** `bench_harness.py`/`torch_transformer_
benchmark.py` (`warmup_model`, called once per shape *before* that shape's
own `benchmark_once`) runs all compilation and any guard-miss recompilation
inside the untimed warmup loop, synchronized before the timed CUDA-event
loop starts. Structurally, that means this cost inflates a full sweep's
*total wall-clock time* (more to compile, more allocator checkpoint/restore
churn) but cannot leak into the *per-shape CUDA-event speedup numbers* that
actually populate the leaderboard — those are measured strictly after
warmup completes, on a model whose guards already match. **K1's original
premise (non-static parameter addresses causing a hidden per-call D2D copy
tax) was not observed at all** — no skip/alias/copy warnings anywhere in
the log, cudagraph capture succeeded cleanly every time.
**S2 remains unexplained.** The one-process run does show accumulating
allocator-checkpoint state (5 checkpoints across 6 shapes) that *could*
plausibly slow down cudagraph pool allocation/replay for shapes appearing
*later* in a 13-shape sweep (S2's #9/#10 come after 8 other shapes'
compile/reduce/amp instances have already accumulated pool state) — but
this run didn't measure actual per-call latency, only log events, so that's
a hypothesis for whoever picks it up next, not a finding. Real next step if
anyone wants it: rerun S2's #9/#10 pairwise test with #1-#8 run first in the
same process (reproducing full pool-accumulation) vs a fresh process, and
diff the CUDA-event timings directly — that would either confirm or rule
out pool accumulation as S2's actual cause.

## Cluster notice — node `xgpj0` (a100-80) has a broken CUDA/torch env, exclude it

Discovered while dispatching K4 (2026-08-31, ~this entry's timestamp). Any job
landing on `xgpj0` fails at `import torch` with `OSError: .../libtorch_global_deps.so:
cannot open shared object file: No such file or directory` — even though the file
exists, has correct permissions, and reads fine via plain `open()`/`ldd` (`ldd`
reports zero missing deps). This is NOT a code or env-setup bug: it's node-local
(confirmed by rerunning the identical command on `xgph1`, which works cleanly,
`torch 2.10.0+cu128, cuda available=True`). Reproduced 3x on `xgpj0` (jobs
778481/778487/778494, both a fresh `sbatch` and an interactive `srun`), including
with `tools/profile_shape8.py`, which succeeded on a different node earlier the
same day — ruling out "the env was always broken." Likely a stale/corrupted
StorPool NFS mount cache local to that one node.
**Action for everyone: add `#SBATCH --exclude=xgpj0` (or `--nodelist=<known-good>`)
to any job until this clears.** Not filed with cluster admins from this session
(no ticket system access) — if anyone has an ops contact, worth a heads-up so it
doesn't silently eat other people's GPU allocation the way it ate three of mine.

## Open — kernel frontier (research pass, 2026-08-31)

- **K1a — REINSTATED with a bounded 4h time-box, narrower scope than the
  closed K1 below.** `docs/k1-spec.md` (sxkhoo, user-approved despite the
  negative literature). NOT the whole-model single-CTA design that was
  falsified — targets shape #2's specific 97%-non-arithmetic overhead
  (374us / ~40 kernels = 9.3us/kernel, on a 128x128x128 GEMM that's only
  0.22us of real math) with a staged fused-FFN-block kernel
  (`norm2->ffn_in->GELU->ffn_out->residual`, no softmax/causal-mask/
  per-head-loop — this is T10 scoped down to where cuBLAS is weakest, not
  T10 itself). Safe by construction: `v_router2` only routes to a winner,
  so a losing K1a can't regress the leaderboard. Kill criteria pre-agreed
  (<20% win over routed `reduce` on #2, correctness bug unfixed in 30min,
  or 4h total) — honour them, don't extend post-hoc. Three-agent lit pass
  (`docs/research-agent-findings.md`) independently confirms my K1
  falsification transfers to A100 generally (AutoMegaKernel 0.55-0.79x,
  Hopper-gated), so K1a's narrower small-shape bet is the only live
  version of this idea. **CLAIMED (opus-1) — `candidates/v_triton_k1a_ffn.py`
  written, CPU-fallback-tested (13/13 structural), GPU correctness+speed
  test dispatched on shape #2.** Autotuned over BLOCK_M in {8,16,32,64,128}
  (M=128 total rows for shape #2, B=1×S=128) — one Triton kernel per layer
  fusing all 5 FFN-block ops (LayerNorm reduction, both GEMMs via
  `input_precision="ieee"` per T10's fix, erf GELU, residual add). Timer
  starts now against the pre-agreed kill criteria.

- **K1 (original whole-model design) — CLOSED: real working kernel built,
  correctness proven, falsified on speed. Do not resume without a
  fundamentally different design.**
  (journal iter 35, `candidates/v_triton_megakernel_s2.py`, jobs
  `778174,778187,778204,778220,778250` — 5 real GPU attempts).
  Attempted a single-CTA (`grid=(1,)`) kernel fusing the entire 4-layer
  forward pass for shape 2 into one launch (was scoped as "one Triton
  kernel per block, 4 launches"; went further, to a true single launch for
  the whole model, since batch=1 makes it tractable). Hit and fixed 4 real
  bugs in sequence, each a genuine Triton constraint, not a logic error:
  (1) `constexpr` values are plain Python ints at trace time, no `.to()`;
  (2) tile shapes must be a power of 2 (`3*D=384` isn't — split into 3
  D=128-wide GEMMs); (3) Triton doesn't support Python slice syntax on
  already-materialized tensor values, only on `tl.load` pointer
  expressions — restructured per-head Q/K/V to compute directly via
  pointer-offset GEMMs rather than slicing a full Q/K/V tensor; (4)
  `OutOfResources`: needed 278528 bytes shared memory vs A100's
  166912-byte limit (~67% over) — fixed with explicit `num_stages=1`.
  **Result after all 4 fixes: CORRECT (max_abs 0.00087, passes gate) but
  0.088x speedup — ~11x SLOWER than the baseline it was meant to beat.**
  Root cause, distinct from the compile bugs above: `grid=(1,)` means this
  kernel runs on exactly **1 of A100's 108 SMs** for the whole forward
  pass — it correctly cuts launch count from ~40-60 to 1, but at the cost
  of using <1% of the GPU's actual parallel hardware, a fundamentally
  different bottleneck than kernel-launch overhead. **Falsify condition
  triggered unambiguously** (team's own bar: stop if it doesn't beat
  `reduce`'s 0.374ms baseline by >20% — this is 11x slower, not within
  20% of a win). Fixing this needs a genuine multi-CTA redesign (one CTA
  per head, or per SM-sized work-chunk with cross-CTA reduction), not
  further tuning of the single-CTA design — real, larger scope than the
  original time-box, not attempted given remaining time. The kernel
  itself is real, working, and correctness-proven — kept in the repo as
  documented negative-result research for the tech report, not deleted.
  Literature: Hazy Research Llama-1B megakernel, AutoMegaKernel
  (arXiv:2606.09682) — both of which use sophisticated multi-CTA/warp-
  specialized persistent-kernel designs, not the naive single-CTA
  approach tried here; that gap is exactly why they're fast and this
  first attempt wasn't. Full brief: `docs/research-kernel-frontier.md`.
  *Checked and deliberately NOT attempted:* Mirage / Mirage Persistent Kernel
  (Wu et al., OSDI'25, github.com/mirage-project/mirage) — an automated kernel
  superoptimizer, including a full "compile a model to one megakernel" mode.
  Sounds like a shortcut for exactly K1; it isn't one for us. Real checked
  blockers: requires rewriting the model in Mirage's own graph-building API
  (no `nn.Module` import path), no prebuilt wheels (from-source build only,
  real risk on our specific cluster CUDA/toolchain), and its only worked
  examples are full-LLM scale (Qwen3-8B), nothing at our small/varied-shape
  scale. Net effort (new API + from-source build + no precedent for our
  workload) exceeds hand-writing K1 ourselves. Cite as future work in the
  tech report; do not attempt integration in the time remaining.
- **K2 — cuBLASLt fused GELU_BIAS epilogue for the FFN.** **Gate checked:
  cuBLASLt's GELU_BIAS is confirmed tanh-approximate, not erf** — but this
  does NOT disqualify it the way the original note assumed. We already
  measured (iter 24, T7 tanh-GELU test) that plain tanh-GELU passes
  correctness on all 13 shapes (max_abs 0.0003-0.0005, well under gate) — the
  approximation itself is fine for this model. **New risk found instead:**
  `torch-cublas-hgemm` (the wrapper the original note cited) does fp16
  *accumulation*, not just fp16 compute with fp32 accumulation — a stronger
  precision cut than anything shipped so far (T7/T6 both specifically keep
  reductions in fp32). Also a small, unclear-maintenance package (79 stars).
  **Superseded by T10** (`candidates/v_triton_fused_ffn.py`) — real GPU
  result now in and T10's own thread is CLOSED for the 3 shapes it would
  have competed on (see T10 above: correct, but a hand-rolled GEMM can't
  beat cuBLASLt there either, in any precision).
  **K2 (cuBLASLt library route) — CLOSED, do not attempt.** Independent
  3-agent lit pass (`docs/research-agent-findings.md`) separately confirms
  cuBLASLt's `GELU_BIAS` is almost certainly tanh not erf (CUTLASS names
  them as distinct epilogues `GELU`/`GELU_taylor`; cuBLASLt exposes only
  the unqualified one; corroborated by an unchallenged report on cutlass
  discussion #700) — would risk the correctness gate for no proven upside.
  **K2' — half-answered for free from K4's own log (job 778517), no new
  GPU time needed for this part.** Inductor's own max-autotune GEMM
  competition (visible in the AUTOTUNE lines K4's diagnostic already
  captured) already tries Triton candidates for our exact GEMM shapes and
  picks ATen's `bias_addmm` as the winner: `addmm(128x128,128x128,128x128)`
  — ATen 0.0123ms vs the best Triton candidate 0.0143ms (~17% slower, not a
  landslide, but ATen wins); `addmm(8192x128,8192x128,128x128)` — ATen
  0.0358ms vs Triton 0.0481ms. So the "is Triton being wrongly skipped"
  half of K2' is answered: **no, it's correctly NOT chosen, on real
  autotune data** — nothing being left on the table by a backend
  misconfiguration. **Still open, genuinely unanswered:** whether fusing
  GELU into Triton's epilogue (saving ATen's separate pointwise-GELU
  kernel launch after `bias_addmm`) could still tip the *total* (GEMM+GELU)
  time below ATen's 2-kernel sequence even though the raw GEMM alone
  loses — this is exactly what T10 tested, but against plain `best.py`,
  never against `compile`/`reduce`'s own already-autotuned baseline.
  **CLOSED (job 778646 + M2's shape #1 trace, job 778628).** Both
  questions now answered with real data: (1) `check_k2prime_backend.py`'s
  own AUTOTUNE dump reconfirms ATen's `bias_addmm` wins the raw GEMM again
  (0.0358ms vs Triton's 0.0481ms) — same conclusion, independently
  reproduced. (2) M2's shape #1 profiler trace (a `compile`-routed shape,
  so Inductor's own auto-fusion is active) shows GELU DOES already get its
  own dedicated kernel (`triton_poi_fused_addmm_gelu_view_3`) separate from
  the main GEMM kernel — but at only **1.78%** of total CUDA time, matching
  #8's 1.92% almost exactly. So the fusion opportunity T10 chased is real
  but consistently small (~2%) everywhere it's been measured, on both the
  `amp` and `compile` routes — not worth further GPU time chasing directly;
  M2's OTHER finding from this same batch (the unfused second AddNorm
  boundary, ~19% on shape #13) is the actionable one. See T15 above.
- **K3 — split-K attention for #9 — BOUNDED, DO NOT BUILD.** Attention is 14%
  of #9; max whole-layer gain ~1.09x, under the noise floor. Logged as
  considered.
- **PagedAttention — REJECTED on mechanism.** Solves KV-cache fragmentation in
  multi-request decode serving; we have no KV cache, no decode loop, no
  concurrency. Would add overhead, remove nothing.

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

- **D2 — THE VIDEO. Now the highest-risk item in the project, and S8 is its
  footage.** A hard §3.5 requirement (public YouTube, linked in Devpost) with
  zero work done and ~26 h left. A 3.72x with no video is an incomplete
  submission; nothing else in this queue has that property. **Plan:** record
  S8's confirmation sweep as the video centerpiece — the same GPU allocation
  produces the error bar AND the footage, and the video then shows the actual
  measurement behind the reported number instead of a staged demo. Script ready:
  `docs/demo-video-script.md`. Devpost five-field draft ready:
  `docs/devpost-draft.md`. **Freeze `v_router2` at T-12h; budget the last 4h for
  record/upload/submit.** Full endgame table: `docs/research-queue-audit.md`.

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

- **T16 — NEW, built, dispatched.** `candidates/v_manual_fp16.py`: attacks
  the OTHER large item M2 found consistently on both profiled amp shapes
  — fp32↔fp16 casting (`aten::to`/`_to_copy`/`copy_`) at **17.1-17.7%** of
  CUDA time on shapes #8 and #13, unaddressed by T7/T15 (which targeted
  the LayerNorm/residual side, not this). Root cause: `_AMPTransformer`
  enters a fresh `torch.autocast` block on every single `forward()` call,
  so every weight tensor gets re-cast fp32→fp16 from scratch every timed
  iteration — autocast's own weight-cast cache only helps *within* one
  call (each weight is used once per call, so it never helps here).
  Fix: cast every Linear's weight/bias to fp16 **once**, in place, on the
  first CUDA forward call, then never again — everything else (activation
  casts at each Linear/attention boundary, fp32 LayerNorm reduction via
  T7's kernel, SDPA's own internal fp32 accumulation) stays numerically
  identical to what the current amp route already does, so this should be
  bit-identical in output, just skipping redundant, deterministic,
  throwaway weight-cast work. Builds on T7+T15's proven AddNorm structure
  (both boundaries). CPU-fallback-tested (13/13-equivalent dev shapes).
  GPU test dispatched — standalone first; only worth integrating into
  `v_router2.py` if the amp-routed shapes (#6/#8/#13) show a real,
  above-noise-floor (>2.7%, per S8) improvement.

- **T17 — NEW, built, dispatched.** `candidates/v_triton_addnorm_fused.py`:
  applies T7+T15's proven AddNorm fusion (both boundaries) to the `fused`
  route (shapes #9/#10/#11/#12), which currently has **none** — checked
  directly in `v_router2.py`: `_FusedBlock.forward` is plain eager
  PyTorch, two fully unfused residual-add+LayerNorm pairs, no Triton
  kernel at all. Unlike `compile`/`reduce` (Inductor auto-fuses this,
  confirmed via M2) and unlike `best`/`amp` (T7+T15 already there),
  `fused`'s only optimization has been the fused-QKV projection since
  T5/T3 — nothing on the norm/residual side since. Reuses T7+T15's exact
  kernel and cross-layer-chaining wiring unmodified, swapping in
  `_FusedAttention`'s fused-QKV projection for T15's separate Q/K/V
  Linears — the only structural difference. CPU-fallback tested (13/13-
  equivalent dev shapes). GPU test dispatched — standalone first, same
  workflow as T15/T16.

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

- **M2 — DONE for shape #8; the first real profiler trace in this repo.**
  (journal iter 36, `tools/profile_shape8.py`, job `778327`). Every
  bottleneck claim before this (including this session's own Roofline
  estimates) was inferred from `opt_ms` + a GFLOP count — never a measured
  per-op/per-kernel CUDA trace. Real breakdown for shape #8's `amp` route:
  **fp16 tensor-core GEMM 55.6%** (real, expected compute), **dtype
  casting (fp32↔fp16 for autocast) 17.1%** (a genuinely new finding — real
  overhead nobody had measured), Triton AddNorm 6.75%, unfused post-FFN
  residual add 6.05%, flash attention 6.02%, unfused norm1 5.81%, GELU
  1.92%. **This directly explains T10's shape-8 regression as more than
  "needs tuning":** T10 forces full fp32 accumulation for correctness
  safety; shape 8's real cost is 55.6% fp16-tensor-core + 17.1% casting —
  T10 in fp32 gets neither the tensor-core speed nor any casting
  reduction, so it's solving the wrong problem on this specific shape.
  T10 (fp32) is a better structural fit for shapes that don't route to
  `amp` (baseline already fp32 there, T10's bandwidth/launch savings are
  real). **Dispatched:** `tools/profile_shapes.py --shapes 1,13` (generalized
  from `profile_shape8.py`) to finish the originally-scoped 2-3-shape
  picture, running alongside K1a/K2'.

- **T10 — LANDED for non-amp shapes, CLOSED (do-not-pursue further) for
  the 3 amp-routed shapes (#6/#8/#13).** `candidates/v_triton_fused_ffn.py`:
  Triton GEMM+bias+erf-GELU epilogue fusing `ffn_in`'s Linear+bias+GELU.
  13/13 correct on real GPU (`input_precision="ieee"` fix for the TF32
  default bug — journal, job `778135`/`778149`), standalone geomean 1.85x
  vs plain best.py, but shape 8 regressed to 0.84x. A 12-config
  `@triton.autotune` sweep (journal iter 37, job `778315`) did NOT fix
  shape 8 (0.83x, unchanged) and made most *other* shapes slightly worse
  (geomean fell to 1.73x) — proved it's a precision/structural mismatch,
  not a tuning gap (see M2 below).
  **T10-fp16 follow-up, CLOSED — real negative result (journal iter 38,
  `candidates/v_triton_fused_ffn_fp16.py`, job `778353`, both array tasks
  COMPLETED):** explicit fp16-compute variant (casts x/weight to fp16
  before `tl.dot`, fp32 accumulate/epilogue, no `input_precision` flag
  needed since TF32 truncation is fp32-input-only). **13/13 correct**
  (max_abs ≤0.0013, well inside budget) — the correctness risk flagged at
  write-time did NOT materialize. But speed on the 3 shapes that actually
  matter for this fix (amp-routed #6/#8/#13, standalone-vs-plain-best.py
  speedup) is **still below `v_router2`'s current amp route**: shape 6
  2.24x vs 3.29x, shape 8 1.31x vs 1.74x, shape 13 4.38x vs 11.66x. fp16
  did move shape 8 off its 0.83x floor (real improvement over T10-fp32),
  but a hand-written Triton fp16 GEMM still can't beat cuBLASLt's
  tensor-core GEMM at these sizes — consistent with M2's profiler finding
  that GELU itself is only 1.92% of shape 8's time, so fusion has almost
  no addressable overhead left once the 55.6% GEMM floor is fixed cost.
  **Conclusion: do not route T10 (either precision) to #6/#8/#13 — leave
  `amp`'s plain `nn.Linear`+GELU there.** T10-fp32 remains real, correct,
  and a net win for the *other* 10 shapes where it doesn't compete against
  cuBLASLt fp16 tensor cores (already reflected in the 1.85x fixed-config
  standalone geomean). No further T10 GPU time planned — closing this
  thread; if someone wants to keep pushing shape 8 specifically, the next
  real lever would have to beat cuBLASLt's GEMM itself (e.g. a genuinely
  different algorithm/library), not more fusion around it.

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

- **S6 — OPEN, root cause unfixed even though the current symptom is
  patched.** (`docs/research-batch2.md`/`research-batch3.md`, never
  formally posted as a claim-board item until now). 31 of 32
  `journal.jsonl` rows have no structured `per_shape` array — every
  result table in `README.md`/`TECH_REPORT.md` has been *hand-transcribed*
  from prose numbers each time the leaderboard moved, not mechanically
  regenerated from the ledger. **Checked directly (2026-08-31): the
  README table is NOT currently stale** — it already carries the full
  2.99x/3.72x, 13/13-shape data with correct routing labels (synced by
  hand in commit `ea789ef`) — so the specific staleness `research-batch3.md`
  flagged is fixed for now. **But the mechanism that keeps it in sync
  doesn't exist**, so it will drift again the next time any candidate's
  numbers change (S8's own repeat-sweeps, T10, T11, or anything else in
  this backlog that lands). **Action:** either (a) fix `bench_harness.py`'s
  postmortem/logging step to actually populate `eval.per_shape` as a real
  array (schema already exists per `AGENTS.md` §4, just never populated),
  so future updates can regenerate tables mechanically instead of by hand,
  or (b) at minimum, add a comment/checklist item to the "guarded best
  update" procedure in `AGENTS.md` §2 reminding whoever updates
  `leaderboard.md` to also re-sync `README.md`'s table in the same commit,
  so the two don't diverge silently again. (a) is the real fix; (b) is the
  cheap stopgap if there's no time for (a).

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
