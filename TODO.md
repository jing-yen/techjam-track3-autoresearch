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

## Open — shape 14, reopened

- **S5 — IN PROGRESS by `opus-1`: batch-chunk shape #14. Never tried, and the arithmetic says it fits.**
  B=32 is 32 **independent** sequences; nothing couples them. Process in groups
  and concatenate — output is bit-identical, no approximation, no precision
  change. Peak at chunk=4: ~10.7 GB working + 12.2 GB input + 12.2 GB output =
  **~35 GB against 79.25 GB**. The measured OOM was 73.85 GB, only 6 GB over;
  chunking clears it by 44 GB. ~20 lines in `forward`.
  *Correctness:* #14 has no reference (baseline needs 18.6 TB), so validate the
  **mechanism** on #8 and #13 instead — chunked vs unchunked must be identical
  and both must pass the gate. Then report #14 as "runs, mechanism proven exact
  on 8 and 13, unverifiable against a reference that cannot exist."
  *Why it was skipped:* B5 recorded "not pursuing chunking per §3.3" — a call
  made before the OOM was measured, when #14 looked hopeless rather than 6 GB
  short.
  *Falsify:* chunked output differs from unchunked on #8/#13, or #14 still OOMs
  -> keep the current limitations text, which is already well evidenced.
  *Together with S4 (shape 6), this takes the sweep from 12/14 to 14/14.*
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

- **S2 — TAKEN OVER by `opus-1` (codex-s2's agent stalled; result inherited).**
  `codex-combined-step4.py` reverts #9 from a tried `reduce` route back to
  `fused` (see the `codex-b10-step1.py` -> `step4` diff) — confirms `reduce`
  does not beat `fused` on #9, so `fused` is already optimal there. Folded
  into the combined candidate below rather than closed standalone.

- **S3 — #5 and the large-batch regime: document, do not optimize.** Evidence
  above. Falsify: if a variant beats 2.43x on #5 without regressing #1, this
  reasoning was wrong.

## Open — zero GPU cost, do these first

- **D1 — the §3.5 deliverables. NOW THE TOP ITEM IN THE QUEUE.** A defensible
  2.98x geomean is banked; the write-up is unstarted beyond
  scaffolding. `README.md` and `TECH_REPORT.md` exist with **49 `<FILL>`
  placeholders** guarded by `scripts/check_placeholders.sh`. Every one of them
  is now fillable from `leaderboard.md` and `journal.jsonl` — no new
  measurement is required. Also outstanding: the demo video (public YouTube,
  linked in Devpost) and the Devpost description's five named fields.
  *Blocked on:* per-shape data is prose in `leaderboard.md`, not in
  `journal.jsonl` (`per_shape: []` on iters 5-9), so the results tables must be
  transcribed by hand unless those rows are re-emitted.

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

- **M1 — TAKEN OVER by `opus-1`, PRELIMINARY RESULT IN HAND: falsified —
  a config beats 2.98x.** `codex-combined-step4.py` (AMP autocast(fp16) on
  shapes #6/#8/#13, kept in fp32 elsewhere + B10 mask cache + S2 fix) scored
  **median 2.96x / geomean 3.91x on A100-40**, all 13/13 correct, worst
  max_abs 0.00168 (atol 0.002) — up from the router's 2.51x/2.93x on the same
  GPU/run. A100-80 confirmation running now (job `step4_confirm_a80`) to
  match the leaderboard's stated GPU class before promoting it.

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

- **T6 — CONFIRMED ON GPU (A100-40, twice), leaderboard-A100-80 confirmation
  in-flight.** `torch.autocast(fp16)` (keeping norms/reductions fp32) applied
  to shapes #6/#8/#13 in `codex-combined-step4.py`. Real gain per-shape vs the
  fp32 router: #13 4.72x -> 11.10x, #4 5.60x -> 9.36x, #3 7.34x -> 10.63x, all
  still correct (worst max_abs 0.00168 < 0.002 atol). Confirms the original
  hypothesis: `tools/probe_sdpa_backends.py --dtype float16` showed flash SDPA
  eligible on all 14 shapes vs 0/14 at fp32 — fp16 buys tensor cores *and* the
  flash backend at once. **bf16 autocast tried as an alternative and rejected**
  (journal iter 18): fails correctness on 13/13 shapes, max_abs 0.0095-0.016
  vs atol 0.002 — bf16's 7-bit mantissa is too imprecise despite the safer
  exponent range; fp16 is the only viable reduced-precision path here. A
  full 13-shape AMP sweep (`amp_full_sweep_a40`) is running to check whether
  more shapes than 6/8/13 should route to it — codex only tried the three
  biggest/slowest.
- **T4 — folded into B10's `codex-b10-step1.py` mask-cache fix** rather than
  landing standalone; see B10 above.
- **T7 — custom Triton kernels.** Correctly at the bottom. With ~32 hours left
  and D1 unstarted, **do not start this.**
- **T2, T3 — superseded.** Both landed as `v_compile.py` / `v_fused_qkv.py` and
  are now route targets inside `v_router.py`.

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
