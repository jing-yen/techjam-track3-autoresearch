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

## Open — correctness / blocking

- **B1 — Kill the dead `is_causal` path. Highest value in the repo.**
  Because of READ FIRST #1 the seed always takes `best.py:75-93` and materializes
  an additive `[B,1,S,S]` float mask. Two costs:
  (a) an explicit `attn_mask` **disqualifies SDPA's flash backend on every
  shape** — the seed is running mem-efficient or math everywhere;
  (b) shape #14's mask is **1192 GB fp32**, so the candidate OOMs too.
  Wasted alloc per layer per call elsewhere: #6 = 0.61 GB, #13 = 0.25 GB.
  **Fix:** compute `has_padding = not bool(valid_token_mask.all())` **once** in
  the top-level `forward` (one sync per call, not per layer), thread the bool
  down, and use `is_causal=causal` with no `attn_mask` when it is False.
  Keep the existing additive-mask branch for the padded case.
  Gate: re-run `--padding-ratio 0.0` and `--padding-ratio 0.3` — both must pass.

- **B4 — RESOLVED, no action needed. Tolerance is `rel < 0.02, abs < 0.002`.**
  Problem statement §3.2: "the diff should be small enough (relative error <
  0.02, abs error < 0.002)". This matches the argparse defaults at
  `torch_transformer_benchmark.py:618-619` and what the repo already uses
  (`PROGRAM.md:12`, `bench_harness.py:340-341`). The docstring at
  `torch_transformer_benchmark.py:11` (`atol=0.001, rtol=0.01`) is **stale text
  the organizer never updated** — ignore it. An earlier revision of this file
  told you to develop against the tight pair. That was over-cautious. Keep
  0.002/0.02 and do not change the harness defaults.

- **B8 — Match the official timing protocol in `bench_harness.py`.**
  Official: warmup 20, repeats 100, rounds 3, **alternating** baseline/candidate
  (`torch_transformer_benchmark.py:622-624, :546-560`).
  Harness: warmup 5, repeats 20, rounds 1, **sequential blocks**
  (`bench_harness.py:343-345, :231-243`). Current harness numbers are noisier and
  order-biased; do not report them as speedups. Note neither flushes L2 and both
  reuse one fixed input (`:529-536`), so this measures pipelined steady state.

- **B9 — Hoist `load_candidate` out of the per-shape loop** (`bench_harness.py:168`).
  Re-imports the candidate 12x per sweep, discarding any `torch.compile` cache and
  inflating compile cost 12x. Blocks honest measurement of T1/T2.

## Open — measurement before more code

- **B2 — RESOLVED (iter 8, real A100 data).** `tools/probe_sdpa_backends.py`
  forced each backend per official shape via `torch.nn.attention.sdpa_kernel`
  on A100-80, fp32. **Flash is eligible on ZERO shapes** (confirms
  fp16/bf16-only, independent of head_dim) — **`mem_efficient` fires on all
  14**, including the small-head-dim shapes (#7 head_dim=8, #1/#12/#13
  head_dim=32) that B7 predicted would be flash-eligible by head_dim alone.
  `PROGRAM.md:61`'s "biggest single win [is flash]" is **false as measured**:
  mem-efficient SDPA is what has been running the whole time, on every
  candidate, in every prior benchmark run in this repo. This does not change
  any correctness or speedup number already on the leaderboard (mem-efficient
  was already what fired) — it just settles what to call it in the report.
  See `.runs/sdpa_backend_probe.json` and `LOG.md` iter 8.

- **B7 — head_dim limits.** sm80 flash caps head_dim at 128. Per shape:
  #8 = **256 (over the limit — flash impossible)**, #9 = **128 (exactly at it)**,
  #14 = 64, #1/#12/#13 = 32, #10 = 64, #7/#11 = 8. Feeds T5's dispatch table.

- **T8 — Profile #2 (0.12 GFLOP) against #6 (1174 GFLOP).** Handoff's
  launch-overhead hypothesis for the small shapes is still UNPROFILED. Settle it
  before assigning R1 work. Note the official timer records CUDA events
  back-to-back with no per-iteration sync (`:492-500`), so the CPU can run ahead
  and launch cost surfaces as the serial bottleneck when GPU work is sub-100us.

## Open — shape 14

- **B5 — Shape #14 is infeasible in fp32 on any GPU we have.** One `[B,S,D]`
  activation is 12.2 GB fp32 / 6.1 GB fp16; seven live tensors is **~85 GB fp32 /
  ~43 GB fp16**. `cluster.config.json` defaults to `gpu:a100-80:1` **and** fp32 →
  impossible even with a perfect kernel. fp16 on A100-80 is borderline; H100-96
  fp16 fits. The only fp32 path is chunking over the batch, which changes what the
  measured latency means. **RESOLVED — do not spend hours here.** §3.5 wants it as a documented
  limitation, not a solved shape. See "Resolved by the problem statement" (the reference cannot produce ground truth at all —
  `torch_transformer_benchmark.py:97` materializes the scores).

## Open — optimization (unchanged intent, re-ranked)

- **T1 — `torch.compile(mode="reduce-overhead")`** on the whole model. Do **after**
  B1 and B9. Expect the launch-overhead-bound shapes (#2 batch=1, #3, #12 seq=32)
  to gain most. Warm-compile each shape; verify guards do not break the gate.
  Note `--compile-user` is an official flag (`:628`, applied at `:703` after
  weight-copy and `.eval()`) — the organizer sanctions this — but do not depend on
  the grader passing it. Compile inside the candidate.
- **T2 — `torch.compile(mode="max-autotune")`.** Compare against T1 on the
  matmul-bound shapes (#8 d=1024, #6 batch=10000). Watch compile time against the
  30 min `--time` default in `cluster.config.json`.
- **T4 — Memory-layout cleanups** feeding SDPA `[B,H,S,D]`. Drop the
  `.contiguous()` at `best.py:95` if the following `view` allows it; check strides.
  Merge with B2's backend logging — same run.
- **T3 — Fused QKV projection** (`Linear(d_model, 3*d_model)`). Set
  `STRICT_WEIGHT_COPY=False` + a `copy_model_weights` that splits the fused
  weight/bias. `bench_harness.py:169-177` already honors both knobs.
- **T6 — fp16/bf16 path** with the fp32 softmax reduction kept. Risky, and only
  worth it if the organizer tests those dtypes. But it is the **only** route to
  flash (B2) and the only route to shape #14 (B5), so it is no longer optional if
  either of those matters. Verify the gate on all 13 feasible shapes.
- **T7 — Custom Triton: fused LayerNorm+residual**, later fused FFN GELU.
  Late-stage. Only where a profile shows remaining overhead after B1 + T1.

## In progress

_(none yet — claim something above)_

## Done

- **T5 — Per-shape dispatch** → `candidates/v_router.py`. Routes each shape to
  whichever of best/v_compile/v_fused_qkv empirically won it (measured, not
  the head_dim/backend-table heuristic originally proposed — the three
  candidates' relative strengths didn't reduce to one clean rule across the
  12-shape sample). A100-80, official protocol: 2.27x median / 2.47x
  geomean, beats every single candidate, 12/12 correct. New leaderboard
  best (iter 9). (agent: opus-1)
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

