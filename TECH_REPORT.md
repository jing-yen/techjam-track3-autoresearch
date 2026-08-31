# Tech Report — TikTok TechJam 2026, Track 3

**Implement a GPU Kernel for a Transformer Layer**

> Placeholders are marked `<FILL>`. Run `scripts/check_placeholders.sh` before
> submitting. No number in this document may be written by hand; every one comes
> from a `journal.jsonl` row produced by `bench_harness.py`.

---

## 1. Summary

We optimize the organizer's pre-LayerNorm Transformer block across the 14
published input shapes. Rather than hand-tuning a single kernel, we built an
**autonomous multi-agent research loop** that proposes candidate implementations,
measures them against the organizer's own correctness and timing code, and keeps
or prunes on the measured result.

Headline: **median speedup 2.98x, geometric mean 3.81x** on an **NVIDIA A100-80
PCIe** at **float32** (with fp16 `autocast` and a custom Triton kernel fusing
residual-add into LayerNorm at both boundaries per layer on the three shapes
that benefit — §4), with **13 of 14 shapes** passing the
correctness gate. The 14th (shape #14) is not absent for memory reasons
anymore — a targeted fix (§8) makes it run — but it still has no reference to
score it against, so it is reported separately rather than in the 13-shape
aggregate.

---

## 2. Environment

Captured mechanically by `scripts/capture_env.sh` on the GPU node. Full dump in
`docs/environment.txt`.

| | |
|--|--|
| GPU | NVIDIA A100-80 PCIe (compute capability 8.0), NUS SoC cluster (leaderboard numbers); A100-40 PCIe also used for parallel exploration jobs, same software stack |
| GPU memory | 79.25 GiB usable on the A100-80 nodes (measured at the shape-14 OOM boundary) |
| CPU | AMD EPYC 7352, 2 sockets × 24 cores (96 threads total) |
| System memory | 251 GiB, 247 GiB available |
| Disk | `cfs.comp.nus.edu.sg:/mnt/storpool/home`, 170 TiB total, 35 TiB free |
| OS / kernel | Linux 6.8.0-138-generic (Ubuntu), x86_64 |
| Python | 3.12.3 |
| PyTorch | 2.10.0+cu128 |
| CUDA / cuDNN | CUDA 12.8 (driver 580.173.02), cuDNN 9.10.02 |
| Triton | 3.6.0 |
| Scheduler | Slurm, `gpu:a100-80:1` (leaderboard) or `gpu:a100-40:1`/`gpu:h100-47:1` (exploration), one exclusive GPU per benchmark job |

Development machines are Apple Silicon MacBooks with no CUDA. They run CPU
correctness tests only; every timing number in this report comes from the GPU
node.

**TF32 note — we measured both configurations.** The organizer's script enables
TF32 by default for both sides (`torch_transformer_benchmark.py:638-645,
:684-688`). An earlier revision of our candidate disabled it globally at module
import, because `torch.compile`'s autotuner selected TF32 GEMM kernels for the
candidate while the baseline used cuBLAS, drifting ~0.005 against the 0.002
absolute tolerance on 9 of 12 shapes. Since `allow_tf32` is process-global, that
also de-accelerated the reference: an internally fair comparison, but not the
organizer's default, and on GEMM-bound shapes a slowed reference would have
flattered our ratio.

We flagged this against ourselves and re-measured with the pin scoped to the
compiled path and TF32 otherwise left at the organizer default. All 12 shapes
still pass (`max_abs` ~0.001, 2x under the gate) and the geometric mean **rose**
from 2.47x to 2.98x. **All numbers in this report are the organizer-default
configuration.** We record the episode because the configuration we suspected of
flattering us was in fact the pessimistic one, and only measurement settled it.

---

## 3. Method: the autoresearch loop

Two agent layers that communicate only through git.

```
  RESEARCH LAYER                        IMPLEMENTATION LAYER
  research-loop.sh                      autoresearch.workflow.js
  ─────────────────                     ────────────────────────
  plan      (Claude Opus 5, max)        sync -> strategist -> coder
  review    (Codex gpt-5.6-sol, ultra)       -> runner -> postmortem
  reconcile (Claude Opus 5, max)
        │                                        │
        └────────► TODO.md ─────────────────────►│
                                                 │
        ◄───────── journal.jsonl ◄───────────────┘
```

**The ledger is the arbiter.** `bench_harness.py` imports the organizer's
`compare_outputs`, `generate_random_case`, `warmup_model` and `benchmark_once`
rather than reimplementing them, so our reported numbers come from the same code
path the task defines. Each experiment appends one JSON row to `journal.jsonl`
recording the candidate, shape, pass/fail, max abs and rel error, both latencies,
the speedup, and the environment. **No task is considered done until a ledger row
exists.** Claims without a row are not admitted to this report.

**Correctness precedes speed.** The harness refuses to report a speedup for a
candidate that fails the gate, mirroring the organizer script's behavior
(`torch_transformer_benchmark.py:725-728`).

**One writer per file** keeps concurrent agents from colliding: `TODO.md` is
written only by the research layer, `candidates/<agent-id>-*.py` only by its
owning agent, `journal.jsonl` only by the harness, `candidates/best.py` only
through a guarded compare-and-swap after a fresh pull.

---

## 4. Optimizations

| # | Change | Rationale | Status | Result |
|--|--|--|--|--|
| T0 | `scaled_dot_product_attention` replaces the explicit `[B,H,S,S]` score matrix | Reference builds `QK^T` explicitly and softmaxes in fp32 (`torch_transformer_benchmark.py:97, :111`); a fused kernel avoids materializing scores | landed | 2.07x / 1.96x standalone |
| B1 | Remove the unreachable `is_causal` fast path so the fused causal kernel is actually selected | `valid_token_mask` is never `None` (`torch_transformer_benchmark.py:255-259`), so the seed always built an additive `[B,1,S,S]` mask | landed (`1f99f8d`) | correctness/routing fix; no isolated speedup measured |
| T1 | `torch.compile(mode="reduce-overhead")` | Small shapes (#2 B=1, #3, #12 S=32) are launch-overhead bound (confirmed: baseline is ~1.87 ms for B=1, 4, 16 alike) | landed → `v_compile_reduce.py` | 2.29x / 2.39x; best on #3, #4, #5 |
| T3 | Fused QKV projection, one `Linear(d, 3d)` | Three kernels become one; requires a custom weight mapping (`STRICT_WEIGHT_COPY=False`) | landed → `v_fused_qkv.py` | 2.16x / 2.09x |
| T5 | Per-shape dispatch on `(B, S, d, H)` | The rules explicitly allow shape checks; no single candidate wins everywhere | landed → `v_router.py` | **2.27x / 2.47x — best** |
| T6 | fp16 via `torch.autocast`, norms and reductions kept fp32 | Backend probe shows flash eligible on 14/14 shapes at fp16 vs 0/14 at fp32 | written, **not validated on GPU** | blanket cast failed 11/12; autocast untested |
| T7 | Triton fused LayerNorm + residual ("AddNorm") | Reserved for remaining overhead after the above | **landed** | Roofline recheck found shape #8 at only ~28% of A100 fp16 peak — real headroom, reopened. Custom Triton kernel confirmed on A100-80, 13/13 correct: standalone +7.4% geomean vs plain LayerNorm; integrated into `v_router2`'s `best`/`amp` eager routes (kept separate from `compile`/`reduce`, which already get fusion from `torch.compile`/Inductor). Attributable per-shape gain on the AMP-routed shapes it touches: #6 +11%, #8 +4%, #13 +6%. See `TODO.md` T7 / `candidates/v_triton_addnorm.py` / `candidates/v_router2.py` |
| T15 | Extend the T7 AddNorm kernel to the SECOND residual+norm boundary (ffn_out-add fused into the next layer's norm1, or into `final_norm`) | A real `torch.profiler` trace (M2, not Roofline inference) found T7 only covered one of two boundaries per layer; the unfused one measured 19.21% of shape #13's total CUDA time — bigger than T7's own already-fused kernel next to it | **landed** | Same kernel as T7, unmodified — only the block/transformer wiring changes so the fusion spans a layer boundary. Confirmed on A100-80, 13/13 correct, worst max_abs unchanged (0.00176). Per-shape gain on the 3 shapes it touches: #6 +17.2%, #8 +4.2%, #13 +12.5% — all above the ±2.7% measurement noise floor. See `TODO.md` T15 / `candidates/v_triton_addnorm2.py` / `candidates/v_router2.py` |

**Correctness invariants preserved throughout** (full list in `PROGRAM.md`): exact
erf GELU, fp32 softmax reduction, `1/sqrt(head_dim)` scale, `triu(diagonal=1)`
causal mask, invalid key positions masked before softmax, padded query rows zeroed
after attention and after every block and after the final norm, output shape
`[B,S,d]`, and baseline-compatible parameter names so
`copy_model_weights(..., strict=True)` succeeds.

**Why the optimization list above never includes pruning, quantization below
fp16, distillation, or architecture simplification.** The efficient-transformer
literature splits cleanly into two families. *Model efficiency* (structured
pruning, block/layer dropping, distillation, low-bit (4/8-bit) quantization,
architecture redesigns like removing skip connections or projections)
produces a **different, cheaper model** — evaluated against task metrics
(F1, perplexity) that tolerate the result no longer matching the original
model's raw output tensors. *Kernel/systems efficiency* (fusion, dispatch,
compilation, precision within tolerance, memory layout) computes the
**same model faster** on the same hardware, with output unchanged up to
floating-point rounding. The correctness gate here — every element within
0.002 absolute or 2% relative of the fixed baseline's output — puts this
task entirely in the second family by construction, not by choice: a pruned
or distilled model's output diverges from the dense original by far more
than that bound, so those techniques are inadmissible regardless of their
reported speedups. (We separately measured bf16 autocast, one step milder
than 4/8-bit quantization, and it already fails this gate on 13/13 shapes —
see `TODO.md` T6.) Every optimization landed or attempted in this report is a
kernel/systems-level technique for that reason.

---

## 5. Results

Candidate `candidates/v_router2.py`, A100-80, official protocol, job
`router2_t15_confirm` (journal iter 42).

| # | B | S | d | H | baseline ms | ours ms | speedup | routed to |
|--|--|--|--|--|--|--|--|--|
| 1 | 64 | 128 | 128 | 4 | 2.625 | 1.210 | 2.17x | compile |
| 2 | 1 | 128 | 128 | 4 | 1.914 | 0.274 | **6.97x** | compile |
| 3 | 4 | 128 | 128 | 4 | 1.958 | 0.227 | 8.61x | reduce |
| 4 | 16 | 128 | 128 | 4 | 1.938 | 0.275 | 7.04x | reduce |
| 5 | 128 | 128 | 128 | 4 | 2.723 | 1.010 | 2.70x | reduce |
| 6 | 10000 | 128 | 128 | 4 | 185.926 | 48.196 | 3.86x | amp (fp16 + Triton AddNorm x2) |
| 7 | 64 | 128 | 32 | 4 | 1.900 | 0.483 | 3.93x | compile |
| 8 | 64 | 128 | 1024 | 4 | 8.025 | 4.422 | 1.81x | amp (fp16 + Triton AddNorm x2) |
| 9 | 64 | 128 | 128 | 1 | 1.779 | 0.797 | 2.23x | fused |
| 10 | 64 | 128 | 128 | 2 | 1.960 | 0.794 | 2.47x | fused |
| 11 | 64 | 128 | 128 | 16 | 3.478 | 1.166 | 2.98x | fused |
| 12 | 64 | 32 | 128 | 4 | 1.937 | 0.785 | 2.47x | fused |
| 13 | 64 | 1024 | 128 | 4 | 43.236 | 3.294 | **13.12x** | amp (fp16 + Triton AddNorm x2) |

All 13 pass the correctness gate, worst `max_abs` 0.00176 (shape 8), still under
the 0.002 absolute tolerance. TF32 is enabled on every route except `compile`
(S1 found an asymmetric-TF32-kernel-selection bug specific to `torch.compile`'s
max-autotune path); shapes 6/8/13 additionally run under fp16 `autocast` (T6)
plus the custom Triton AddNorm kernel at both residual+norm boundaries per
layer (T7 + T15). Shape 14 is excluded from this sweep for a different reason
than memory — it *runs* (§8), but has no reference to compare against, so it
cannot be scored on this correctness-gated table at all.

Aggregates: **median 2.98x, geometric mean 3.81x** over the 13 shapes with a
reference.

**Where the remaining time is.** Shape 8 and shape 6 are still the largest
absolute-time shapes despite their fp16 route. Shape 8's Roofline reading
(measured before the fp16/Triton work, at fp32/TF32 on the `fused` route:
420.9 GFLOP in 26.284 ms = 16.0 TFLOP/s, 82% of the A100's 19.5 TFLOPS fp32
peak) motivated reopening T7 once the route changed to fp16 — the correct
comparison there is now against the ~312 TFLOP/s dense fp16 Tensor Core
ceiling, where the *current* number (420.9 GFLOP / 4.422 ms ≈ 95 TFLOP/s) is
still only about 30% of peak — real headroom remains, which is exactly why
T7 and T15 (Triton AddNorm, both boundaries) targeted this shape and
produced a measurable, if modest, gain there (§4). A real `torch.profiler`
trace on this shape (M2, §4) later replaced this Roofline inference with a
measured per-kernel breakdown: 55.6% fp16 tensor-core GEMM (real, expected),
17.1% fp32↔fp16 casting overhead (a genuinely new finding), with the fused
AddNorm kernels and GELU together under 10% — the remaining ~30%-of-peak
gap here is compute- and casting-bound, not further fusable with kernel
tricks at this point (see T10/K2' in `TODO.md` for the fusion attempts that
were tried and found to plateau around cuBLASLt's own GEMM performance).

**The head-count sweep measures the reference, not us.** Shapes 1, 9, 10 and 11
are identical arithmetic. Our runtime is comparatively flat across them; the
baseline grows with head count because it materializes `[B,H,S,S]` and performs
more transpose work as heads increase — part of why shape 9 (one head) shows
the smallest ratio (2.25x) despite an efficient candidate.

**Independent check — not performed.** We did not reproduce any shape through the
unmodified `torch_transformer_benchmark.py`. Our harness reuses that script's own
`compare_outputs`, `generate_random_case`, `warmup_model` and `benchmark_once`
(§3), so the code path is shared rather than reimplemented, but a shared code
path is not the same evidence as an independent run. We record this as a gap.

**Run-to-run variance, measured directly (not just inferred).** Rather than
lean only on cross-iteration comparisons, we ran the current champion twice
back-to-back with zero code changes (job `778942`, journal iter 49) to get a
real error bar. **Aggregate geomean moved +3.3% (3.771x→3.895x), median
+3.4%** between two identical runs — genuinely comparable in size to T15's
own claimed aggregate delta, so the aggregate number alone should be read as
"a legitimate official-protocol number," not as proof the aggregate *delta*
is signal rather than noise. But per-shape, that noise is highly
non-uniform, not flat: **shape 1 (`compile` route) swung +35.7%** between
the two identical runs — the largest single-shape swing measured in this
project — while **every `amp`-routed shape (#6/#8/#13 — the only shapes T15
touches) swung under 0.5%** (#6 −0.04%, #8 +0.4%, #13 +0.2%). Measured
against that real, shape-specific noise floor, T15's per-shape gains (#6
+17.2%, #8 +4.2%, #13 +12.5%) are 10-30x above noise — solidly signal, not
the aggregate coincidence a flatter noise estimate might suggest. The
earlier cross-iteration comparison (iter 21→30, T7's landing: shape 7 swung
-30%, 5.85x→4.08x) is consistent with this — `compile`-routed shapes carry
real, large run-to-run variance unrelated to any candidate change; `amp`-
routed shapes do not. A seed-robustness check (S9, same job) confirms shape
8's correctness is stable across seeds too (max_abs 0.00165/0.00160 at two
extra seeds, vs 0.00176 at the default), not a lucky single-seed pass.

---

## 6. AI skills and tools used

The problem statement encourages AI-assisted development and awards bonus points
for describing it. Our use of AI is not incidental to the solution; it *is* the
solution's architecture.

### 6.1 Models and roles

| Stage | Model | Reasoning effort | Why |
|--|--|--|--|
| Research: plan | Claude Opus 5 (Claude Code) | max | Ranking hypotheses under uncertainty is the judgment-heavy step |
| Research: adversarial review | OpenAI Codex `gpt-5.6-sol` | ultra | A *different model family* auditing the plan; correlated reviewers catch less |
| Research: reconcile | Claude Opus 5 | max | Applying or rejecting each finding, with reasons recorded |
| Implementation: strategist / postmortem | strong model, high effort | — | Direction choice and failure diagnosis |
| Implementation: coder / runner | cheap model, low effort | — | Mechanical variant writing and job submission |

### 6.2 The structured handoff between models

The review leg runs read-only and emits JSON validated against
`schemas/review.json`. That schema is what makes the handoff mechanical rather
than interpretive: every finding must carry an `item_id`, a `severity`, a
`category` from a fixed enum, and — critically — an **`evidence` field containing
a `file:line`, a ledger row, or a URL. A finding with no source is dropped.**

```bash
codex exec --sandbox read-only --model gpt-5.6-sol \
  -c model_reasoning_effort=ultra \
  --output-schema schemas/review.json \
  --output-last-message .research/review.json \
  "$(cat prompts/review.md)"
```

The review prompt deliberately **forbids the reviewer from proposing new
optimizations.** An earlier iteration allowed it and produced confident,
unfalsifiable suggestions that cost a cycle to disprove. Narrowed to auditing
citations, arithmetic and feasibility against the actual source, the same model
became reliably useful. This is the single most important lesson we learned about
multi-model collaboration.

### 6.3 Token and cost discipline

Each agent turn reads a pinned set — the correctness contract, its own champion
candidate, and the last ~20 ledger rows, roughly 15k tokens — rather than the
whole repository. Agents waiting on the exclusive GPU lock sleep rather than
polling with model calls.

**Measured, not estimated.** One full research-loop pass (plan -> review ->
reconcile) on 2026-08-30 cost **$3.89 in Claude usage** across 1.49M tokens, plus
3.32M tokens on Codex:

| leg | model / effort | total tokens | fresh input | output | cached input | cost |
|--|--|--|--|--|--|--|
| plan | Opus 5 / max | 278,793 | 8 | 15,604 | 184,049 | $1.27 |
| review | `gpt-5.6-sol` / ultra | 3,320,741 | 141,734 | 22,015 | 3,156,992 | — |
| reconcile | Opus 5 / max | 1,210,355 | 26 | 38,935 | 1,060,000 | $2.62 |

Over 95% of input was served from cache. The reconcile leg dominates because it
re-opens and re-verifies every reviewer finding at its cited `file:line` rather
than accepting it — which is precisely the step that caught the reviewer being
right (§6.4).

### 6.4 A worked example of the loop catching a real defect

The AI-generated seed candidate contained a fast path guarded by
`if valid_token_mask is None`, intended to let SDPA handle causal masking
internally without materializing a mask. It passed every CPU correctness test.

The audit leg cross-referenced the guard against its callers and found that
`generate_random_case` returns an **all-True mask, not `None`**, whenever
`padding_ratio <= 0` (`torch_transformer_benchmark.py:255-259`), and that every
call site passes it (`:391, :392, :472, :494, :504`). The fast path was therefore
unreachable. The candidate silently took the slow branch on every shape, building
an additive `[B,1,S,S]` mask that disqualifies the FlashAttention backend — and
which for shape 14 would have been 1192 GB.

Correctness tests could not have found this: the slow path is *correct*, just
slow, and on CPU the backend distinction does not exist. It took a reviewer with
the source and an explicit instruction to verify every citation.

The fix landed in `1f99f8d`. We did not measure B1 in isolation — no A100 sweep
exists from before it, so no attributable before/after delta can be quoted, and
we do not quote one. What can be said precisely is that the unreachable branch
would have allocated a 1192 GB mask on shape 14 and forced an additive-mask code
path on all twelve measured shapes.

**A second worked example, in the opposite direction.** The same review loop
produced a confident, well-cited claim that was still wrong. An early queue item
asserted that sm80 caps FlashAttention head_dim at 128, excluding shape 8. The
Codex review corrected the cap to 256, citing `sdp_utils.cpp` in the PyTorch
source — a real correction. Both were then superseded by measurement: a direct
backend probe showed flash eligible on **0 of 14 shapes at fp32 and 14 of 14 at
fp16**. The gating variable was never head_dim; it was dtype. We record this
because it is the clearest evidence in the project for why the ledger, not the
review loop, is the arbiter: two rounds of well-sourced reasoning were closer to
each other than either was to the measurement.

### 6.5 Tools

Claude Code (CLI, non-interactive `-p` mode driving `research-loop.sh`), OpenAI
Codex CLI (`codex exec`, read-only sandbox, structured outputs), git as the
shared agent state store, Slurm array jobs to amortize queue latency across
candidate batches, PyTorch profiler and `torch.nn.attention.sdpa_kernel` for
backend attribution.

---

## 7. What we verified versus what we assumed

Stated explicitly because an unmarked assumption is how a benchmark result becomes
wrong without anyone noticing.

**Verified against source, with line numbers:** the correctness rule is an
elementwise OR with inclusive `<=` (`:314-316`); NaN and Inf fail outright
(`:309`); the reference builds `QK^T` explicitly and does not use SDPA (`:97`);
softmax is computed in fp32 and cast back (`:111`); the script's own defaults
match none of the 14 official shapes (`:598-604`); TF32 is on for both sides
(`:638-645`); a failing candidate is never timed (`:725-728`).

**Verified by arithmetic:** shape 14's memory floor; the 18.6 TB score matrix;
head_dim 256 on shape 8 exceeding the sm80 FlashAttention limit of 128.

**Assumed, and why it is acceptable:** that the docstring at
`torch_transformer_benchmark.py:11` (`atol=0.001, rtol=0.01`) is stale and the
problem statement §3.2 (`abs < 0.002, rel < 0.02`) is authoritative — the argparse
defaults at `:618-619` agree with the problem statement, so the docstring is the
outlier.

---

## 8. Limitations

See the "Limitations" section of `README.md`, which covers shape 14 (why it
originally OOMed, the batch-chunking fix that makes it run in ~74.6s, and why
it still has no reference to score it against), the steady-state nature of the
timing protocol, the self-administered correctness gate, and the PyTorch-only
scope.
