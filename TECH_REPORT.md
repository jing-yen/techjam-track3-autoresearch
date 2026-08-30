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

Headline: median speedup `<FILL>`, geomean `<FILL>`, on `<FILL: GPU>` at
`<FILL: dtype>`, with `<FILL>` of 14 shapes passing the correctness gate.

---

## 2. Environment

Captured mechanically by `scripts/capture_env.sh` on the GPU node. Full dump in
`docs/environment.txt`.

| | |
|--|--|
| GPU | `<FILL: e.g. NVIDIA A100-SXM4-80GB, sm80, driver <FILL>>` |
| GPU memory | `<FILL>` |
| CPU | `<FILL: model, cores, threads>` |
| System memory | `<FILL>` |
| Disk | `<FILL: filesystem, free space>` |
| OS / kernel | `<FILL>` |
| Python | `<FILL>` |
| PyTorch | `<FILL>` |
| CUDA / cuDNN | `<FILL>` |
| Triton | `<FILL>` |
| Scheduler | Slurm, one exclusive GPU per benchmark job |

Development machines are Apple Silicon MacBooks with no CUDA. They run CPU
correctness tests only; every timing number in this report comes from the GPU
node.

**TF32 note.** The organizer's script enables TF32 and sets
`float32_matmul_precision="high"` by default for **both** the baseline and the
candidate (`torch_transformer_benchmark.py:638-645, :684-688`). The reference
therefore already uses tensor cores in fp32. "Enable tensor cores" was not an
available optimization, and we report speedups against that already-accelerated
baseline.

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
| T0 | `scaled_dot_product_attention` replaces the explicit `[B,H,S,S]` score matrix | Reference builds `QK^T` explicitly and softmaxes in fp32 (`torch_transformer_benchmark.py:97, :111`); a fused kernel avoids materializing scores | landed | `<FILL>` |
| B1 | Remove the unreachable `is_causal` fast path so the fused causal kernel is actually selected | `valid_token_mask` is never `None` (`torch_transformer_benchmark.py:255-259`), so the seed always built an additive `[B,1,S,S]` mask, which disqualifies the FlashAttention backend | `<FILL: landed / not landed>` | `<FILL>` |
| T1 | `torch.compile(mode="reduce-overhead")` | Small shapes (#2 B=1, #3, #12 S=32) are suspected launch-overhead bound | `<FILL>` | `<FILL>` |
| T2 | `torch.compile(mode="max-autotune")` | Matmul-bound shapes (#8 d=1024, #6 B=10000) | `<FILL>` | `<FILL>` |
| T3 | Fused QKV projection, one `Linear(d, 3d)` | Three kernels become one; requires a custom weight mapping | `<FILL>` | `<FILL>` |
| T5 | Per-shape dispatch on `(S, d, H, B)` | The rules explicitly allow shape checks; head_dim ranges 8 to 256 across the suite | `<FILL>` | `<FILL>` |
| T7 | Triton fused LayerNorm + residual | Only where a profile showed remaining overhead | `<FILL: attempted? >` | `<FILL>` |

**Correctness invariants preserved throughout** (full list in `PROGRAM.md`): exact
erf GELU, fp32 softmax reduction, `1/sqrt(head_dim)` scale, `triu(diagonal=1)`
causal mask, invalid key positions masked before softmax, padded query rows zeroed
after attention and after every block and after the final norm, output shape
`[B,S,d]`, and baseline-compatible parameter names so
`copy_model_weights(..., strict=True)` succeeds.

---

## 5. Results

`<FILL: paste the per-shape table from README.md once measured>`

Aggregates: median speedup `<FILL>`, geomean `<FILL>` over `<FILL>` shapes with a
reference. Shape 14 is excluded; see §7.

Independent check: shape `<FILL>` reproduced through the unmodified organizer
script gave `<FILL>x`, against `<FILL>x` from our harness. `<FILL: agree /
explain the gap>`

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
polling with model calls. Estimated total spend: `<FILL>`.

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
the source and an explicit instruction to verify every citation. `<FILL: state
the measured speedup delta once B1 lands.>`

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

See the "Limitations" section of `README.md`, which covers shape 14's
infeasibility with the supporting arithmetic, the steady-state nature of the
timing protocol, the self-administered correctness gate, and the PyTorch-only
scope.
