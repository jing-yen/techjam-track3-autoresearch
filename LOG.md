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
