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
