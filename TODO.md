# TODO — Idea backlog & claim board

Claim an `Open` item by moving it to `In progress` with your agent-id (see
`AGENTS.md` §1). Ranked roughly by expected impact. Add follow-ups freely.

## Open

- **T1 — torch.compile (reduce-overhead)** on the whole model. Expect launch-overhead-bound shapes (#2 batch=1, #3, #12 short seq) to gain most. Warm-compile each of the 14 shapes; verify no correctness break from guards.
- **T2 — torch.compile (max-autotune)**. Heavier autotune; compare vs T1 on matmul-bound shapes (#8 d=1024, #6 batch=10000). Watch compile time.
- **T3 — Fused QKV projection** (`Linear(d_model, 3*d_model)`). Set `STRICT_WEIGHT_COPY=False` + provide a `copy_model_weights` that splits the fused weight/bias into q/k/v. Fewer kernels.
- **T4 — Memory-layout cleanups** feeding SDPA `[B,H,S,D]`: drop needless `.contiguous()`/transposes, check strides so the flash backend is actually selected. Profile which SDPA backend fires per shape.
- **T5 — Per-shape specialization**: branch on shape → tiny seq (#12,#2) eager/compiled small path; long seq (#13,#14) mem-efficient SDPA; large dims (#8) tuned matmul path. The rules allow shape checks.
- **T6 — fp16/bf16 path** with fp32 softmax reduction. Only if the organizer tests those dtypes; verify the gate carefully (risky).
- **T7 — Custom Triton: fused LayerNorm+residual** (and later fused FFN GELU). Late-stage; only where profiling shows remaining overhead after SDPA+compile.
- **T8 — Profile shapes #13 and #14** (attention-bound) to confirm the SDPA backend and find the next bottleneck before writing more code.

## In progress

_(none yet — claim something above)_

## Done

- **T0 — SDPA seed** → `candidates/best.py`. Replaced explicit attention with `scaled_dot_product_attention`, preserving all correctness invariants. Correctness verified locally (CPU) on dev shapes + `official-safe` with/without padding. GPU speedup pending first cluster run. (agent: seed)

## Seeded by human

_Humans: drop ideas here as free text — an agent will formalize each into a
proper hypothesis + candidate and treat it as high priority. Example:_

- _(example) "Try flash-attn v3 varlen for shape #14 instead of SDPA."_
