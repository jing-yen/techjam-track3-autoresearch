# PROGRAM.md — Optimization Playbook & Correctness Contract

> This is the swarm's shared "priors" file (à la Karpathy's `program.md`). Every
> agent reads it before proposing a change. Humans and agents **extend** it by
> appending sections. It feeds `TODO.md`.

## The task, in one paragraph

Fill in `UserOptimizedTransformer.forward()` so the Transformer runs as fast as
possible on the target GPU (A100/H100) while staying numerically correct vs the
organizer's `BaselineTransformer`. Correctness is checked **per element** across
14 fixed shapes: an element passes iff `abs(opt-ref) <= 0.002` **OR**
`abs(opt-ref) <= 0.02*abs(ref)`. **Every** element of **every** runnable shape
must pass, or the candidate is disqualified (speedup is not even measured by the
organizer on a failing candidate). We maximize **median speedup** over the
shapes, subject to that hard gate.

## Correctness contract — invariants a candidate MUST preserve

Break any of these and correctness fails. The seed (`candidates/best.py`) honors
all of them; keep them when you branch.

1. **Exact GELU.** Use `F.gelu(..., approximate="none")` (erf form). `"tanh"` will
   fail the gate.
2. **Numerically stable softmax.** The baseline computes softmax in fp32 then
   casts back. In fp32 this is automatic; in fp16/bf16 you must ensure the
   attention reduction accumulates in fp32 (SDPA's flash/mem-efficient kernels do).
3. **Padded-query output = 0.** When `valid_token_mask` is given, zero the output
   rows of padded (invalid) query positions — after attention (`out_proj`), after
   each block, and after the final norm. The baseline does all three.
4. **Key padding.** Invalid **key** positions must not contribute to attention
   (mask them to `-inf` before softmax, or via an additive `-inf` attn bias).
5. **Causal mask.** Upper triangle strictly above the diagonal is masked:
   `triu(diagonal=1)`. Prefer SDPA `is_causal=True` when there is no padding so no
   `[S,S]` mask is materialized (this is what makes shape #14 feasible).
6. **Scale.** `1/sqrt(head_dim)`. (SDPA's default — do not pass a custom scale.)
7. **Shape.** Return `[batch, seq_len, d_model]`.
8. **Weights.** `copy_model_weights(baseline, optimized, strict=True)` must line
   up. Keep the baseline parameter names
   (`layers.{i}.norm1/norm2`, `layers.{i}.attention.{q,k,v,out}_proj`,
   `layers.{i}.ffn_in/ffn_out`, `final_norm`). If you fuse/rename params, set
   `STRICT_WEIGHT_COPY = False` in the candidate module **and** provide a
   module-level `copy_model_weights(baseline, optimized)` that maps them.

## Candidate module contract

A candidate is a `.py` file that defines:

```python
from torch_transformer_benchmark import BaselineTransformer
class UserOptimizedTransformer(BaselineTransformer):
    def forward(self, x, valid_token_mask=None): ...
```

Optional module-level knobs: `STRICT_WEIGHT_COPY` (bool), and a custom
`copy_model_weights(baseline, optimized)`. See `candidates/best.py` for the
reference structure that keeps strict weight copy working.

## Optimization playbook — directions ranked by expected A100/H100 impact

1. **SDPA (done in seed).** `F.scaled_dot_product_attention` — fused softmax, no
   `[B,H,S,S]` materialization, flash/mem-efficient backends. Biggest single win
   and the only path that survives shape #14.
2. **`torch.compile`.** Wrap the model (or forward) with `torch.compile(mode=
   "max-autotune"|"reduce-overhead")` to fuse LayerNorm/residual/GELU/linears and
   cut launch overhead. Watch: recompiles per new shape (14 shapes → warm each);
   guards must not break correctness.
3. **dtype / TF32.** For fp32 inputs, TF32 matmuls already apply to both sides.
   For fp16/bf16 test cases, cast carefully and verify the gate (this is the
   risky path — keep the fp32 softmax reduction).
4. **Memory-layout cleanups.** Remove needless `.contiguous()`/transposes feeding
   SDPA's `[B,H,S,D]`; ensure QKV projections produce SDPA-friendly strides.
5. **Fused QKV projection.** One `Linear(d_model, 3*d_model)` instead of three.
   Faster, but breaks strict weight copy → provide the mapping (split the fused
   weight/bias into q/k/v on load).
6. **Per-shape specialization.** The rules allow shape checks → different code
   paths per shape. E.g. tiny `seq=32/128` → eager or compiled small path;
   `seq=1024/100000` → mem-efficient SDPA; `d_model=1024` → different tiling;
   huge `batch=10000` → watch memory. A real late-stage lever.
7. **Custom Triton kernels.** Fused LN+residual, fused attention+bias, fused
   FFN(GELU). Highest effort/risk; reserve until SDPA+compile are banked and only
   where profiling shows a remaining bottleneck.

## The 14 official shapes (see Appendix "Test Shapes")

All causal. `d_model` = "QKV Dim". Ids 6 and 14 are memory-extreme.

| # | batch | seq | d_model | heads | layers | ffn | note |
|--|--|--|--|--|--|--|--|
| 1 | 64 | 128 | 128 | 4 | 4 | 128 | base |
| 2 | 1 | 128 | 128 | 4 | 4 | 128 | tiny batch → launch-overhead bound |
| 3 | 4 | 128 | 128 | 4 | 4 | 128 | |
| 4 | 16 | 128 | 128 | 4 | 4 | 128 | |
| 5 | 128 | 128 | 128 | 4 | 4 | 128 | |
| 6 | 10000 | 128 | 128 | 4 | 4 | 128 | **huge batch** — memory/throughput |
| 7 | 64 | 128 | 32 | 4 | 4 | 32 | small dims |
| 8 | 64 | 128 | 1024 | 4 | 4 | 1024 | large dims → matmul bound |
| 9 | 64 | 128 | 128 | 1 | 4 | 128 | 1 head |
| 10 | 64 | 128 | 128 | 2 | 4 | 128 | 2 heads |
| 11 | 64 | 128 | 128 | 16 | 4 | 128 | many heads |
| 12 | 64 | 32 | 128 | 4 | 4 | 128 | short seq |
| 13 | 64 | 1024 | 128 | 4 | 4 | 128 | long seq → attention bound |
| 14 | 32 | 100000 | 1024 | 16 | 2 | 1024 | **seq=100k** — baseline OOMs (~40GB/head); SDPA-only |

**Shape #14 note:** the baseline cannot produce a reference (OOM), so the harness
marks it `baseline_oom`, skips its correctness gate, and still times the
candidate. "Runs at all" is a reportable win; don't let it block the sweep.

## How progress is measured

The harness (`bench_harness.py`) emits per-shape `passed`, `max_abs`, `max_rel`,
`baseline_ms`, `opt_ms`, `speedup`, and an aggregate `correctness_passed` +
`median_speedup` + `geomean_speedup`. The swarm keeps the correct candidate with
the best median speedup as `candidates/best.py`.
