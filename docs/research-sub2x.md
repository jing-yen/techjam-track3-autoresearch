# Why four shapes fall below 2x, and what can be done

Research brief · 2026-08-31 · A100-80 (xgph1), fp32, official timing protocol
Subject: `candidates/v_router.py` at 2.27x median / 2.47x geomean, journal iter 9

**Scope note.** This is an engineering brief, not a literature review. Claims are
grounded in (a) measurements in this repository's ledger, (b) line-cited source in
this repository, and (c) the NVIDIA A100 datasheet. Where a claim is inference
rather than measurement it is labelled INFERRED. Where it is untested it is
labelled UNVERIFIED.

## Method

For each shape, arithmetic throughput was computed from the appendix FLOP model
(all four layers; causal halving applied to both attention matmuls) divided by
the measured optimized wall-clock, then compared against the card's fp32 ceiling.
Attention parallelism was taken as `batch x heads`, one program per `(b,h)` pair,
against the A100's 108 SMs.

Ceilings from the vendor datasheet: **FP32 19.5 TFLOPS**, **TF32 Tensor Core
156 TFLOPS**, 108 SMs
([NVIDIA A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf);
[Ampere architecture whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)).

| # | speedup | opt ms | GFLOP | TFLOP/s | % of 19.5 | attn blocks |
|--|--|--|--|--|--|--|
| 8 | 1.14x | 26.284 | 420.9 | **16.0** | **82%** | 256 |
| 9 | 1.47x | 1.345 | 7.5 | 5.6 | 29% | 64 |
| 10 | 1.70x | 1.368 | 7.5 | 5.5 | 28% | 128 |
| 5 | 1.86x | 2.495 | 15.0 | 6.0 | 31% | 512 |
| *1 (ref)* | *2.02x* | *1.300* | *7.5* | *5.8* | *30%* | *256* |
| *11 (ref)* | *2.73x* | *1.858* | *7.5* | *4.0* | *21%* | *1024* |

**Finding 0: the four are not one problem.** They have three distinct causes, and
only one of them is worth GPU time.

---

## #8 — at the arithmetic ceiling, not inefficient

420.9 GFLOP in 26.284 ms is **16.0 TFLOP/s against a 19.5 TFLOP/s fp32 peak —
82% of theoretical**. For a real workload mixing GEMM, LayerNorm, GELU and
attention, that is close to as good as fp32 gets on this card.

**No kernel-level optimization can recover much here.** Fusion, layout, and
tiling all address overhead, and there is at most 18% of overhead present. This
also explains why `torch.compile` gains almost nothing on #8: there is little to
fuse away.

**But the precision is a self-imposed constraint, and it is the real lever.**

`candidates/v_router.py:39-42` executes at **module import, outside any
function**:

```python
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
```

These are process-global flags. `bench_harness.py:313-314` sets its own value
first (default `True`), then imports the candidate, whose import overrides it.
**The baseline is therefore de-TF32'd too.** The comparison stays internally
fair, but both sides run at ~1/8 of the card's TF32 throughput.

Two consequences pull in opposite directions.

**Integrity.** The organizer's script defaults to TF32 **on**
(`torch_transformer_benchmark.py:687`, `--matmul-precision high` at `:638`). Our
2.47x is measured in a configuration the organizer does not default to, in which
the reference is handicapped on exactly the shapes where it would otherwise gain
most. INFERRED: on GEMM-heavy shapes this plausibly *inflates* our ratio. This
must be disclosed in the tech report regardless of what we do next.

**Opportunity.** #8 has roughly 8x of theoretical headroom behind that flag.

**Why the flag was set** (`8010964`): Inductor's `max-autotune` selected TF32 GEMM
kernels for the *candidate* while the baseline used cuBLAS, drifting ~0.005
against `atol=0.002` on 9/12 shapes.

**Contradiction disclosure.** That is evidence of *asymmetry between two TF32
implementations*, not evidence that TF32 fails the gate. The gate is an
elementwise OR (`torch_transformer_benchmark.py:314-316`): a 0.005 drift on a
value of magnitude 1.0 passes on the relative arm. The failures concentrate where
`|ref|` is small — plausible for post-LayerNorm activations — and would apply to
*any* two differing fp32 reduction orders, TF32 or not.

**The routing makes this sharper. #8 routes to `fused`, which is never compiled.**
The Inductor autotuner cannot act on that path at all. A global pin aimed at a
compile-path bug is being applied to a shape that never touches the compiler.

**Recommended test (S1), one sweep:** remove the import-time override, let the
harness default stand, measure correctness and speed on all 12. Then scope the
override to the compiled path only.
**Falsifier:** TF32-on fails the gate on the `fused` path for #8 -> the pin is
justified as written; record and close.
**Honest expectation:** #8's absolute time should fall sharply. The **ratio may
fall, rise, or hold**, because the baseline accelerates too. Report what is
measured. Do not select the configuration that yields the larger number without
saying so.

---

## #9 and #10 — the reference is unusually good here

Shapes 1, 9, 10 and 11 are **identical arithmetic** (7.52 GFLOP); only head count
differs. Our candidate is nearly flat across them. The baseline is not:

| # | heads | head_dim | baseline ms | ours ms | speedup |
|--|--|--|--|--|--|
| 9 | 1 | 128 | **1.974** | 1.345 | 1.47x |
| 10 | 2 | 64 | 2.332 | 1.368 | 1.70x |
| 1 | 4 | 32 | 2.623 | 1.300 | 2.02x |
| 11 | 16 | 8 | **5.074** | 1.858 | 2.73x |

The baseline degrades **2.6x** from H=1 to H=16 on identical FLOPs, because it
materializes `[B,H,S,S]` and performs more transpose/reshape work as heads
increase (`torch_transformer_benchmark.py:77-83, :97, :113-117`). Our candidate
degrades only 1.4x.

**Therefore the speedup spread across the head sweep is a property of the
reference, not of our kernel.** #9's low ratio is the baseline being efficient,
not us being slow. Reporting it as a weakness would misdescribe the data.

INFERRED secondary factor: at H=1, `batch x heads` = 64 attention programs
against 108 SMs, so attention is under-occupied. But attention is roughly a
quarter of this layer's work at `d=128`, so this bounds the available gain to
single-digit percent. UNVERIFIED — no occupancy profile has been taken.

**Genuine, cheap lever (S2):** #9 and #10 currently route to `fused`, while #1 —
same arithmetic — routes to `compile` and beats them. The routing table was built
from a sweep taken before B10's sync was identified and before
`v_compile_reduce.py` existed. The head-count shapes may simply be mis-routed.
They ride along free in any sweep.
**Falsifier:** compile does not beat fused on #9/#10 -> routing is already
optimal, close it.

---

## #5 — no headroom; the falling ratio is a scaling artifact

#5 is exactly twice #1's work (B=64 -> 128, all else equal).

```
ours:      1.300 -> 2.495 ms   = 1.92x   (near-linear)
baseline:  2.623 -> 4.653 ms   = 1.77x   (sub-linear)
```

Our candidate scales near-linearly because it is already efficient — doubling the
work doubles the time. The baseline scales *sub*-linearly because at B=64 it was
still partly overhead-padded, and B=128 amortizes that overhead. Two well-behaved
curves with different slopes necessarily produce a falling ratio.

**#5 is not underperforming.** At 2.495 ms it is marginally *better* than twice
#1's 1.300 ms. The ratio fell because the reference improved, not because we
regressed. **Recommendation: document, do not optimize.**
**Falsifier:** a variant beats 1.86x on #5 without regressing #1 -> this
reasoning is wrong.

---

## Ranked recommendations

| rank | action | cost | expected | risk |
|--|--|--|--|--|
| 1 | **S1 — resolve TF32** | 1 sweep | #8 absolute time falls sharply; ratio uncertain | High: may *lower* the headline number. Must be reported either way. |
| 2 | **B10 — remove the `.all()` sync** | small edit + the same sweep | low single-digit % on the small shapes | Low |
| 3 | **T1 — benchmark `v_compile_reduce.py`** (written, never run) | same sweep | targets the already-5x launch-bound shapes | Low |
| 4 | **S2 — re-route #9/#10 to compile** | rides along | modest | Low |
| 5 | **S3 — document #5** | prose | none | None |

Items 1-4 fit in **one cluster job**. That is the entire remaining optimization
budget worth spending; everything after it should go to the §3.5 deliverables.

## Limitations

1. Throughput percentages use a hand-derived FLOP model, not a profiler counter.
   INFERRED, not measured. A `ncu` roofline would settle #8 definitively; none
   has been run.
2. The occupancy explanation for #9/#10 is INFERRED from `batch x heads` versus
   SM count. No occupancy or stall-reason profile exists.
3. All measurements are single-run medians from one node (xgph1). No
   between-node variance has been characterized.
4. Shape #6 (B=10000) is excluded from every sweep and therefore from this
   analysis. Shape #14 is documented as infeasible (journal iter 7).
5. The 156 TFLOPS TF32 figure is a theoretical peak. Real GEMMs at these sizes
   typically reach a substantial fraction of it, not all of it; the "8x headroom"
   figure is an upper bound, not a prediction.

## AI disclosure

This brief was produced with AI assistance (Claude Opus 5). Measurements are from
`journal.jsonl` and `leaderboard.md`; source citations were verified against the
files in this repository at the line numbers given; hardware figures are from the
cited NVIDIA datasheet.
