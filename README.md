# TikTok TechJam 2026 · Track 3 — Autoresearch Swarm for a Transformer GPU Kernel

> **Status:** measurements pending first GPU run. Every number in this README and
> in `TECH_REPORT.md` marked `<FILL>` is a placeholder. **Do not submit with any
> `<FILL>` remaining.** Run `scripts/check_placeholders.sh` before you ship.

## Project overview

Track 3 asks for a faster GPU implementation of a fixed pre-LayerNorm Transformer
block that stays numerically correct against the organizer's reference across 14
published input shapes.

We did not hand-optimize one kernel. We built **an autonomous multi-agent research
loop that proposes, measures, and prunes kernel variants against a hard
correctness gate**, and let it search the space. The problem statement asks
participants to "use AI-assisted methods" and awards bonus points for a report on
"the AI skills/tools used"; this repo is that method made reproducible rather than
anecdotal.

Three things make it work:

1. **A scoreboard nobody can argue with.** `bench_harness.py` reuses the
   organizer's own `compare_outputs`, `generate_random_case` and `benchmark_once`,
   so our numbers come from the same code path the task defines. Every experiment
   appends one row to `journal.jsonl`. No task is done until a ledger row exists.
2. **A hard gate before speed is ever discussed.** Per element:
   `abs(opt-ref) <= 0.002` **OR** `abs(opt-ref) <= 0.02*abs(ref)`. Every element of
   every runnable shape. A failing candidate is never timed.
3. **Two cooperating agent layers.** A *research layer* (plan → adversarial review
   → reconcile, across two model families) that decides what is worth GPU time,
   and an *implementation layer* that turns one hypothesis into one measured
   candidate. They communicate only through git: `TODO.md` one way,
   `journal.jsonl` the other.

## What's here

| file | purpose |
|--|--|
| `AGENTS.md` | The protocol any agent follows to join the swarm. |
| `PROGRAM.md` | Optimization playbook + the correctness contract candidates must preserve. |
| `TODO.md` | Research queue. Every item carries `file:line` evidence and a falsification gate. |
| `LOG.md` / `journal.jsonl` | Human- and machine-readable experiment log. |
| `leaderboard.md` | Current best correct candidate + per-shape speedups. |
| `candidates/best.py` | Current best implementation. |
| `bench_harness.py` | Reuses the organizer benchmark; emits per-shape correctness + speedup as JSON. |
| `runner.py` + `sbatch_template.sh` | Evaluate candidates on the cluster (Slurm array) or locally. |
| `research-loop.sh` + `prompts/` + `schemas/` | The research layer: Claude plan → Codex review → Claude reconcile. |
| `autoresearch.workflow.js` | The implementation layer's agent loop. |
| `scripts/capture_env.sh` | Dumps CPU/GPU/disk/versions for the tech report. |
| `tests/` | Unit tests for the harness + runner (CPU only). |

## Setup and installation

**Requirements:** Linux host with an NVIDIA GPU (we used an **NVIDIA A100-80 PCIe**),
Python 3.11+, a CUDA-enabled PyTorch. The dev laptops are Apple Silicon and can
run only the CPU correctness tests.

```bash
git clone https://github.com/jing-yen/techjam-track3-autoresearch.git
cd techjam-track3-autoresearch

conda create -n techjam python=3.11 -y && conda activate techjam
pip install torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # expect True
```

For Slurm clusters, fill in `cluster.config.json` (ssh host, gres, module load,
remote workdir) — see `CLUSTER_SETUP.md`.

## Steps to reproduce our results

```bash
# 1. Plumbing check (CPU, no GPU needed). Expect 9/9 pass.
python tests/test_bench_harness.py && python tests/test_runner.py

# 2. Correctness smoke test on tiny shapes (CPU).
python bench_harness.py --candidate candidates/best.py --shapes dev --device cpu

# 3. Capture the environment for the report. Run ON the GPU node.
bash scripts/capture_env.sh > docs/environment.txt

# 4. The headline result: all 14 official shapes on GPU.
python runner.py --candidates candidates/best.py --shapes all --dtype float32

# 5. Reproduce a single shape through the organizer's own script, unmodified,
#    as an independent check that our harness agrees with it. Example, shape 1:
python torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --dtype float32
```

Step 4 writes per-shape correctness and speedup as JSON and is what populates
`leaderboard.md`. Step 5 exists because a harness that disagrees with the
organizer's script is worthless; they must match.

**Always pass every shape parameter explicitly.** The organizer script's defaults
(`B=8, S=128, D=512, H=8, FFN=2048, 6 layers, non-causal, fp32`) match none of the
14 official shapes.

## Results

Measured on **NVIDIA A100-80 PCIe** (NUS SoC cluster, node xgph1), **float32**,
official timing protocol (warmup 20, repeats 100, rounds 3, alternating order).
Candidate: `candidates/v_router.py`, journal iter 9. Full environment in
`docs/environment.txt`. Raw data in `journal.jsonl`.

| # | B | S | d | H | passed | baseline ms | ours ms | speedup | routed to |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 64 | 128 | 128 | 4 | ✅ | 2.623 | 1.300 | **2.02x** | compile |
| 2 | 1 | 128 | 128 | 4 | ✅ | 1.895 | 0.374 | **5.07x** | compile |
| 3 | 4 | 128 | 128 | 4 | ✅ | 1.900 | 0.447 | **4.25x** | compile |
| 4 | 16 | 128 | 128 | 4 | ✅ | 1.863 | 0.859 | **2.17x** | best |
| 5 | 128 | 128 | 128 | 4 | ✅ | 4.653 | 2.495 | **1.86x** | fused |
| 6 | 10000 | 128 | 128 | 4 | not run | — | — | — | excluded (see limitations) |
| 7 | 64 | 128 | 32 | 4 | ✅ | 1.893 | 0.527 | **3.59x** | compile |
| 8 | 64 | 128 | 1024 | 4 | ✅ | 29.936 | 26.284 | **1.14x** | fused |
| 9 | 64 | 128 | 128 | 1 | ✅ | 1.974 | 1.345 | **1.47x** | fused |
| 10 | 64 | 128 | 128 | 2 | ✅ | 2.332 | 1.368 | **1.70x** | fused |
| 11 | 64 | 128 | 128 | 16 | ✅ | 5.074 | 1.858 | **2.73x** | fused |
| 12 | 64 | 32 | 128 | 4 | ✅ | 1.874 | 0.793 | **2.36x** | fused |
| 13 | 64 | 1024 | 128 | 4 | ✅ | 62.067 | 14.031 | **4.42x** | fused |
| 14 | 32 | 100000 | 1024 | 16 | OOM | — | — | — | see limitations |

**Median speedup 2.27x, geometric mean 2.47x**, across the 12 shapes that produced a
reference. Sum-of-wall-clock across those 12: 117.9 ms -> 55.4 ms (2.13x).
All 12 pass the correctness gate, max_abs ~1e-6 — four orders of magnitude
under the 0.002 tolerance.

## Limitations, and what we would improve given more time

Written honestly. Several of these are things we found and did not have time to
fix; they are recorded here rather than hidden.

**Shape 14 does not run, and we can quantify exactly why.** At
`B=32, S=100000, d=1024`, one `[B,S,D]` activation is 12.2 GB in fp32 (6.1 GB in
fp16). A forward pass holds roughly seven such tensors live, so the floor is about
**85 GB fp32 / 43 GB fp16** before any attention workspace. The reference is worse
still: it materializes the full `[B,H,S,S]` score matrix
(`torch_transformer_benchmark.py:97`), which is 18.6 TB in fp32 — 37 GB for a
*single* `(batch, head)` slice, of which there are 512. **The organizer's own
reference therefore cannot produce ground truth for shape 14 on any single GPU**,
so there is nothing to check a candidate against even if it ran. Given more time
we would implement sequence-chunked streaming attention and validate it against a
chunked oracle we write ourselves, while stating plainly that this is a different
benchmark from the other thirteen.

**Our timing is steady-state, not cold-start.** We inherit the organizer's
protocol, which reuses one fixed input for all iterations and never flushes L2
(`torch_transformer_benchmark.py:529-536`). Caches stay hot and CUDA events are
recorded back-to-back with no per-iteration sync, so what we report is pipelined
throughput. This flatters any implementation whose working set fits in cache. A
development timer with an L2 flush would give different, likely lower, numbers.

**The correctness gate is self-administered.** Section 3.2 of the problem
statement has participants run on their own machines, so no external party
re-executes our benchmark. We mitigate this by reusing the organizer's own
comparison function verbatim rather than reimplementing its tolerance logic, and
by reproducing at least one shape through the unmodified organizer script
(reproduction step 5).

**Only the PyTorch path is implemented.** The problem statement allows either
framework; we chose torch and did not touch
`tensorflow_transformer_benchmark.py`.

**We disabled TF32, which deviates from the organizer's default, and it matters.**
`candidates/v_router.py:39-42` sets `allow_tf32 = False` and
`float32_matmul_precision("highest")` at module import. These are process-global
PyTorch flags, and the harness sets its own value before importing the candidate,
so **the baseline is de-TF32'd too**. The organizer's script defaults TF32 **on**
(`torch_transformer_benchmark.py:687`). Our comparison is therefore internally
fair but measured in a non-default configuration in which both sides run well
below the card's tensor-core throughput. We did this because `torch.compile`'s
autotuner selected TF32 GEMM kernels for the candidate while the baseline used
cuBLAS, drifting ~0.005 against the 0.002 absolute tolerance on 9 of 12 shapes.
Given more time we would scope the pin to the compiled path only and re-measure
both configurations, reporting both.

**Shape 8 is at the fp32 arithmetic ceiling, and it is half our remaining
runtime.** 420.9 GFLOP in 26.284 ms is 16.0 TFLOP/s against the A100's 19.5
TFLOPS fp32 peak — **82% of theoretical**. It is also 47% of our total optimized
wall clock across the 12 shapes. Its 1.14x is not inefficiency; it is the honest
limit of fp32 on this card. The remaining lever is precision, not kernel work.

**FlashAttention never runs in our configuration.** A direct probe
(`tools/probe_sdpa_backends.py`, `docs/sdpa_backend_probe.json`) forcing each
SDPA backend per shape found flash eligible on **0 of 14 shapes at fp32** and
memory-efficient attention on 14 of 14. Every speedup we report comes from
memory-efficient SDPA. At fp16 the same probe finds flash eligible on **all 14**,
including the head_dim-256 shape — so fp16 would unlock both tensor cores and
flash. A naive blanket fp16 cast failed correctness on 11 of 12 shapes
(max_abs 0.006-0.009); an autocast variant that keeps LayerNorm and the softmax
reduction in fp32 (`candidates/v_amp.py`) is written but was not validated on GPU
before the deadline.

**Shape 6 (batch 10000) was never run.** It is excluded from our `official-safe`
sweep alongside shape 14 on memory grounds and we did not return to it.

**What we would do next, in order:** re-measure with TF32 enabled symmetrically
and report both configurations; validate the autocast fp16 path on GPU, which the
backend probe shows would unlock flash on all 14 shapes; remove the per-forward
`.all()` device sync that our padding-detection fix introduced; and only then
consider Triton kernels, which our own profiling suggests would gain little —
shape 8 is already at 82% of the fp32 ceiling and the small shapes are dominated
by launch overhead that `torch.compile` already removes.

## Team member contributions

| member | layer | contribution |
|--|--|--|
| `<FILL: name>` | Research | Problem-statement analysis, benchmark-script audit, research queue (`TODO.md`), plan/review/reconcile loop, tech report. |
| `<FILL: name>` | Implementation | Candidate kernels, `bench_harness.py`, `runner.py`, cluster integration. |
| `<FILL: name>` | `<FILL>` | `<FILL>` |
| `<FILL: name>` | `<FILL>` | `<FILL>` |
| `<FILL: name>` | `<FILL>` | `<FILL>` |

## Correctness gate

Per element: `abs(opt-ref) <= 0.002` **OR** `abs(opt-ref) <= 0.02*abs(ref)`
(problem statement §3.2). Every element of every runnable shape must pass. NaN,
Inf and shape mismatches fail outright. Speedup is only meaningful for a correct
candidate, and `bench_harness.py` refuses to report one otherwise. See
`PROGRAM.md` for the full list of invariants a candidate must preserve.
