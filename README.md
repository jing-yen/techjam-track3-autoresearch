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

**Requirements:** Linux host with an NVIDIA GPU (we used `<FILL: GPU model>`),
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
python runner.py --candidates candidates/best.py --shapes all --dtype <FILL: dtype>

# 5. Reproduce a single shape through the organizer's own script, unmodified,
#    as an independent check that our harness agrees with it. Example, shape 1:
python torch_transformer_benchmark.py \
  --batch-size 64 --seq-len 128 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --dtype <FILL: dtype>
```

Step 4 writes per-shape correctness and speedup as JSON and is what populates
`leaderboard.md`. Step 5 exists because a harness that disagrees with the
organizer's script is worthless; they must match.

**Always pass every shape parameter explicitly.** The organizer script's defaults
(`B=8, S=128, D=512, H=8, FFN=2048, 6 layers, non-causal, fp32`) match none of the
14 official shapes.

## Results

Measured on `<FILL: GPU>`, `<FILL: dtype>`, torch `<FILL>`. Full environment in
`docs/environment.txt`. Raw data in `journal.jsonl`.

| # | B | S | d | H | L | passed | baseline ms | ours ms | speedup |
|--|--|--|--|--|--|--|--|--|--|
| 1 | 64 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 2 | 1 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 3 | 4 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 4 | 16 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 5 | 128 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 6 | 10000 | 128 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 7 | 64 | 128 | 32 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 8 | 64 | 128 | 1024 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 9 | 64 | 128 | 128 | 1 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 10 | 64 | 128 | 128 | 2 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 11 | 64 | 128 | 128 | 16 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 12 | 64 | 32 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 13 | 64 | 1024 | 128 | 4 | 4 | `<FILL>` | `<FILL>` | `<FILL>` | `<FILL>` |
| 14 | 32 | 100000 | 1024 | 16 | 2 | see limitations | — | — | — |

Median speedup `<FILL>`, geomean `<FILL>`, across `<FILL>` shapes that produced a
reference.

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

`<FILL: if B1 is still unfixed at submission, say so here — the seed candidate's`
`is_causal fast path is unreachable because valid_token_mask is never None, which`
`costs the FlashAttention backend on every shape. See TODO.md item B1.>`

**What we would do next, in order:** confirm which SDPA backend actually fires per
shape and per dtype; land CUDA-graph capture for the launch-overhead-bound small
shapes (#2, #3, #12); build a genuine per-shape dispatch table now that we know
head_dim 256 on shape 8 exceeds the sm80 FlashAttention limit; and only then write
Triton kernels, and only where a profile shows something left.

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
