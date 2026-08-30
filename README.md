# TikTok TechJam 2026 · Track 3 — Autoresearch Swarm for a Transformer GPU Kernel

An autonomous, **GitHub-coordinated multi-agent** system that optimizes the
Transformer layer in the organizer's `torch_transformer_benchmark.py` — proposing
kernel changes, running them on an A100/H100 Slurm cluster, keeping/pruning on
measured speedup (gated by correctness), and logging every step to a shared,
self-documenting notebook.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and
[Weco/AIDE](https://github.com/WecoAI/weco-cli): agents edit code, run
experiments, and accept/revert on a measured score — here that score is
**median speedup subject to a hard correctness gate**, and many agents cooperate
through git.

## What's here

| file | purpose |
|--|--|
| `AGENTS.md` | **Start here.** The protocol any agent follows to join the swarm. |
| `PROGRAM.md` | Optimization playbook + the correctness contract candidates must preserve. |
| `TODO.md` | Idea backlog + claim board. Humans seed ideas here. |
| `LOG.md` / `journal.jsonl` | Human- and machine-readable experiment log (the shared notebook). |
| `leaderboard.md` | Current best correct candidate + per-shape speedups. |
| `candidates/best.py` | Current best implementation (starts as the SDPA seed). |
| `bench_harness.py` | Reuses the organizer benchmark; emits per-shape correctness + speedup as JSON. |
| `runner.py` + `sbatch_template.sh` | Evaluate candidates on the cluster (Slurm array) or locally. |
| `autoresearch.workflow.js` | The Claude Code Workflow that drives one agent instance (model/effort tiered). |
| `cluster.config.json` | Your cluster's Slurm settings (fill the placeholders). |
| `tests/` | Unit tests for the harness + runner (run on CPU). |

## Quickstart

```bash
# 1. Laptop sanity check (CPU, no cluster):
python tests/test_bench_harness.py && python tests/test_runner.py
python bench_harness.py --candidate candidates/best.py --shapes dev --device cpu

# 2. Configure the cluster (SoC A100/H100 via Slurm) — see CLUSTER_SETUP.md:
$EDITOR cluster.config.json    # ssh host (xlogin), gres, module load, remote workdir

# 3. Evaluate the seed on real GPUs:
python runner.py --candidates candidates/best.py --shapes all

# 4. Run the swarm (from Claude Code): launch autoresearch.workflow.js via the
#    Workflow tool. Multiple instances + humans can run concurrently — see AGENTS.md.
```

## Correctness gate

Per element: `abs(opt-ref) <= 0.002` **OR** `abs(opt-ref) <= 0.02*abs(ref)`; every
element of every runnable shape must pass. Speedup is only meaningful for correct
candidates. See `PROGRAM.md`.
