# AGENTS.md — How to join the autoresearch swarm

This repo *is* the shared brain. Multiple agents (Claude Code Workflow
instances, teammates' machines, or a human running an agent) collaborate by
reading and appending to git-tracked state. Follow this protocol.

## 0. One-time setup

- Clone the repo on a machine that can submit jobs to the A100/H100 Slurm
  cluster (or on the GPU login node itself).
- Fill in `cluster.config.json` (partition, account, gres, module load, ssh
  host / remote workdir). See that file.
- Smoke test: `python3 runner.py --candidates candidates/best.py --shapes official-safe`
  should return JSON with `correctness_passed: true` and real speedups.
- Pick a short **agent-id** (e.g. `opus-1`, `sonnet-a`, `alice`). All your
  candidate files are namespaced `candidates/<agent-id>-stepN.py` so filenames
  never collide.

## 1. The per-turn loop

Each turn does exactly one experiment:

1. **Sync in:** `git pull --rebase`.
2. **Read state:** `PROGRAM.md` (playbook + correctness contract),
   `leaderboard.md` (current best), `TODO.md` (open ideas — human-seeded first).
3. **Claim work:** pick one `Open` TODO item (or turn a human free-text idea into
   a proper one). Move it to `In progress` with your agent-id + a one-line intent.
   **Commit + push immediately** (`git commit -am "claim: <id> by <agent-id>"`,
   `git pull --rebase`, `git push`). If the push rebase-conflicts on that claim,
   someone beat you — pick another item.
4. **Implement:** branch conceptually from `candidates/best.py`; write your
   candidate to `candidates/<agent-id>-stepN.py`. Preserve every invariant in the
   PROGRAM.md correctness contract.
5. **Evaluate on the cluster:**
   `python3 runner.py --candidates candidates/<agent-id>-stepN.py --shapes official-safe`
   (use `--shapes all` for a finalist; `--shapes dev --mode local --device cpu`
   for a quick laptop sanity check). Read the returned JSON.
6. **Record:** append one line to `journal.jsonl` (schema below) and a short
   human-readable entry to `LOG.md`. If your candidate is **correct** AND beats
   the current best median speedup, do the **guarded best update** (§2).
7. **Close out:** mark the TODO item `Done` with the result; optionally append
   follow-up ideas under `Open`.
8. **Sync out:** `git pull --rebase` then `git push`. Repeat from step 1.

## 2. Guarded best update (avoid races)

Only replace `candidates/best.py` + `leaderboard.md` when you are strictly
better *after* pulling the latest:

1. `git pull --rebase`.
2. Re-read `leaderboard.md`'s current best median speedup.
3. If your correct candidate's median speedup is still strictly higher, copy your
   file to `candidates/best.py`, update `leaderboard.md`, commit, `pull --rebase`,
   push. If the push conflicts, repeat from step 1.
4. If someone pushed a better best meanwhile, you are not best — just leave your
   candidate recorded in the journal and move on.

## 3. Merge discipline (why this rarely conflicts)

- `journal.jsonl`, `LOG.md` — **append-only**; git merges unions cleanly. On the
  rare conflict, keep **both** sides' new lines.
- `candidates/<agent-id>-stepN.py` — agent-id-namespaced; never collide.
- `TODO.md` — a claim board; claims are small append/move commits pushed
  atomically. Conflict = someone else moved it; re-pull and pick another.
- `candidates/best.py` + `leaderboard.md` — only via the guarded update (§2).

## 4. `journal.jsonl` schema (one JSON object per line)

```json
{"iter": 12, "round": 3, "agent_id": "opus-1", "node_id": "opus-1-step5",
 "parent_id": "best", "direction": "compile",
 "hypothesis": "torch.compile(max-autotune) fuses LN/GELU/linears; expect launch-overhead-bound small-batch shapes (#2,#3) to gain most",
 "todo_ref": "T2",
 "diff": "candidates/opus-1-step5.py",
 "eval": {"correctness_passed": true, "median_speedup": 1.9, "geomean_speedup": 1.8,
          "per_shape": [{"shape_id": 1, "passed": true, "max_abs": 1e-6, "max_rel": 0.5,
                          "baseline_ms": 2.1, "opt_ms": 1.1, "speedup": 1.9}]},
 "decision": "new-best",
 "recovery": null,
 "cost": {"tokens": null, "wall_s": 41.0, "slurm_job": "12345"}}
```

`decision` ∈ `keep | prune | new-best`. `recovery` describes how a failure
(compile error, OOM, timeout) was handled, or `null`. `cost.tokens` is optional
(the Workflow fills it).

## 5. For LLM-driven agents (the Workflow): model/effort tiering

To respect usage limits: use a **strong** model + higher reasoning effort for the
**Strategist** (choose directions) and **Postmortem** (diagnose + decide) stages,
and a **cheap** model + low effort for the **Coder** (write a variant) and
mechanical **runner/sync** stages. See `autoresearch.workflow.js`.

## 6. For humans: seeding ideas

Add ideas under `TODO.md` → **"## Seeded by human"**. Free text is fine — an
agent will formalize it into a proper hypothesis + candidate. Human-seeded items
are treated as **high priority** by the Strategist.
