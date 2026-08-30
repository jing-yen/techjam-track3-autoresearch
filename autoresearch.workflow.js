export const meta = {
  name: 'track3-autoresearch',
  description: 'Autonomous multi-agent loop that optimizes the Transformer GPU kernel (TikTok TechJam Track 3), coordinating through the git repo',
  whenToUse: 'Run one agent instance of the autoresearch swarm: propose kernel changes, evaluate on the cluster, keep/prune on measured speedup, log to the shared notebook.',
  phases: [
    { title: 'Round', detail: 'sync-in -> strategise -> code (parallel) -> evaluate -> postmortem, repeated until budget/convergence', model: 'opus + sonnet + haiku (tiered)' },
    { title: 'Finalize', detail: 'draft the tech report from journal.jsonl + LOG.md' },
  ],
}

// ---- config (override via the Workflow `args` object) --------------------- //
const DEFAULTS = {
  agentId: 'opus-1',
  repoDir: '/Users/jingyen/Downloads/TikTok TechJam/track3-autoresearch',
  rounds: 3,            // max rounds when no token budget is set
  nVariants: 3,         // candidate variants proposed per round
  convergenceK: 2,      // stop after this many consecutive rounds with no new best
  runnerMode: 'local',  // 'local' | 'slurm' | 'ssh'  (local = laptop/on-node dry run)
  device: 'cpu',        // 'cpu' for local dry run; 'cuda' on the cluster
  shapes: 'dev',        // 'dev' | 'official-safe' | 'all'
  push: false,          // git push after commits (needs a configured remote)
  finalize: true,       // draft the tech report at the end
}
const cfg = Object.assign({}, DEFAULTS, (args && typeof args === 'object') ? args : {})

// ---- model / effort tiers (the whole point of "optimize for usage limits") -//
const STRONG = { model: 'opus', effort: 'high' }   // strategise + postmortem
const CODER = { model: 'sonnet', effort: 'low' }    // write variants
const MECH = { model: 'haiku', effort: 'low' }      // git/runner/mechanical

const CD = `cd ${JSON.stringify(cfg.repoDir)}`
const pushNote = cfg.push
  ? 'Then run `git pull --rebase` and `git push` (a remote is configured).'
  : 'Commit locally only; do NOT push (no remote configured for this run).'

// ---- schemas -------------------------------------------------------------- //
const STATE_SCHEMA = {
  type: 'object', additionalProperties: true,
  properties: {
    best_median: { type: ['number', 'null'] },
    open_todos: { type: 'array', items: { type: 'object', additionalProperties: true } },
    tried_directions: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['open_todos', 'tried_directions'],
}
const HYPOTHESES_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    hypotheses: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        properties: {
          todo_ref: { type: 'string' },
          direction: { type: 'string' },
          hypothesis: { type: 'string' },
          approach: { type: 'string' },
        },
        required: ['direction', 'hypothesis', 'approach'],
      },
    },
  },
  required: ['hypotheses'],
}
const CODER_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    path: { type: ['string', 'null'] },
    direction: { type: 'string' },
    summary: { type: 'string' },
    dev_correct: { type: 'boolean' },
    dev_error: { type: ['string', 'null'] },
  },
  required: ['path', 'direction', 'dev_correct'],
}
const EVAL_LIST_SCHEMA = {
  type: 'object', additionalProperties: true,
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: true,
        properties: {
          candidate: { type: 'string' },
          correctness_passed: { type: 'boolean' },
          median_speedup: { type: ['number', 'null'] },
        },
        required: ['candidate', 'correctness_passed'],
      },
    },
  },
  required: ['results'],
}
const POSTMORTEM_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    new_best: { type: 'boolean' },
    best_median: { type: ['number', 'null'] },
    decisions: { type: 'array', items: { type: 'object', additionalProperties: true } },
    summary: { type: 'string' },
  },
  required: ['new_best', 'best_median'],
}

// ---- prompt builders ------------------------------------------------------ //
function syncPrompt(r) {
  return `You are the SYNC-IN step of the autoresearch swarm, agent-id "${cfg.agentId}", round ${r}.
${CD}
1. If a git remote exists, run \`git pull --rebase\` (ignore/report errors if no remote — this may be a local-only run).
2. Read PROGRAM.md (playbook + correctness contract), leaderboard.md (current best median speedup), TODO.md (the "## Open" and "## Seeded by human" sections), and the last ~15 lines of journal.jsonl.
Return: best_median (current best median speedup, or null if pending), open_todos (the claimable items, human-seeded first, each {ref, direction, text}), tried_directions (directions already attempted per the journal), and a short notes string. Do not modify any files.`
}

function strategistPrompt(r, state) {
  return `You are the STRATEGIST (strong model) of the autoresearch swarm, agent-id "${cfg.agentId}", round ${r}.
Goal: pick ${cfg.nVariants} DISTINCT, high-value optimization hypotheses to try this round, then CLAIM them.
Context (from sync-in): ${JSON.stringify(state).slice(0, 4000)}
Rules:
- Read PROGRAM.md for the ranked playbook and the correctness contract.
- Prioritize human-seeded TODO items. Avoid directions already in tried_directions unless you have a concretely different angle.
- Each hypothesis must be a single, testable change branching from candidates/best.py, and must state which shapes you expect it to help and why.
- CLAIM each chosen item: in ${cfg.repoDir}/TODO.md move it from "## Open" to "## In progress" annotated "claimed by ${cfg.agentId} round ${r}" (formalize a human free-text idea into a proper item first if needed). ${CD} && git add -A && git commit -m "claim: round ${r} by ${cfg.agentId}". ${pushNote}
Return the list of hypotheses (todo_ref, direction, hypothesis, approach).`
}

function coderPrompt(r, k, h) {
  const out = `candidates/${cfg.agentId}-r${r}-${k}.py`
  return `You are a CODER (cheap model, low effort) of the autoresearch swarm, agent-id "${cfg.agentId}", round ${r}, variant ${k}.
${CD}
Task: implement exactly ONE change and write it to ${out}.
Hypothesis: ${JSON.stringify(h)}
Hard rules (from PROGRAM.md "Correctness contract"): exact GELU approximate="none"; fp32-stable softmax; zero padded-query outputs; causal via SDPA is_causal when no padding; key-padding via additive -inf mask; scale 1/sqrt(head_dim); return [B,S,d_model]; keep baseline param names for strict weight copy OR set STRICT_WEIGHT_COPY=False and provide copy_model_weights. Start from candidates/best.py and change only what the hypothesis requires.
Then SANITY-CHECK locally on CPU:
  python runner.py --candidates ${out} --mode local --device cpu --shapes dev
Parse its JSON. Set dev_correct = (the single result's correctness_passed). If it errored, capture dev_error (the first error string) and set dev_correct=false; you may make ONE fix attempt, otherwise report the failure honestly.
Return {path: "${out}" (or null if you failed to produce a file), direction, summary (1 line), dev_correct, dev_error}.`
}

function runnerPrompt(r, survivors) {
  return `You are the RUNNER (mechanical) of the autoresearch swarm, round ${r}.
${CD}
Evaluate these dev-correct candidates on the real target with the harness:
  python runner.py --candidates ${survivors.join(' ')} --mode ${cfg.runnerMode} --device ${cfg.device} --shapes ${cfg.shapes}
This prints a JSON array (one object per candidate). Return it as {results: [...]} — pass the objects through faithfully (keep candidate, correctness_passed, median_speedup, geomean_speedup, per_shape, errors). Do not edit files.`
}

function postmortemPrompt(r, evalRes, bestMedian) {
  return `You are the POSTMORTEM/ANALYST (strong model) of the autoresearch swarm, agent-id "${cfg.agentId}", round ${r}.
${CD}
Current best median speedup: ${bestMedian === null ? 'null (pending)' : bestMedian}.
Evaluation results this round: ${JSON.stringify(evalRes).slice(0, 6000)}
Do:
1. For each candidate decide keep | prune | new-best. A candidate is eligible to become best only if correctness_passed is true AND its median_speedup strictly beats the current best.
2. Append one line per candidate to journal.jsonl (schema in AGENTS.md §4: iter, round=${r}, agent_id="${cfg.agentId}", node_id, parent_id, direction, hypothesis, diff=<candidate path>, eval{correctness_passed, median_speedup, geomean_speedup, per_shape}, decision, recovery, cost). Diagnose any correctness_fail / candidate_error / baseline_oom and record how it was handled in "recovery".
3. Append a short human-readable entry per candidate to LOG.md.
4. If there is a new best: GUARDED UPDATE (AGENTS.md §2) — git pull --rebase, re-check you still beat leaderboard.md, then copy the winning candidate to candidates/best.py and update leaderboard.md (including the per-shape table).
5. Mark the round's TODO items "## Done" with their result; optionally add follow-up ideas under "## Open".
6. ${CD} && git add -A && git commit -m "round ${r} results by ${cfg.agentId}". ${pushNote}
Return {new_best, best_median (the median speedup of the best after this round), decisions:[{candidate, decision, median_speedup}], summary}.`
}

// ---- main loop ------------------------------------------------------------ //
log(`autoresearch agent "${cfg.agentId}" starting: mode=${cfg.runnerMode} device=${cfg.device} shapes=${cfg.shapes} rounds<=${cfg.rounds}`)

let bestMedian = null
let dryRounds = 0
const roundSummaries = []

for (let r = 1; r <= cfg.rounds; r++) {
  if (budget.total && budget.remaining() < 40000) {
    log(`budget nearly exhausted (${Math.round(budget.remaining() / 1000)}k left) — stopping before round ${r}`)
    break
  }
  const phaseName = `Round ${r}`
  phase(phaseName)

  // 1. sync-in + state (mechanical)
  const state = await agent(syncPrompt(r), { ...MECH, label: `sync-in r${r}`, phase: phaseName, schema: STATE_SCHEMA })
  if (state && typeof state.best_median === 'number' && bestMedian === null) bestMedian = state.best_median

  // 2. strategist (strong) — choose + claim N hypotheses
  const strat = await agent(strategistPrompt(r, state || {}), { ...STRONG, label: `strategist r${r}`, phase: phaseName, schema: HYPOTHESES_SCHEMA })
  const hyps = ((strat && strat.hypotheses) || []).slice(0, cfg.nVariants)
  if (!hyps.length) { log(`round ${r}: strategist proposed nothing — stopping`); break }

  // 3. coders (cheap, parallel) — write + local-CPU sanity each variant
  const coded = await parallel(hyps.map((h, k) => () =>
    agent(coderPrompt(r, k, h), { ...CODER, label: `coder:${h.direction} r${r}`, phase: phaseName, schema: CODER_SCHEMA })
  ))
  const survivors = coded.filter(Boolean).filter(c => c.path && c.dev_correct).map(c => c.path)
  const dropped = coded.filter(Boolean).filter(c => !(c.path && c.dev_correct))
  if (dropped.length) log(`round ${r}: ${dropped.length}/${coded.length} variants failed local sanity and were dropped`)
  if (!survivors.length) { log(`round ${r}: no dev-correct variants — continuing`); dryRounds++; if (dryRounds >= cfg.convergenceK) break; continue }

  // 4. runner (mechanical) — evaluate survivors on the real target
  const evalRes = await agent(runnerPrompt(r, survivors), { ...MECH, label: `runner r${r}`, phase: phaseName, schema: EVAL_LIST_SCHEMA })

  // 5. postmortem (strong) — decide, log, guarded best update, commit
  const pm = await agent(postmortemPrompt(r, evalRes || { results: [] }, bestMedian), { ...STRONG, label: `postmortem r${r}`, phase: phaseName, schema: POSTMORTEM_SCHEMA })
  roundSummaries.push({ round: r, ...(pm || {}) })
  if (pm && pm.new_best) {
    bestMedian = (typeof pm.best_median === 'number') ? pm.best_median : bestMedian
    dryRounds = 0
    log(`round ${r}: NEW BEST median speedup = ${bestMedian}`)
  } else {
    dryRounds++
    log(`round ${r}: no new best (${dryRounds}/${cfg.convergenceK} dry rounds)`)
  }
  if (dryRounds >= cfg.convergenceK) { log(`converged after ${dryRounds} dry rounds`); break }
}

// ---- finalize: draft the tech report from the shared notebook ------------- //
if (cfg.finalize) {
  phase('Finalize')
  await agent(
    `You are the FINALIZER (strong model). ${CD}
Read journal.jsonl, LOG.md, and leaderboard.md. Write report/TECH_REPORT.md for TikTok TechJam Track 3 containing: (1) environment (GPU, torch, dtype), (2) the optimizations tried and why (grouped by direction, citing SDPA/torch.compile/etc.), (3) the final best candidate and its per-shape speedups + correctness, (4) how failures were handled (robustness), (5) resource usage (rounds, approximate tokens/wall-clock), and (6) a short reflection on limitations + next steps. Create the report/ directory if needed. Commit locally with message "draft tech report". ${cfg.push ? 'Then push.' : 'Do not push.'}`,
    { ...STRONG, label: 'finalize report', phase: 'Finalize' }
  )
}

return {
  agentId: cfg.agentId,
  bestMedian,
  rounds: roundSummaries.length,
  roundSummaries,
}
