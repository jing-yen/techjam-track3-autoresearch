#!/usr/bin/env bash
#
# Research layer: plan -> review -> reconcile -> STAGED TODO.md
#
# Nothing here touches git. The loop writes only into .research/ (gitignored)
# and ends by showing you a diff. Applying it to TODO.md and pushing are two
# separate, deliberate commands you run yourself.
#
# Run ONCE per batch of GPU results. By hand. Never on a cron: the input to
# this loop is measurements, and with no new journal rows it just reshuffles
# prose and burns tokens.
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${RESEARCH_MODEL:-opus}"
EFFORT="${RESEARCH_EFFORT:-max}"

# Review leg runs on Codex. The binary ships inside the ChatGPT app bundle,
# not on PATH. If codex rejects effort "ultra", try "xhigh".
CODEX_BIN="${CODEX_BIN:-/Applications/ChatGPT.app/Contents/Resources/codex}"
CODEX_MODEL="${CODEX_MODEL:-gpt-5.6-sol}"
CODEX_EFFORT="${CODEX_EFFORT:-ultra}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN=".research/runs/$STAMP"
mkdir -p "$RUN"

# Tight allowlist: the legs read five files and write one. Nothing else.
# Without this, `claude -p` blocks on a permission prompt with no one to answer.
TOOLS=(Read Write Edit Glob Grep "Bash(tail:*)" "Bash(git log:*)" "Bash(git diff:*)")

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Extract token usage from a `claude -p --output-format json` result and append
# it to the run's usage ledger.
usage_line() {
  python3 - "$1" "$2" "$RUN/usage.jsonl" <<'PYEOF'
import json, sys
leg, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(path))
except Exception as e:
    print(f"  [{leg}] could not parse usage: {e}"); sys.exit(0)
u = d.get("usage", {})
rec = {"leg": leg, "model": d.get("modelUsage") and list(d["modelUsage"]) or None,
       "cost_usd": d.get("total_cost_usd"), "duration_ms": d.get("duration_ms"),
       "num_turns": d.get("num_turns"),
       "input": u.get("input_tokens"), "output": u.get("output_tokens"),
       "cache_create": u.get("cache_creation_input_tokens"),
       "cache_read": u.get("cache_read_input_tokens")}
open(out, "a").write(json.dumps(rec) + "\n")
tot = sum(v or 0 for v in (rec["input"], rec["output"], rec["cache_create"], rec["cache_read"]))
print(f"  [{leg}] {tot:,} tokens  (in {rec['input']:,} / out {rec['output']:,} / "
      f"cache-w {rec['cache_create']:,} / cache-r {rec['cache_read']:,})  "
      f"${rec['cost_usd']:.4f}  {rec['num_turns']} turns")
if d.get("result"): print("  ---\n  " + str(d["result"])[:600].replace("\n", "\n  "))
PYEOF
}

# --- guard: is there anything new to plan against? -------------------------
if [ -f .research/.last_journal_lines ]; then
  prev=$(cat .research/.last_journal_lines)
  now=$(wc -l < journal.jsonl | tr -d ' ')
  if [ "$now" = "$prev" ]; then
    echo "journal.jsonl has not grown since the last run ($now rows)."
    echo "This loop eats measurements. Run the benchmark first, or pass --force."
    [ "${1:-}" = "--force" ] || exit 1
  fi
fi

# --- leg 1: PLAN -----------------------------------------------------------
say "leg 1/3  PLAN  (claude, $MODEL, effort=$EFFORT)"
claude -p "$(cat prompts/plan.md)" \
  --model "$MODEL" --effort "$EFFORT" \
  --output-format json \
  --permission-mode acceptEdits \
  --allowedTools "${TOOLS[@]}" \
  > "$RUN/1-plan.json" 2> "$RUN/1-plan.err" || true
usage_line plan "$RUN/1-plan.json"

[ -s .research/TODO.proposed.md ] || { echo "FAIL: leg 1 produced no proposal"; exit 1; }

# --- leg 2: REVIEW ---------------------------------------------------------
say "leg 2/3  REVIEW  (codex $CODEX_MODEL, effort=$CODEX_EFFORT, read-only)"
if [ -x "$CODEX_BIN" ]; then
  "$CODEX_BIN" exec \
    --sandbox read-only \
    --skip-git-repo-check \
    --model "$CODEX_MODEL" \
    -c model_reasoning_effort="$CODEX_EFFORT" \
    --output-schema schemas/review.json \
    --json \
    --output-last-message .research/review.json \
    "$(cat prompts/review.md)" \
    > "$RUN/2-review.jsonl" 2> "$RUN/2-review.err" || true
  tail -3 "$RUN/2-review.err" || true
else
  echo "codex not found at $CODEX_BIN -> set CODEX_BIN. Falling back to Claude."
  echo "NOTE: same model reviewing itself catches less. Stopgap only."
  claude -p "$(cat prompts/review.md)

Write your JSON to .research/review.json. It must validate against schemas/review.json." \
    --model "$MODEL" --effort "$EFFORT" \
    --output-format json \
    --permission-mode acceptEdits \
    --allowedTools "${TOOLS[@]}" \
    > "$RUN/2-review.json" 2> "$RUN/2-review.err" || true
  usage_line review "$RUN/2-review.json"
fi

[ -s .research/review.json ] || { echo "FAIL: leg 2 produced no review"; exit 1; }
python3 -c "import json,sys; d=json.load(open('.research/review.json')); \
print(f\"  verdict={d['verdict']}  findings={len(d['findings'])}\")" \
  || { echo "FAIL: review.json is not valid JSON"; exit 1; }

# --- leg 3: RECONCILE ------------------------------------------------------
say "leg 3/3  RECONCILE  (claude)"
claude -p "$(cat prompts/reconcile.md)" \
  --model "$MODEL" --effort "$EFFORT" \
  --output-format json \
  --permission-mode acceptEdits \
  --allowedTools "${TOOLS[@]}" \
  > "$RUN/3-reconcile.json" 2> "$RUN/3-reconcile.err" || true
usage_line reconcile "$RUN/3-reconcile.json"

[ -s .research/TODO.next.md ] || { echo "FAIL: leg 3 produced no candidate"; exit 1; }

# --- the gate --------------------------------------------------------------
cp .research/TODO.proposed.md .research/review.json "$RUN/" 2>/dev/null || true
wc -l < journal.jsonl | tr -d ' ' > .research/.last_journal_lines

say "TOKEN / COST SUMMARY"
python3 - "$RUN" <<'PYEOF'
import json, sys, os, glob
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/usage.jsonl")] if os.path.exists(f"{run}/usage.jsonl") else []
tot_cost = tot_tok = 0
print(f"{'leg':<12}{'total tok':>12}{'input':>10}{'output':>9}{'cache-w':>10}{'cache-r':>11}{'cost':>10}")
for r in rows:
    t = sum(r.get(k) or 0 for k in ("input","output","cache_create","cache_read"))
    tot_tok += t; tot_cost += r.get("cost_usd") or 0
    print(f"{r['leg']:<12}{t:>12,}{r.get('input') or 0:>10,}{r.get('output') or 0:>9,}"
          f"{r.get('cache_create') or 0:>10,}{r.get('cache_read') or 0:>11,}{r.get('cost_usd') or 0:>9.4f}")
# codex leg
for f in glob.glob(f"{run}/2-review.jsonl"):
    # Codex reports usage once, on turn.completed. cached_input_tokens is a
    # SUBSET of input_tokens -- adding them double-counts.
    u = {}
    for line in open(f):
        try: e = json.loads(line)
        except Exception: continue
        cand = e.get("usage") or (e.get("msg") or {}).get("usage")
        if cand: u = cand
    if u:
        cin, cached = u.get("input_tokens",0), u.get("cached_input_tokens",0)
        out = u.get("output_tokens",0)
        print(f"{'review(cdx)':<12}{cin+out:>12,}{cin-cached:>10,}{out:>9,}"
              f"{u.get('cache_write_input_tokens',0):>10,}{cached:>11,}{'  n/a':>10}")
print("-"*74)
print(f"{'CLAUDE TOTAL':<12}{tot_tok:>12,}{'':>40}{tot_cost:>9.4f}")
PYEOF

say "STAGED — nothing has been written to TODO.md"
git --no-pager diff --no-index --stat TODO.md .research/TODO.next.md || true
cat <<'NEXT'

Review it:
    git --no-pager diff --no-index TODO.md .research/TODO.next.md

Accept:
    cp .research/TODO.next.md TODO.md

Then commit yourself. This script never runs git.
NEXT
