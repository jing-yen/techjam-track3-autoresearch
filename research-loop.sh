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
  --permission-mode acceptEdits \
  --allowedTools "${TOOLS[@]}" \
  2>&1 | tee "$RUN/1-plan.log"

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
    --output-last-message .research/review.json \
    "$(cat prompts/review.md)" \
    2>&1 | tee "$RUN/2-review.log"
else
  echo "codex not found at $CODEX_BIN -> set CODEX_BIN. Falling back to Claude."
  echo "NOTE: same model reviewing itself catches less. Stopgap only."
  claude -p "$(cat prompts/review.md)

Write your JSON to .research/review.json. It must validate against schemas/review.json." \
    --model "$MODEL" --effort "$EFFORT" \
    --permission-mode acceptEdits \
    --allowedTools "${TOOLS[@]}" \
    2>&1 | tee "$RUN/2-review.log"
fi

[ -s .research/review.json ] || { echo "FAIL: leg 2 produced no review"; exit 1; }
python3 -c "import json,sys; d=json.load(open('.research/review.json')); \
print(f\"  verdict={d['verdict']}  findings={len(d['findings'])}\")" \
  || { echo "FAIL: review.json is not valid JSON"; exit 1; }

# --- leg 3: RECONCILE ------------------------------------------------------
say "leg 3/3  RECONCILE  (claude)"
claude -p "$(cat prompts/reconcile.md)" \
  --model "$MODEL" --effort "$EFFORT" \
  --permission-mode acceptEdits \
  --allowedTools "${TOOLS[@]}" \
  2>&1 | tee "$RUN/3-reconcile.log"

[ -s .research/TODO.next.md ] || { echo "FAIL: leg 3 produced no candidate"; exit 1; }

# --- the gate --------------------------------------------------------------
cp .research/TODO.proposed.md .research/review.json "$RUN/" 2>/dev/null || true
wc -l < journal.jsonl | tr -d ' ' > .research/.last_journal_lines

say "STAGED — nothing has been written to TODO.md"
git --no-pager diff --no-index --stat TODO.md .research/TODO.next.md || true
cat <<'NEXT'

Review it:
    git --no-pager diff --no-index TODO.md .research/TODO.next.md

Accept:
    cp .research/TODO.next.md TODO.md

Then commit yourself. This script never runs git.
NEXT
