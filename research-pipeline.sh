#!/usr/bin/env bash
#
# Pipelined research loop: 2 researchers + 1 fact-checker, one round behind.
#
#   round N:  [agent A: topic 1] [agent B: topic 2]  run in parallel with
#             [agent C: fact-check round N-1's output]
#
# Verification never blocks research — it lags by exactly one round. Round 1's
# checker verifies "round 0", which is whatever docs/ already contains.
#
# Usage:  ./research-pipeline.sh "<topic for A>" "<topic for B>"
#
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${PIPELINE_MODEL:-sonnet}"
[ $# -eq 2 ] || { echo "usage: $0 '<topic A>' '<topic B>'"; exit 1; }

N=$(( $(ls -1d .research/rounds/*/ 2>/dev/null | wc -l | tr -d ' ') + 1 ))
PREV=$(printf '%03d' $((N-1))); CUR=$(printf '%03d' "$N")
mkdir -p ".research/rounds/$CUR"

if [ -d ".research/rounds/$PREV" ]; then
  CHECK_TARGET=".research/rounds/$PREV/findings-a.md .research/rounds/$PREV/findings-b.md"
else
  CHECK_TARGET="$(ls -1t docs/research-*.md 2>/dev/null | head -2 | tr '\n' ' ')"
fi

echo "==> round $CUR"
echo "    A: $1"
echo "    B: $2"
echo "    C: fact-checking -> $CHECK_TARGET"

run () { # name promptfile extra outfile
  claude -p "$(cat "$2")

$3" --model "$MODEL" --permission-mode acceptEdits \
    --allowedTools Read Glob Grep WebSearch WebFetch "Bash(python3:*)" \
    > "$4" 2>".research/rounds/$CUR/$1.err" || echo "[$1] FAILED, see .err"
}

run researchA prompts/pipeline-research.md "YOUR ASSIGNED TOPIC: $1" \
    ".research/rounds/$CUR/findings-a.md" &
run researchB prompts/pipeline-research.md "YOUR ASSIGNED TOPIC: $2" \
    ".research/rounds/$CUR/findings-b.md" &
run factcheck prompts/pipeline-factcheck.md "VERIFY THESE FILES: $CHECK_TARGET" \
    ".research/rounds/$CUR/verification.md" &
wait

echo "==> round $CUR complete: .research/rounds/$CUR/"
wc -l .research/rounds/$CUR/*.md 2>/dev/null || true
echo "Read the verification FIRST — it tells you whether last round's conclusions survived."
