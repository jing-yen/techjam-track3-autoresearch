#!/usr/bin/env bash
# Refuse to submit while any <FILL> placeholder remains.
set -uo pipefail
cd "$(dirname "$0")/.."
# Skip the lines that *describe* the placeholder convention.
FILT='grep -v -i placeholder'
hits=$(grep -rn "<FILL" README.md TECH_REPORT.md 2>/dev/null | eval $FILT | wc -l | tr -d ' ')
if [ "$hits" -gt 0 ]; then
  echo "BLOCKED: $hits placeholder(s) still unfilled."
  grep -rn "<FILL" README.md TECH_REPORT.md | eval $FILT | sed 's/^/  /'
  exit 1
fi
echo "OK: no placeholders remain."
