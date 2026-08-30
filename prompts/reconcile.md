You are closing the research loop. Read .research/TODO.proposed.md and .research/review.json,
produce the final TODO.md. ONE ROUND. There is no second pass and no reply to
the reviewer.

## Process

For each finding in review.json, in severity order:

- **Verify it yourself first.** Open the cited file:line. The reviewer is
  another model and can be wrong. Do not accept a finding on authority.
- **Agree** -> apply the fix to the item.
- **Disagree** -> keep the original item unchanged, and record the
  disagreement with your reason.

## Output

Write `.research/TODO.next.md`. Same structure as .research/TODO.proposed.md, plus one section at the
bottom:

```
## Review log — <date>

| finding | severity | verdict | reason |
|---|---|---|---|
| bad_citation on B3 | major | accepted | line was 641 not 638, fixed |
| misranked T5 vs T1 | minor | rejected | T5 depends on B2's backend data, T1 does not; order stands |
```

Keep the log to the current round only. Delete the previous round's log.

## Hard rules

- Do not add new items that were not in .research/TODO.proposed.md or .research/review.json.
- Do not soften an item to make a finding go away. Either fix it or reject it.
- If you rejected a blocker-severity finding, say so prominently at the top.
- Never claim a speedup without a journal row.

Do NOT write to TODO.md and do NOT run git. The driver script stages your
output for human approval.
