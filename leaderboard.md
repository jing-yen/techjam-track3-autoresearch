# Leaderboard — current best correct candidate

Update only via the **guarded best update** in `AGENTS.md` §2 (pull → re-check →
replace only if strictly better).

| field | value |
|--|--|
| best candidate | `candidates/best.py` (SDPA seed) |
| node_id | `best` |
| correctness | ✅ verified locally (CPU) on dev + `official-safe`; GPU pending |
| median speedup | _pending first cluster run_ |
| geomean speedup | _pending_ |
| dtype | float32 |
| updated by | seed |

## Per-shape speedups (fill after first cluster run)

| shape | passed | baseline_ms | opt_ms | speedup |
|--|--|--|--|--|
| _run `python runner.py --candidates candidates/best.py --shapes all` to populate_ | | | | |
