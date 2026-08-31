# Queue audit at T-26 hours: the backlog is healthy, the submission is not

Engineering brief · 2026-08-31 · quick mode
Source: full `TODO.md` read (all sections), `journal.jsonl` (32 rows),
`leaderboard.md`, `scripts/check_placeholders.sh` output. No candidate source
re-read.

## Verdict in one line

The technical queue is in the best shape it has ever been — and the two items
that can still sink the submission are not in it as first-class work: **the demo
video does not exist, and no one has scheduled the time to make it.**

## State of the queue (all 20+ items audited)

**Closed with evidence, correctly:** S1-S5, T0-T8, B0-B11 mostly, L1-L3, M1.
The falsification rate is high and honest — L1, L2, S2, M1 all closed as
*negative* results with measurements attached. That discipline is itself
report-worthy.

**Open, correctly ranked by the teammate:** S8 (error bar), S9 (seed sweep),
M2 (first-ever profile), S6 (ledger root cause), then T10/T11/T12/T7b/T9.

**Mis-weighted, two items:**

1. **D1's remainder is treated as a footnote.** "Both need a human to actually
   record/write/submit" is the entire risk statement for a **hard submission
   requirement**. §3.5: the video must exist, be public on YouTube, and be
   linked in Devpost. A 3.72x geomean with no video is an incomplete
   submission; a 2x with a clear video is not. Nothing else in the queue has
   this property.
2. **B0 remains unchecked** ("Still worth getting", `TODO.md:611`). Thirty
   seconds to diff our `torch_transformer_benchmark.py` against the organizer's
   current download. The tail risk is silent total loss. It has survived three
   research passes without anyone doing it.

## The one genuinely new idea this audit produced

**S8 and the demo video are the same event.** S8 requires re-running the
unchanged champion twice on the official protocol. A demo video for a
backend-track submission is, per the problem statement's own note, "a
walkthrough video showing API usage, inference examples, or result analysis."

Record the S8 confirmation sweep as the video's centerpiece:

- terminal capture of `python runner.py --candidates candidates/v_router2.py
  --shapes official-safe` submitting to Slurm and returning 13/13 PASS with the
  per-shape table,
- cut to the leaderboard/README table it regenerates,
- one screen of the swarm machinery (TODO claim board, a journal row, the
  plan→review→reconcile transcript) — the AI-tooling story §3.5 gives bonus
  points for,
- voiceover from the script below.

One GPU allocation produces the error bar S8 wants **and** the footage D2 needs.
No staged demo, no mock data — the video shows the actual measurement that
produced the reported number, which is a stronger artifact than a slide deck.

## Recommended endgame (25.8 h)

| when | what | who |
|--|--|--|
| now | B0 diff (30 s); draft Devpost text + video script (drafts committed alongside this brief) | research layer |
| next GPU slot | **S8 x2 + S9 seeds, screen-recorded** | teammate |
| after | error bar into README/TECH_REPORT; if variance is large, soften the T7 narrative exactly as `research-batch3.md` prescribed | research layer |
| T-12h | **freeze `v_router2`**. No new kernels after this point — an untested last-minute change that fails one shape forfeits everything | team decision |
| T-12h → T-4h | record/edit/upload video (YouTube processing + a re-upload buffer is why 4h, not 1h); fill 5 team names; submit Devpost | human |

**Explicitly frozen, with reasons already in the queue:** T7b and T9 (target
gains below the documented ±2.7%-per-shape noise floor), T10 (own risk note:
"budget for it to take longer than T7 did"), T11/T12 (compile-time and
launch-overhead leads, not leaderboard leads). M2 profiling is worth doing
**only if** it rides along in the S8 allocation; it stops being worth a
dedicated slot the moment the freeze passes, because its purpose is to justify
kernels nobody should start.

## Limitations

1. This audit re-read the queue and ledgers, not candidate source; item status
   is taken from the teammate's own notes where a journal row confirms it.
2. The S8-as-video plan assumes screen recording on the laptop driving the ssh
   session is acceptable footage quality; UNVERIFIED, trivial to test.
3. Time estimates for video production are experience-based guesses, not data.

## AI disclosure

Produced with AI assistance (Claude Opus 5). Queue status claims are traceable
to `TODO.md` line numbers and journal iterations cited inline.
