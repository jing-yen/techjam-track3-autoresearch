# Demo video script (~3 min, walkthrough format per §3.5's backend-track note)

**Format:** screen recording + voiceover. No slides needed except the title card.
Record the S8 confirmation sweep live — the video then shows the real
measurement behind the reported number.

## Shot list

**0:00-0:20 — Title card + problem.**
"Track 3: make this Transformer layer faster on a GPU without changing its
answers. Fourteen input shapes, published in advance. Correctness first: every
element within 0.002 absolute or 2% relative, or the speed doesn't count."

**0:20-0:50 — The reference and the bottleneck.** Show
`torch_transformer_benchmark.py:97-111` briefly.
"The reference builds the full attention score matrix in memory. For the largest
shape that matrix would be 18.6 terabytes — the reference cannot run its own
biggest test. Our implementation never builds it."

**0:50-1:40 — The swarm.** Show TODO.md claim board, one journal.jsonl row, the
plan→review→reconcile loop output, a commit with the Co-Authored-By trailer.
"We ran this as a multi-agent research loop: one layer proposes and ranks
hypotheses with evidence requirements, a different model family audits every
claim against the source, and nothing counts until a benchmark row lands in the
ledger. The loop caught real bugs — an unreachable fast path, a TF32 precision
drift, and a headline number that was 24% inflated by a mismeasured protocol.
All of it is in the git history."

**1:40-2:30 — The live run.** Terminal: `python runner.py --candidates
candidates/v_router2.py --shapes official-safe`. Show Slurm submission, then the
returning JSON: 13/13 passed, per-shape speedups.
"One implementation, four code paths — the rules allow shape-specific dispatch.
Compiled for the launch-bound small shapes, fused kernels plus fp16 autocast for
the compute-heavy ones, and a custom Triton kernel for the residual-add plus
LayerNorm. Median 2.99x, geometric mean 3.72x, all shapes passing."

**2:30-2:50 — Shape 14.** Show the chunked run line.
"The largest shape OOMed at 73.85 of 79.25 gigabytes — so we split its 32
independent sequences into groups of four. Exact, bit-identical, and it runs.
The reference still can't."

**2:50-3:00 — Close.** README on screen.
"Everything reproduces from the public repo: one command per result, every
number traceable to a ledger row. Links in the description."

## Production notes

- Record at ≥1080p; terminal font large enough to read at 720p playback.
- YouTube: public, title "TechJam 2026 Track 3 — [team name]", repo link in
  description, then paste the URL into Devpost.
- Budget one full hour for upload + processing + a re-upload if the first take
  has a problem. Do not start this at T-1h.
