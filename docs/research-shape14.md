# Shape 14: which long-sequence methods are admissible, and which are not

Research brief · 2026-08-31 · quick mode
Question: are there methods beyond fp16/H100 that make shape 14 (B=32, S=100000,
d=1024, H=16, L=2) run, and can any of them exploit structure in the score table?

**Scope note.** Engineering brief, not a systematic review. Claims are grounded in
cited papers, this repo's measurements, and arithmetic shown inline. Untested
items are labelled UNVERIFIED.

---

## The constraint that decides everything

The gate is per-element agreement with the organizer's reference:
`abs(opt-ref) <= 0.002` **OR** `abs(opt-ref) <= 0.02*abs(ref)`, every element
(`torch_transformer_benchmark.py:314-316`).

**This splits the entire long-sequence literature in two**, and the split does
not follow the usual efficiency taxonomy:

| family | changes the answer? | admissible here |
|--|--|--|
| Exact-but-restructured (tiling, online softmax, chunking, device sharding) | no | **yes** |
| Approximate (sparse, low-rank, kernel/linear) | **yes** | **no** |

Nearly all published work on 100k-token attention is in the second row. It is
excellent research and it is inadmissible for this task, because the task is not
"produce good outputs on long sequences" — it is "reproduce these exact numbers,
faster."

---

## Answering the score-table question directly

The intuition — "the score table has structure, exploit it" — is precisely the
premise of the sparse/low-rank family, and it is well developed:

- **Longformer** and **BigBird** attend to a sparse subset of positions (local
  windows, global tokens, and in BigBird's case random ones), cutting the number
  of attended tokens per query.
- **Linformer** factorizes the attention matrix as low-rank; the documented cost
  is that it "loses high-rank information, reducing model capacity", and its
  decomposition is tied to a fixed context length.
- **Performer** replaces the softmax kernel with a kernel approximation to reach
  linear complexity, explicitly assuming neither sparsity nor low rank.

Survey coverage of the family and its accuracy costs:
[Hugging Face, *Long-range Transformers*](https://huggingface.co/blog/long-range-transformers);
[*Demystifying Sparse Attention*](https://medium.com/@rajboopathiking/demystifying-sparse-attention-longformer-bigbird-reformer-and-linformer-explained-029b97588144).
The recurring finding is a quality-efficiency trade-off, with sparse transformers
"often suffering from accuracy degradation".

**Why none of it is usable.** Every one of these deliberately computes a
*different function* from full softmax attention. Skipping a score-table entry is
not a rounding error — it removes a term from the softmax denominator and
redistributes probability mass across every remaining position. The output will
not land within 0.002 absolute or 2% relative of the reference. It would fail the
gate on shapes where a reference exists, and on shape 14 there is no reference to
check against at all — so we could not even demonstrate that it worked.

**Verdict: rejected on correctness grounds, not on effort grounds.** Worth one
sentence in the tech report, because knowing *why* the obvious literature is
inapplicable is itself a result.

---

## The exact family, and what each one buys us

### Already deployed — this is what killed the 18.6 TB

Memory-efficient / flash attention never materializes `QK^T`. It tiles the
computation into blocks sized for on-chip SRAM and uses an **online softmax**
with a rescaling fix-up per block, so the quadratic matrix "is never materialized
in HBM, which reduces the memory requirement from quadratic with sequence length
to linear."

- Rabe & Staats, *Self-Attention Does Not Need O(n²) Memory* (2021) — the online
  softmax formulation FlashAttention builds on.
- Dao, Fu, Ermon, Rudra & Ré, *FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness*, NeurIPS 2022
  ([proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html),
  [code](https://github.com/Dao-AILab/flash-attention)).

**This is the honest framing of our shape-14 result.** The 18.6 TB score matrix
is already gone — our candidates dispatch mem-efficient SDPA (measured:
`docs/sdpa_backend_probe.json`, flash 0/14 at fp32, mem-efficient 14/14). What
remains, ~85 GB, is **not attention**. It is the activations themselves.

### Ring Attention — right idea, wrong hardware

Liu, Zaharia & Abbeel, *Ring Attention with Blockwise Transformers for
Near-Infinite Context* (2023),
[arXiv:2310.01889](https://arxiv.org/abs/2310.01889),
[code](https://github.com/haoliuhl/ringattention). Blockwise attention and
feedforward distributed across devices, overlapping KV-block communication with
computation, enabling sequences "up to device count times longer... **without
resorting to approximations**."

Exact, and the closest thing in the literature to our problem. **Inapplicable:
it scales with device count and we request `gpu:a100-80:1`.** Worth citing in the
report as the principled answer we could not run.

### Batch chunking — exact, trivial, and NOT YET TRIED

Shape 14's residual memory is `[B,S,D]` activations, and **B=32 is 32 independent
sequences.** Nothing couples them. Process them in groups and concatenate:

```
full batch 32   7 live tensors    85.0 GB   >  79.25 GB   OOM
chunk of 16                       42.7 GB
chunk of 8                        21.4 GB
chunk of 4                        10.7 GB
chunk of 2                         5.3 GB
```

Peak at chunk=4, counting the persistent input and output tensors that must
survive the whole call:

```
10.7 (working) + 12.2 (input) + 12.2 (output)  =  ~35 GB   against 79.25
```

**Fits with room to spare, in fp32, with no precision change and no
approximation.** Output is bit-identical to the unchunked path because the
sequences never interact — this is what every production inference server does.

Roughly twenty lines inside `forward`: slice the batch, loop, `torch.cat`.

**Why it was skipped:** `TODO.md` B5 records "not pursuing fp16/H100/chunking per
rubric §3.3." That was a scope call made when shape 14 looked hopeless, before
the OOM was measured at 73.85 GB — only **6 GB** over. Chunking clears that by 44
GB.

---

## The correctness problem, and how to solve it anyway

Shape 14 has **no reference**: the baseline needs 18.6 TB for its score matrix,
so the organizer's own code cannot produce ground truth. A chunked shape 14
therefore "runs" but cannot be verified against the reference.

**This is solvable, and the solution is stronger than it sounds.** Validate the
chunking mechanism on shapes that *do* have a reference:

1. Run shapes 8 and 13 chunked and unchunked.
2. Confirm chunked output is bit-identical to unchunked, and that both pass the
   gate against the reference.
3. That proves the mechanism is exact.
4. Then report shape 14 as: runs, using a mechanism proven exact on shapes 8 and
   13, unverifiable against a reference that cannot exist.

That is a materially better deliverable than "OOM", and it is honest about
exactly what was and was not checked.

---

## Ranked recommendations

| rank | method | exact? | fits? | status | effort |
|--|--|--|--|--|--|
| 1 | **Batch chunking** | yes | ~35 GB | **never tried** | ~20 lines |
| 2 | fp16 via autocast | error-bounded, not exact | ~43 GB | `v_amp.py` written, never run on GPU | done, needs a sweep |
| 3 | Chunking + fp16 together | — | ~18 GB | not tried | trivial once (1) exists |
| 4 | Ring Attention | yes | needs N devices | inapplicable | n/a |
| 5 | Sparse / linear / low-rank | **no** | — | **rejected — fails the gate** | n/a |

**Recommendation with ~33 hours left:** do (1). It is the smallest change with
the largest deliverable impact — shape 14 moves from a documented failure to a
documented success with a stated verification caveat, and combined with running
shape 6 (also never attempted, ~6.7 GB, see `TODO.md` S4) the sweep goes from
**12/14 to 14/14**.

If it does not work in an hour, stop and keep the existing limitations text. It
is already well evidenced.

---

## Limitations

1. The ~35 GB chunked estimate is arithmetic from tensor shapes, not a measured
   allocation. PyTorch's caching allocator, fragmentation, and workspace buffers
   are unmodelled. UNVERIFIED until run.
2. "Bit-identical" for chunking assumes no cross-sequence reduction anywhere in
   the block. Confirmed by inspection of the reference (`BaselineTransformer`
   reduces over the feature dim in LayerNorm and over keys in attention, never
   over batch), not by execution.
3. Chunking serializes work that would otherwise run concurrently, so shape 14's
   latency will be worse than a hypothetical unchunked run. Since no unchunked
   run exists, no speedup ratio can be quoted for #14 either way.
4. The sparse/low-rank rejection is argued from the mechanism (softmax
   denominators change), not from a measured failure. No approximate method was
   implemented and tested against the gate.
5. Search covered the well-known long-context families. It was not a systematic
   PRISMA-style review and may miss recent exact methods.

## AI disclosure

Produced with AI assistance (Claude Opus 5). Paper claims are from the cited
sources; memory figures are computed from the tensor shapes in
`bench_harness.py:70-85`; the 73.85 GB OOM boundary is from `journal.jsonl`
iter 7.
