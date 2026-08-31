# Shape #6 profiler trace (M2 completion) — full raw data, for whoever picks this up next

Job `779400`, `tools/profile_shapes.py --shapes 6`, candidate `v_router2.py`
(current champion post-T17, `amp` route), A100-80. Shape #6:
`batch_size=10000, seq_len=128, d_model=128, num_heads=4, ffn_dim=128,
num_layers=4, causal=True` — the largest-batch official shape, M = 1,280,000
rows through the FFN-block kernel.

This is the one M2-scoped shape that had never been individually profiled
(the original M2 scope covered #1, #8, #13; #6 was added here since it had
the single largest individual per-shape gain from T15, +17.2%, and nobody had
looked at *why*). Full table below — not summarized — so a fresh reader can
form their own hypothesis rather than trust mine alone.

## Full torch.profiler table (CPU+CUDA activities, 20 warmup + 20 active iters)

```
=== Shape #6 ({'batch_size': 10000, 'seq_len': 128, 'd_model': 128, 'num_heads': 4, 'ffn_dim': 128, 'num_layers': 4, 'causal': True}) -- candidate: v_router2.py ===
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                           aten::linear         0.86%       8.192ms        57.73%     547.013ms     569.805us       0.000us         0.00%     870.954ms     907.243us           960
                                            aten::addmm         1.91%      18.063ms        10.00%      94.724ms     197.342us     287.822ms        29.57%     315.180ms     656.625us           480
ampere_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_sta...         0.00%       0.000us         0.00%       0.000us       0.000us     287.822ms        29.57%     287.822ms     599.629us           480
                            _fused_add_layernorm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us     276.938ms        28.45%     276.938ms       1.731ms           160
                                               aten::to         0.22%       2.121ms        35.41%     335.546ms     262.145us       0.000us         0.00%     240.594ms     187.964us          1280
                                         aten::_to_copy         0.67%       6.324ms        35.19%     333.426ms     260.489us       0.000us         0.00%     240.594ms     187.964us          1280
                                            aten::copy_         0.87%       8.207ms        33.86%     320.809ms     250.632us     188.510ms        19.37%     240.594ms     187.964us          1280
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us     188.510ms        19.37%     188.510ms     147.273us          1280
                     aten::scaled_dot_product_attention         0.32%       3.075ms         0.99%       9.400ms     117.505us       0.000us         0.00%     128.839ms       1.610ms            80
              aten::_scaled_dot_product_flash_attention         0.10%     988.702us         0.67%       6.326ms      79.072us       0.000us         0.00%     128.839ms       1.610ms            80
                         aten::_flash_attention_forward         0.16%       1.560ms         0.46%       4.349ms      54.363us     128.839ms        13.24%     128.839ms       1.610ms            80
void pytorch_flash::flash_fwd_kernel<Flash_fwd_kerne...         0.00%       0.000us         0.00%       0.000us       0.000us     128.839ms        13.24%     128.839ms       1.610ms            80
                                       cudaLaunchKernel         1.57%      14.836ms        41.21%     390.501ms     201.289us       0.000us         0.00%      79.442ms      40.950us          1940
                                    Command Buffer Full        39.65%     375.665ms        39.65%     375.665ms     758.919us      79.442ms         8.16%      79.442ms     160.489us           495
                                       aten::layer_norm         0.15%       1.436ms         0.88%       8.338ms     416.880us       0.000us         0.00%      59.559ms       2.978ms            20
                                aten::native_layer_norm         0.40%       3.755ms         0.73%       6.902ms     345.089us      56.998ms         5.86%      59.559ms       2.978ms            20
void at::native::(anonymous namespace)::vectorized_l...         0.00%       0.000us         0.00%       0.000us       0.000us      56.998ms         5.86%      56.998ms       2.850ms            20
                                             aten::gelu         0.11%       1.034ms         0.18%       1.661ms      20.760us      34.268ms         3.52%      34.268ms     428.353us            80
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us      34.268ms         3.52%      34.268ms     428.353us            80
                                       cuLaunchKernelEx         0.16%       1.560ms         0.16%       1.560ms       9.749us       2.780ms         0.29%       2.780ms      17.376us           160
                                Activity Buffer Request         0.23%       2.217ms         0.23%       2.217ms       2.217ms       2.561ms         0.26%       2.561ms       2.561ms             1
                                            aten::empty         0.17%       1.657ms         0.17%       1.657ms       4.362us       0.000us         0.00%       0.000us       0.000us           380
                                             aten::view         0.42%       3.973ms         0.42%       3.973ms       2.027us       0.000us         0.00%       0.000us       0.000us          1960
                                    aten::empty_strided         0.87%       8.232ms         0.87%       8.232ms       4.900us       0.000us         0.00%       0.000us       0.000us          1680
                                                aten::t         0.12%       1.179ms         0.33%       3.136ms       6.534us       0.000us         0.00%       0.000us       0.000us           480
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 947.541ms
Self CUDA time total: 973.376ms
```

## Things worth noticing (not conclusions — starting points)

1. **`_fused_add_layernorm_kernel` (T7+T15's kernel) is 28.45% of CUDA time,
   276.938ms — almost as large as the fp16 GEMM itself (29.57%, 287.822ms).**
   This is the kernel with `grid=(n_rows,)` — one program per row. At this
   shape's M=1,280,000 rows, that's 1.28 million separate program instances
   across 160 calls (20 active iters × 4 layers × 2 boundaries). It is doing
   real, necessary work (this is exactly the kernel T15 confirmed as a real
   win here, +17.2% individually — the largest of any shape), but at this
   scale it is no longer a small cost sitting next to the GEMM; it is
   comparable in size to it.

2. **`Command Buffer Full` — 79.442ms self CUDA, 495 occurrences, and
   375.665ms of *CPU* time attributed to it (39.65% of total CPU time).**
   This is a CUDA driver/runtime signal that the command submission queue was
   full — the CPU was trying to enqueue GPU work faster than the driver could
   accept it, and had to wait. This is a DIFFERENT symptom of the same class
   of problem that turned out to explain T17's regression (`docs`/`TODO.md`
   T17 entry, journal iter 48-52): CPU-side dispatch pressure becoming a real
   cost, not hidden behind GPU compute. It was not present (or not a
   significant line item) in the M2 traces for shapes #1/#8/#13 profiled
   earlier — worth checking whether it shows up there too, or is specific to
   very high per-forward kernel-launch counts (shape #6 has ~1940
   `cudaLaunchKernel` calls across just 20 iterations here).

3. **The `aten::to`/`_to_copy`/`copy_` casting tax is back, and large: 19.37%
   self CUDA (188.5ms), 33.86% of CPU time.** This is the SAME mechanism T16
   targeted (and got a real-but-mixed result on, see `TODO.md` T16) — fp16
   autocast re-casting weights from fp32 on every forward call. T16's fix
   (pre-cast once) was tested standalone and helped shape #8 but regressed
   shape #6 specifically (-5.3%, outside the noise floor) in that test. Worth
   knowing that this cost is real and large in isolation (188.5ms here) even
   though T16's specific fix didn't net out positive for this shape overall —
   the two findings aren't necessarily contradictory (T16's fix may have
   introduced its own overhead that ate the savings; that was never fully
   isolated).

4. **`aten::native_layer_norm` (unfused) still has 20 calls (1 per forward),
   56.998ms total.** This is *expected*, not a gap: it's layer 0's `norm1`,
   which by design (T15) has no preceding residual-add to fuse it with (see
   `candidates/v_triton_addnorm2.py`'s docstring). Not a lead.

## The obvious next question this raises

`_AMPTransformer` (the `amp` route, serving shapes #6/#8/#13) does **not**
use CUDA graph capture at all — it's eager `torch.autocast`, same as it's
always been. T17 (`TODO.md`, journal iter 48-52) found and fixed an almost
identical symptom (CPU dispatch overhead exceeding available concurrent GPU
work) on a *different* route using *the exact same mechanism* — manual CUDA
graph capture — for a 51-244% improvement on the shapes it touched. The
`Command Buffer Full` line above is real, measured evidence (not inference)
that the `amp` route may have some of the same problem. Whether capturing
`_AMPTransformer`'s forward pass as a CUDA graph would help shape #6 (and
possibly #8/#13) the way it helped T17's shapes is untested — flagging as a
concrete, profiler-motivated next candidate for whoever picks it up, rather
than closing the M2 thread as "just profiled, nothing to act on."

## Limitations

One shape, one profiler run (20 active iterations) — not repeated, no error
bar on the breakdown itself (though S8's separate noise-floor work suggests
`amp`-routed shapes are among the most measurement-stable in this project).
`Command Buffer Full`'s exact meaning and whether it's a genuine bottleneck
signal or a profiler artifact was not independently verified against
external documentation — stated as "worth noticing," not "confirmed."
