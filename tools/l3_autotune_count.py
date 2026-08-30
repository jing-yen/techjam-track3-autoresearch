"""
L3 -- does Inductor's max-autotune deduplicate autotuning across the N
structurally-identical repeated transformer layers, or does it redundantly
re-autotune each occurrence separately?

Result (journal iter 25, shape 1: batch=64,seq=128,d=128,num_layers=4):
CONFIRMED no deduplication -- 24 total AUTOTUNE log lines collapse to only
2 distinct GEMM shape signatures, each occurring 12 times (the identical
shape is autotuned separately at every occurrence in the unrolled 4-layer
graph). This is the mechanism behind iter 7's shape-14 finding (max-autotune
never finished within the 15-min limit): with several large-matmul call
sites per layer, redundant per-occurrence autotuning compounds directly.
Explanatory only -- does not change any leaderboard number, since v_router2
already avoids routing large-batch/large-seq shapes through the compile
route for this class of reason.
"""
import contextlib
import importlib.util
import io
import logging
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from bench_harness import OFFICIAL_SHAPES  # noqa: E402
from torch_transformer_benchmark import TransformerConfig  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

spec = importlib.util.spec_from_file_location("cand", os.path.join(_ROOT, "candidates", "v_compile.py"))
cand = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cand)

cfg_dict = OFFICIAL_SHAPES[1]  # batch=64, seq=128, d=128, num_layers=4
config = TransformerConfig(**cfg_dict)
model = cand.UserOptimizedTransformer(config).cuda()
x = torch.randn(cfg_dict["batch_size"], cfg_dict["seq_len"], cfg_dict["d_model"], device="cuda")
mask = torch.ones(cfg_dict["batch_size"], cfg_dict["seq_len"], dtype=torch.bool, device="cuda")

logging.getLogger("torch._inductor").setLevel(logging.DEBUG)
import torch._inductor.config as ind_cfg  # noqa: E402
ind_cfg.trace.enabled = False

buf = io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
    with torch.no_grad():
        model(x, mask)
    torch.cuda.synchronize()

log = buf.getvalue()
lines = [l for l in log.splitlines() if l.startswith("AUTOTUNE")]
print(f"Total AUTOTUNE lines: {len(lines)}")
c = Counter(lines)
for line, count in c.most_common(20):
    print(f"  x{count}  {line}")
print()
print(f"Distinct AUTOTUNE signatures: {len(c)}")
print(f"num_layers in this shape: {cfg_dict['num_layers']}")
