#!/usr/bin/env bash
# Capture the exact environment for the tech report (TechJam Track 3 sec 3.5
# requires CPU / GPU / DISK). Run ON THE GPU NODE, not the login node:
#
#   ssh xlogin
#   srun --gres=gpu:a100-80:1 --pty bash
#   bash scripts/capture_env.sh > docs/environment.txt
#
set -uo pipefail
echo "# Environment capture — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo
echo "## Host"
hostname; uname -a
echo
echo "## CPU"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz" || sysctl -n machdep.cpu.brand_string
echo
echo "## Memory"
free -h 2>/dev/null || vm_stat
echo
echo "## GPU"
nvidia-smi 2>/dev/null || echo "no nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null
echo
echo "## Disk"
df -h . 2>/dev/null
echo
echo "## Software"
python -c "
import torch, platform
print('python  ', platform.python_version())
print('torch   ', torch.__version__)
print('cuda    ', torch.version.cuda)
print('cudnn   ', torch.backends.cudnn.version())
print('device  ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')
try:
    import triton; print('triton  ', triton.__version__)
except ImportError: print('triton   not installed')
print('tf32_matmul', torch.backends.cuda.matmul.allow_tf32)
"
