# Cluster Setup — NUS SoC (A100/H100 via Slurm)

## 1. SSH access — and the password question

**Short answer: no, do not (and you cannot) put a password in your SSH config.**
SSH's config file has no password directive, and the runner calls `ssh`/`scp`
non-interactively many times per run — a password prompt would break the loop.
Use **key-based auth** so `ssh xlogin` is passwordless.

You already have an ed25519 key (`~/.ssh/id_ed25519.pub`). Do this once:

**a) Add an `xlogin` alias** to `~/.ssh/config`:

```sshconfig
Host xlogin
    HostName xlogin.comp.nus.edu.sg
    User YOUR_SOC_UNIX_ID
    IdentityFile ~/.ssh/id_ed25519
    # Recommended: authenticate once, reuse the connection for all runner calls.
    # This is the fallback if SoC still prompts (password/OTP) despite the key.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 8h
```

**b) Install your public key on the cluster** (asks for your SoC password *once*):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub xlogin
# verify — should NOT prompt for a password:
ssh xlogin 'hostname'
```

**Notes**
- Off campus you must be on the **SoC VPN** (not the NUS VPN).
- If SoC enforces a one-time password/OTP even with keys, the `ControlMaster`
  block above means you authenticate **once** (e.g. `ssh xlogin` in a terminal)
  and every subsequent `ssh`/`scp` from the runner reuses that live connection
  with no prompt, as long as it stays open (`ControlPersist 8h`).

## 2. One-time environment on the cluster

```bash
ssh xlogin
git clone https://github.com/jing-yen/techjam-track3-autoresearch.git
cd techjam-track3-autoresearch

# A CUDA-enabled PyTorch (Linux pip wheels ship CUDA by default):
conda create -n techjam python=3.11 -y && conda activate techjam
pip install torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"  # expect True
```

If your conda lives elsewhere, update `module_load` in `cluster.config.json`
to source the right activate script.

## 3. Choosing the GPU

Set `gres` in `cluster.config.json` (or `RUNNER_GRES`):

| GPU | gres value | nodes | when |
|--|--|--|--|
| A100 40GB | `gpu:a100-40:1` | xgpg | small shapes |
| **A100 80GB** (default) | `gpu:a100-80:1` | xgph | needed for shape #14 (seq=100k) |
| H100 96GB | `gpu:h100-96:1` | xgpi | fastest; best for #14 |

`--time` defaults to 30 min (SoC max is 3 h — bump to `03:00:00` for a full
`--shapes all` sweep with `torch.compile` warmups). `partition` is left blank;
the `--gres` request routes the job to a GPU node. Add `partition` only if your
jobs don't schedule.

## 4. Run

```bash
# from your Mac (mode: ssh) OR on the login node (set mode: slurm):
python runner.py --candidates candidates/best.py --shapes all      # measure the seed on GPU
python tests/test_bench_harness.py && python tests/test_runner.py   # sanity

# then launch the swarm from Claude Code via the Workflow tool with, e.g.:
#   args = { runnerMode: "ssh", device: "cuda", shapes: "official-safe",
#            push: true, agentId: "opus-1", rounds: 5, nVariants: 3 }
```

## 5. First real numbers

The seed's speedups so far are CPU noise. `python runner.py --candidates
candidates/best.py --shapes all` on an A100/H100 gives the first honest
per-shape speedups and populates `leaderboard.md`.
