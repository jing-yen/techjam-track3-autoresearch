#!/usr/bin/env python3
"""
Cluster runner: evaluate one or more candidates and return structured JSON.

Modes
-----
* ``local``  : run ``bench_harness.py`` as a subprocess per candidate (no
               scheduler). Used for laptop dry-runs and for the "agent runs on
               the GPU node" path.
* ``slurm``  : submit all candidates as ONE Slurm **array** job (one task per
               candidate) to amortize queue latency, poll until done, collect
               each ``result_<i>.json``. Orchestrator runs on the login node.
* ``ssh``    : same as ``slurm`` but every cluster command is wrapped in
               ``ssh <host> '...'`` and candidate files are scp'd to the remote
               workdir first. Orchestrator runs off-cluster (e.g. a laptop).

Output: a JSON list (one bench_harness result object per candidate, in input
order) printed to stdout. Missing/failed array tasks become a
``{"status": "runner_error", ...}`` record so a crashed task never stalls the
loop — this is the recovery path the swarm relies on.

Config: ``cluster.config.json`` (see that file). Env vars override:
RUNNER_MODE, RUNNER_SSH_HOST, RUNNER_REMOTE_WORKDIR, RUNNER_PARTITION,
RUNNER_ACCOUNT, RUNNER_GRES, RUNNER_TIME, RUNNER_MODULE_LOAD, RUNNER_PYTHON,
RUNNER_DEVICE, RUNNER_POLL_INTERVAL_S, RUNNER_MAX_WAIT_S.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

_ENV_OVERRIDES = {
    "mode": "RUNNER_MODE",
    "ssh_host": "RUNNER_SSH_HOST",
    "remote_workdir": "RUNNER_REMOTE_WORKDIR",
    "partition": "RUNNER_PARTITION",
    "account": "RUNNER_ACCOUNT",
    "gres": "RUNNER_GRES",
    "exclude": "RUNNER_EXCLUDE",
    "array_throttle": "RUNNER_ARRAY_THROTTLE",
    "time": "RUNNER_TIME",
    "module_load": "RUNNER_MODULE_LOAD",
    "python": "RUNNER_PYTHON",
    "device": "RUNNER_DEVICE",
    "poll_interval_s": "RUNNER_POLL_INTERVAL_S",
    "max_wait_s": "RUNNER_MAX_WAIT_S",
}

_DEFAULTS = {
    "mode": "local", "ssh_host": "", "remote_workdir": HERE,
    "partition": "", "account": "", "gres": "gpu:a100:1", "exclude": "",
    "array_throttle": 1, "time": "00:20:00",
    "module_load": "", "python": sys.executable or "python", "device": "cuda",
    "poll_interval_s": 10, "max_wait_s": 3600,
}


def load_config(path: Optional[str]) -> Dict:
    cfg = dict(_DEFAULTS)
    if path and os.path.exists(path):
        with open(path) as f:
            disk = json.load(f)
        for k, v in disk.items():
            if not k.startswith("_"):
                cfg[k] = v
    for key, env in _ENV_OVERRIDES.items():
        if os.environ.get(env) is not None:
            cfg[key] = os.environ[env]
    cfg["poll_interval_s"] = float(cfg["poll_interval_s"])
    cfg["max_wait_s"] = float(cfg["max_wait_s"])
    return cfg


# --------------------------------------------------------------------------- #
# Shell helpers
# --------------------------------------------------------------------------- #
def _sh(cmd: str, ssh_host: str = "", check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    full = f"ssh {ssh_host} {shlex.quote(cmd)}" if ssh_host else cmd
    return subprocess.run(
        full, shell=True, check=check, text=True,
        capture_output=capture,
    )


def parse_job_id(sbatch_stdout: str) -> str:
    """Extract the job id from 'Submitted batch job 12345'."""
    m = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not m:
        raise RuntimeError(f"could not parse sbatch job id from: {sbatch_stdout!r}")
    return m.group(1)


def render_sbatch(cfg: Dict, job: Dict) -> str:
    template_path = os.path.join(HERE, "sbatch_template.sh")
    with open(template_path) as f:
        tmpl = f.read()
    account_line = f"#SBATCH --account={cfg['account']}" if cfg.get("account") else ""
    partition_line = f"#SBATCH --partition={cfg['partition']}" if cfg.get("partition") else ""
    exclude_line = f"#SBATCH --exclude={cfg['exclude']}" if cfg.get("exclude") else ""
    throttle = int(cfg.get("array_throttle") or 0)
    array_throttle = f"%{throttle}" if throttle > 0 else ""
    subs = {
        "JOB_NAME": job["job_name"],
        "PARTITION_LINE": partition_line,
        "ACCOUNT_LINE": account_line,
        "EXCLUDE_LINE": exclude_line,
        "ARRAY_THROTTLE": array_throttle,
        "GRES": cfg["gres"],
        "TIME": cfg["time"],
        "LOGDIR": job["logdir"],
        "ARRAY_MAX": str(len(job["candidates"]) - 1),
        "MODULE_LOAD": cfg.get("module_load", ""),
        "WORKDIR": job["workdir"],
        "CANDIDATE_LIST": " ".join(shlex.quote(c) for c in job["candidates"]),
        "SHAPES": job["shapes"],
        "DTYPE": job["dtype"],
        "DEVICE": cfg["device"],
        "EXTRA_ARGS": job.get("extra_args", ""),
        "RESULTDIR": job["resultdir"],
        "PYTHON": cfg["python"],
    }
    out = tmpl
    for k, v in subs.items():
        out = out.replace("${" + k + "}", v)
    return out


# --------------------------------------------------------------------------- #
# Result loading (shared by all modes)
# --------------------------------------------------------------------------- #
def _load_result(path: str, ssh_host: str, candidate: str, note: str) -> Dict:
    try:
        if ssh_host:
            cp = _sh(f"cat {shlex.quote(path)}", ssh_host=ssh_host, check=True)
            return json.loads(cp.stdout)
        with open(path) as f:
            return json.loads(f.read())
    except Exception as e:  # noqa: BLE001
        return {
            "candidate": candidate,
            "correctness_passed": False,
            "median_speedup": float("nan"),
            "status": "runner_error",
            "error": f"missing/unreadable result ({note}): {type(e).__name__}: {e}",
            "per_shape": [],
            "errors": [f"runner: {note}"],
        }


# --------------------------------------------------------------------------- #
# local mode
# --------------------------------------------------------------------------- #
def run_local(candidates: List[str], cfg: Dict, shapes: str, dtype: str,
              extra_args: str, resultdir: str, device_override: Optional[str]) -> List[Dict]:
    os.makedirs(resultdir, exist_ok=True)
    device = device_override or cfg["device"]
    # Local mode runs on THIS machine, so use the current interpreter rather than
    # cfg["python"] (which names the cluster's python for sbatch).
    py = sys.executable or cfg["python"]
    results = []
    for i, cand in enumerate(candidates):
        out_path = os.path.join(resultdir, f"result_{i}.json")
        cmd = [
            py, os.path.join(HERE, "bench_harness.py"),
            "--candidate", cand, "--shapes", shapes, "--dtype", dtype,
            "--device", device, "--out", out_path,
        ] + shlex.split(extra_args)
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if os.path.exists(out_path):
            results.append(_load_result(out_path, "", cand, "local"))
        else:
            results.append({
                "candidate": cand, "correctness_passed": False,
                "median_speedup": float("nan"), "status": "runner_error",
                "error": f"bench_harness produced no output (rc={proc.returncode}). stderr tail: {proc.stderr[-500:]}",
                "per_shape": [], "errors": ["runner: no result file"],
            })
    return results


# --------------------------------------------------------------------------- #
# slurm / ssh mode
# --------------------------------------------------------------------------- #
def run_slurm(candidates: List[str], cfg: Dict, shapes: str, dtype: str,
              extra_args: str, job_name: str) -> List[Dict]:
    ssh_host = cfg["ssh_host"] if cfg["mode"] == "ssh" else ""
    workdir = cfg["remote_workdir"] if ssh_host else HERE
    jobdir = f"{workdir}/.runs/{job_name}"
    logdir = f"{jobdir}/logs"
    resultdir = f"{jobdir}/results"

    # Make dirs (remote or local).
    _sh(f"mkdir -p {shlex.quote(logdir)} {shlex.quote(resultdir)}", ssh_host=ssh_host)

    # For ssh mode, ship candidate files to the remote workdir.
    remote_candidates = candidates
    if ssh_host:
        remote_candidates = []
        for i, cand in enumerate(candidates):
            base = os.path.basename(cand)
            dest = f"{workdir}/.runs/{job_name}/{i}_{base}"
            subprocess.run(f"scp {shlex.quote(cand)} {shlex.quote(ssh_host + ':' + dest)}",
                           shell=True, check=True, text=True, capture_output=True)
            remote_candidates.append(dest)

    job = {
        "job_name": job_name, "logdir": logdir, "resultdir": resultdir,
        "workdir": workdir, "candidates": remote_candidates, "shapes": shapes,
        "dtype": dtype, "extra_args": extra_args,
    }
    script = render_sbatch(cfg, job)
    script_path = f"{jobdir}/job.sbatch"
    # Write the script (remote or local).
    if ssh_host:
        _sh(f"cat > {shlex.quote(script_path)} <<'__EOF__'\n{script}\n__EOF__", ssh_host=ssh_host)
    else:
        with open(script_path, "w") as f:
            f.write(script)

    cp = _sh(f"sbatch {shlex.quote(script_path)}", ssh_host=ssh_host)
    job_id = parse_job_id(cp.stdout)

    # Poll until the job leaves the queue.
    deadline = time.monotonic() + cfg["max_wait_s"]
    while True:
        q = _sh(f"squeue -j {job_id} -h -t all", ssh_host=ssh_host, check=False)
        if not q.stdout.strip():
            break
        if time.monotonic() > deadline:
            _sh(f"scancel {job_id}", ssh_host=ssh_host, check=False)
            break
        time.sleep(cfg["poll_interval_s"])

    results = []
    for i, cand in enumerate(candidates):
        rp = f"{resultdir}/result_{i}.json"
        results.append(_load_result(rp, ssh_host, cand, f"slurm task {i} of job {job_id}"))
    return results


def evaluate(candidates: List[str], cfg: Dict, shapes: str, dtype: str,
             extra_args: str, job_name: str, resultdir: str,
             device_override: Optional[str]) -> List[Dict]:
    if cfg["mode"] == "local":
        return run_local(candidates, cfg, shapes, dtype, extra_args, resultdir, device_override)
    if cfg["mode"] in ("slurm", "ssh"):
        return run_slurm(candidates, cfg, shapes, dtype, extra_args, job_name)
    raise SystemExit(f"unknown mode: {cfg['mode']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate candidates on the cluster and emit JSON")
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--shapes", default="official-safe")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--config", default=os.path.join(HERE, "cluster.config.json"))
    p.add_argument("--mode", default=None, help="override config mode (local|slurm|ssh)")
    p.add_argument("--device", default=None, help="override device (mainly for local dry-runs, e.g. cpu)")
    p.add_argument("--job-name", default=None)
    p.add_argument("--extra-args", default="", help="extra flags passed through to bench_harness")
    p.add_argument("--resultdir", default=os.path.join(HERE, ".runs", "local"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode
    # Deterministic, seedable-from-caller job name (no time in the module itself
    # if the caller passes one; else use pid to avoid clashes).
    job_name = args.job_name or f"ars_{os.getpid()}"
    results = evaluate(
        args.candidates, cfg, args.shapes, args.dtype, args.extra_args,
        job_name, args.resultdir, args.device,
    )
    print(json.dumps(results, indent=2))
    ok = all(r.get("correctness_passed") for r in results)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
