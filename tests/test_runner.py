"""Unit tests for runner.py. Local mode runs on CPU; slurm bits are tested by
parsing/rendering only (no real scheduler here).

Run under pytest, or standalone: `python tests/test_runner.py`.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import runner  # noqa: E402

IDENTITY = os.path.join(HERE, "fixtures", "identity_candidate.py")
BROKEN = os.path.join(HERE, "fixtures", "broken_candidate.py")
SDPA = os.path.join(ROOT, "candidates", "best.py")


def test_parse_job_id():
    assert runner.parse_job_id("Submitted batch job 987654\n") == "987654"
    try:
        runner.parse_job_id("nope")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_render_sbatch_substitutes_everything():
    cfg = dict(runner._DEFAULTS)
    cfg.update(partition="gpu", account="myacct", gres="gpu:a100:2", time="00:05:00",
               module_load="module load cuda", python="python", device="cuda")
    job = dict(job_name="jobx", logdir="/w/logs", resultdir="/w/res", workdir="/w",
               candidates=["/w/a.py", "/w/b.py"], shapes="dev", dtype="float32", extra_args="")
    script = runner.render_sbatch(cfg, job)
    # Our template placeholders must all be substituted...
    for ph in ("JOB_NAME", "PARTITION_LINE", "ACCOUNT_LINE", "GRES", "TIME", "LOGDIR",
               "ARRAY_MAX", "MODULE_LOAD", "WORKDIR", "CANDIDATE_LIST", "SHAPES",
               "DTYPE", "DEVICE", "EXTRA_ARGS", "RESULTDIR", "PYTHON"):
        assert "${" + ph + "}" not in script, f"unsubstituted placeholder ${{{ph}}}"
    # ...but bash runtime variables must survive for Slurm to expand at run time.
    assert "${SLURM_ARRAY_TASK_ID}" in script
    assert "--array=0-1" in script
    assert "#SBATCH --account=myacct" in script
    assert "--partition=gpu" in script
    assert "/w/a.py" in script and "/w/b.py" in script
    # empty account/partition -> those lines are omitted entirely
    cfg["account"] = ""
    cfg["partition"] = ""
    script2 = runner.render_sbatch(cfg, job)
    assert "--account=" not in script2
    assert "--partition=" not in script2


def test_local_mode_two_candidates():
    cfg = dict(runner._DEFAULTS)
    cfg["mode"] = "local"
    results = runner.evaluate(
        [IDENTITY, SDPA], cfg, shapes="dev", dtype="float32", extra_args="",
        job_name="t", resultdir=os.path.join(ROOT, ".runs", "test"),
        device_override="cpu",
    )
    assert len(results) == 2
    assert all(r["correctness_passed"] for r in results), results


def test_local_mode_broken_candidate_flows_through():
    cfg = dict(runner._DEFAULTS)
    cfg["mode"] = "local"
    results = runner.evaluate(
        [BROKEN], cfg, shapes="dev", dtype="float32", extra_args="",
        job_name="t", resultdir=os.path.join(ROOT, ".runs", "test"),
        device_override="cpu",
    )
    assert len(results) == 1
    assert results[0]["correctness_passed"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    sys.exit(1 if failures else 0)
