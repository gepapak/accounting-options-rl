r"""Re-evaluate frozen engine controllers with the step tracer. The evaluation commands are built by the campaign's own
functions (run_scarcity_tail_campaign.py, byte-identical copy), so each run is the audit's run plus the tracer.

Usage:
    python run_engine_twin.py --jobs L3_price_taker_no_solvency__marl__seed7__original [more names] --workers 2
    python run_engine_twin.py --manifest JOBS.txt --workers 4
A job name is variant__agent__seed<seed>__<region>. Completed jobs (trace and completed evaluation JSON) are skipped.
Env logs are about 400 MB per run and are deleted after each run; the JSON and the trace stay.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_scarcity_tail_campaign as camp  # noqa: E402

TRACES = HERE / "traces"
RUNS = HERE / "campaign_runs" / "evals"
LOG = HERE / "campaign_runs" / "run_engine_twin.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse(name: str) -> dict:
    variant, agent, seed, region = name.split("__")
    seed = int(seed.replace("seed", ""))
    return dict(name=name, variant=variant, agent=agent, seed=seed, region=region,
                base_agent=agent, model_dir=camp.model_dir_for(agent, seed))


def done(job: dict) -> bool:
    return (TRACES / f"{job['name']}.npz").is_file() and camp.job_done(dict(job))


def run(job: dict) -> tuple[str, int, int]:
    name = job["name"]
    if done(job):
        return name, 0, 0
    if not (job["model_dir"] / "final_models").is_dir():
        log(f"[MISSING] {name}: no final_models")
        return name, 2, 0
    cmd = camp.base_eval_cmd(job["base_agent"], job["seed"], job["region"], job["model_dir"])
    capon = job["variant"].endswith("_capon")  # the audit's protocol, with the percent payoff's cap really switched on
    cmd, cfg = camp.apply_variant(cmd, job["variant"].removesuffix("_capon"), job["region"])
    if capon:
        cfg = dict(cfg, mtm_apply_price_return_cap=True)
    out = RUNS / name
    cmd = camp.replace_arg(cmd, "--output_dir", str(out))
    assert cmd[1] == "evaluation.py", cmd[:3]
    cmd = [cmd[0], str(HERE / "twin_launcher.py")] + cmd[2:]
    env = dict(os.environ, **camp.THREAD_ENV, SIM_AUDIT_VARIANT=job["variant"], TWIN_TRACE_OUT=str(TRACES / f"{name}.npz"))
    if job["variant"] != "L0_strict":
        env["SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH"] = "1"
    if cfg:
        env["SIM_AUDIT_CFG_OVERRIDES"] = json.dumps(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_job.json").write_text(json.dumps({**{k: str(v) for k, v in job.items()}, "cmd": cmd, "cfg_overrides": cfg}, indent=1))
    t0 = time.time()
    with (out / "run.log").open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=str(camp.ENG), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        for d in glob.glob(str(out / "**" / "env_logs"), recursive=True):
            shutil.rmtree(d, ignore_errors=True)
    return name, rc, int(time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--manifest")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    names = list(a.jobs)
    if a.manifest:
        names += [x.strip() for x in Path(a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    jobs = [parse(n) for n in names]
    todo = [j for j in jobs if not done(j)]
    log(f"{len(jobs)} jobs, {len(jobs) - len(todo)} complete, running {len(todo)} with {a.workers} workers")
    bad = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for name, rc, sec in ex.map(run, todo):
            log(f"  {name}: exit={rc} ({sec}s)")
            bad += int(rc != 0)
    log(f"finished, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
