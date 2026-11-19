r"""Run audit evaluations with the reward-diagram reading on real clones. Commands come from the campaign's own functions, as in the engine twin check.
Usage: python run_scan.py [--workers 3] [--out scan_campaign] [--manifest SCAN_JOBS.txt | job names (variant__agent__seed<seed>__<region>)]"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent / "engine_twin_2026-11-05"
sys.path.insert(0, str(WORK))
import run_scarcity_tail_campaign as camp  # noqa: E402


def run(args):
    name, outdir = args
    variant, agent, seed, region = name.split("__")
    seed = int(seed.replace("seed", ""))
    scan = outdir / "scans" / f"{name}.npz"
    if scan.is_file():
        return name, 0, 0
    model_dir = camp.model_dir_for(agent, seed)
    cmd = camp.base_eval_cmd(agent, seed, region, model_dir)
    capon = variant.endswith("_capon")
    cmd, cfg = camp.apply_variant(cmd, variant.removesuffix("_capon"), region)
    if capon:
        cfg = dict(cfg, mtm_apply_price_return_cap=True)
    out = outdir / "runs" / name
    cmd = camp.replace_arg(cmd, "--output_dir", str(out))
    assert cmd[1] == "evaluation.py"
    cmd = [cmd[0], "-B", str(HERE / "scan_launcher.py")] + cmd[2:]
    env = dict(os.environ, **camp.THREAD_ENV, SIM_AUDIT_VARIANT=variant, TWIN_TRACE_OUT=str(outdir / "traces" / f"{name}.npz"), SCAN_OUT=str(scan))
    if variant != "L0_strict":
        env["SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH"] = "1"
    if cfg:
        env["SIM_AUDIT_CFG_OVERRIDES"] = json.dumps(cfg)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with (out / "run.log").open("w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=str(camp.ENG), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        for d in glob.glob(str(out / "**" / "env_logs"), recursive=True):
            shutil.rmtree(d, ignore_errors=True)
    return name, rc, int(time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="scan_campaign")
    ap.add_argument("--manifest")
    ap.add_argument("jobs", nargs="*")
    a = ap.parse_args()
    names = list(a.jobs)
    if a.manifest:
        names += [x.strip() for x in (HERE / a.manifest).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")]
    outdir = HERE / a.out
    (outdir / "scans").mkdir(parents=True, exist_ok=True)
    (outdir / "traces").mkdir(exist_ok=True)
    print(time.strftime("%Y-%m-%d %H:%M:%S"), f"{len(names)} jobs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for name, rc, sec in ex.map(run, [(n, outdir) for n in names]):
            print(time.strftime("%H:%M:%S"), f"{name}: exit={rc} ({sec}s)", flush=True)


if __name__ == "__main__":
    main()
