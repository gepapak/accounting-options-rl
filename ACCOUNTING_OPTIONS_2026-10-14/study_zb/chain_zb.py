"""Study Z, Part B chain (registered with PREREGISTRATION_ZB.md): waits for Study Z's run to finish, then runs Part B's registered full run,
then its registered analysis once and the manuscript outputs, holding the computer's idle sleep throughout.

    python chain_zb.py --after-pid 21892

The idle-sleep hold is an ordinary per-process power request (SetThreadExecutionState); it changes no setting and ends with this process.
Everything it does is appended to RUN_LOG_ZB.md and chain_zb.log.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_RUNS = 64
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, INFINITE = 0x00100000, 0xFFFFFFFF


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M")


def note(msg: str) -> None:
    with open(HERE / "chain_zb.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def run_log(text: str) -> None:
    with open(HERE / "RUN_LOG_ZB.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-pid", type=int, required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    note(f"waiting for PID {a.after_pid} (Study Z's run); idle sleep held")
    h = k32.OpenProcess(SYNCHRONIZE, False, a.after_pid)
    if h:
        k32.WaitForSingleObject(h, INFINITE)
        k32.CloseHandle(h)
    note("Study Z's run has exited; launching run_zb.py")
    run_log(f"- **{now()}.** Study Z's run had exited; `chain_zb.py` launched the full run (`run_zb.py --workers {a.workers}`, `run_full.log`): "
            f"{N_RUNS} runs (seeds 220-227). As in Studies X to Z, `run_full.log` prints each run's booked returns when the run finishes; those "
            f"lines may be read to follow progress. The registered analysis (`aggregate_zb.py`) is run once, after all runs have finished.")
    with open(HERE / "run_full.log", "a", encoding="utf-8") as out, open(HERE / "run_full.err", "a", encoding="utf-8") as err:
        p = subprocess.run([sys.executable, "run_zb.py", "--workers", str(a.workers)], cwd=HERE, stdout=out, stderr=err, env=env)
    files = sorted((HERE / "results_zb").glob("*.json"))
    last = max((f.stat().st_mtime for f in files), default=0.0)
    last_s = datetime.datetime.fromtimestamp(last).strftime("%H:%M:%S") if files else "-"
    note(f"run_zb.py exit {p.returncode}; {len(files)} of {N_RUNS} result files (last at {last_s})")
    if len(files) < N_RUNS:
        run_log(f"- **{now()}.** `run_zb.py` exited with {len(files)} of {N_RUNS} runs finished; the registered analysis was not run. Resume with "
                f"`python run_zb.py --workers 7` (finished runs are skipped).")
        return 1
    r = subprocess.run([sys.executable, "aggregate_zb.py"], cwd=HERE, capture_output=True, text=True, encoding="utf-8", env=env)
    (HERE / "chain_zb_aggregate.out").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    verdicts = (HERE / "VERDICTS_ZB.md").read_text(encoding="utf-8") if (HERE / "VERDICTS_ZB.md").exists() else ""
    n_sup = len(re.findall(r"\| SUPPORTED \|", verdicts))
    n_all = len(re.findall(r"\| (?:SUPPORTED|NOT SUPPORTED|NOT TESTABLE) \|", verdicts))
    note(f"aggregate_zb.py exit {r.returncode}; {n_sup} of {n_all} supported")
    extra = ""
    if (HERE / "make_studyZB_outputs.py").exists():
        m = subprocess.run([sys.executable, "make_studyZB_outputs.py"], cwd=HERE, capture_output=True, text=True, encoding="utf-8", env=env)
        note(f"make_studyZB_outputs.py exit {m.returncode}: {m.stdout.strip()[-300:]} {m.stderr.strip()[-300:]}")
        extra = " `make_studyZB_outputs.py` (the manuscript's tables and figure) was run afterwards."
    run_log(f"- **{now()}.** All {N_RUNS} runs finished (the last at {last_s}). The registered analysis (`python aggregate_zb.py`) was run once, "
            f"automatically by `chain_zb.py`, and wrote `runs_zb.csv` and `VERDICTS_ZB.md`: {n_sup} of {n_all} decisions supported.{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
