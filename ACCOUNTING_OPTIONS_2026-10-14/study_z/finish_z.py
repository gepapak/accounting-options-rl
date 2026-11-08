"""Study Z finisher (not registered; written at 17:19 on 3 November 2026, after the full run was launched and before any of its runs had
finished). It keeps the computer from idle-sleeping while the detached run is working, waits for run_z.py to exit, then runs the registered
analysis once and the manuscript outputs, and appends what it did to RUN_LOG_Z.md and finish_z.log.

    python finish_z.py --pid 21892

The idle-sleep hold is an ordinary per-process power request (SetThreadExecutionState); it changes no setting and ends with this process.
It does not prevent a sleep that the user chooses, or one caused by closing the lid.
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
N_RUNS = 72
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SYNCHRONIZE, INFINITE = 0x00100000, 0xFFFFFFFF


def now() -> str:
    return datetime.datetime.now().strftime("%H:%M")


def note(msg: str) -> None:
    with open(HERE / "finish_z.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def run_log(text: str) -> None:
    with open(HERE / "RUN_LOG_Z.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def run(script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, script], cwd=HERE, capture_output=True, text=True, encoding="utf-8", env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    a = ap.parse_args()
    k32 = ctypes.windll.kernel32
    k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    note(f"waiting for run_z.py (PID {a.pid}); idle sleep held while waiting")
    h = k32.OpenProcess(SYNCHRONIZE, False, a.pid)
    if h:
        k32.WaitForSingleObject(h, INFINITE)
        k32.CloseHandle(h)
    files = sorted((HERE / "results_z").glob("*.json"))
    last = max((p.stat().st_mtime for p in files), default=0.0)
    last_s = datetime.datetime.fromtimestamp(last).strftime("%H:%M:%S") if files else "-"
    note(f"run_z.py exited; {len(files)} of {N_RUNS} result files (last at {last_s})")
    if len(files) < N_RUNS:
        run_log(f"- **{now()}.** `run_z.py` exited with {len(files)} of {N_RUNS} runs finished (recorded by `finish_z.py`); the registered "
                f"analysis was not run. Resume with `python run_z.py --workers 7` (finished runs are skipped).")
        return 1
    r = run("aggregate_z.py")
    (HERE / "finish_z_aggregate.out").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    verdicts = (HERE / "VERDICTS_Z.md").read_text(encoding="utf-8") if (HERE / "VERDICTS_Z.md").exists() else ""
    n_sup = len(re.findall(r"\| SUPPORTED \|", verdicts))
    n_all = len(re.findall(r"\| (?:SUPPORTED|NOT SUPPORTED|NOT TESTABLE) \|", verdicts))
    note(f"aggregate_z.py exit {r.returncode}; {n_sup} of {n_all} decisions supported")
    m = run("make_studyZ_outputs.py")
    note(f"make_studyZ_outputs.py exit {m.returncode}: {m.stdout.strip()[-300:]} {m.stderr.strip()[-300:]}")
    run_log(f"- **{now()}.** All {N_RUNS} runs finished (the last at {last_s}). The registered analysis (`python aggregate_z.py`) was run once, "
            f"automatically by `finish_z.py` (written at 17:19, before any run had finished), and wrote `runs_z.csv` and `VERDICTS_Z.md`: "
            f"{n_sup} of {n_all} decisions supported. `make_studyZ_outputs.py` (the manuscript's tables and figure) was run afterwards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
