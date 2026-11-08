"""Run a registered Study RS script from this repository without editing it.

Study RS was designed and run in Paper1/study_rs_2026-11-04, next to the github folder, and its hashed scripts look for the reused code at
../github/accounting-options-rl/. This wrapper puts this repository's copies on PYTHONPATH (inherited by the worker processes of run_rs.py) and
runs the script unchanged, in a child process, with bytecode writing off:

    python portable_rs.py test_envs_rs.py                    # exact checks of the environment (prints only)
    python portable_rs.py aggregate_rs.py --print            # the registered analysis, printed; writes nothing
    python portable_rs.py witness_rs.py                      # the myopic witness; rewrites witness_rs.json: disposable copy only
    python portable_rs.py run_rs.py --workers 8              # retrains the 80 runs: disposable copy only (finished runs are skipped)

hash_prereg_rs.py --verify and verify_completed_rs.py read the original layout and do not run here; tools/verify_studies.py checks the
registered hashes and the recount from this location.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACC = HERE.parent
REPO = ACC.parent
PATHS = [ACC / "study_t", ACC / "study_v", REPO / "LIABILITY_STUDY_2026-09-29", REPO / "GENERALIZATION_STUDY_2026-09-29", HERE]

if __name__ == "__main__":
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in PATHS] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    sys.exit(subprocess.run([sys.executable, "-B"] + sys.argv[1:], cwd=HERE, env=env).returncode)
