"""Run a registered Study N script on another machine without editing it.

The registered scripts (hashes in PREREGISTRATION_HASHES_N.txt) hard-code the analysis machine's path to
LIABILITY_STUDY_2026-09-29. This wrapper puts this repository's copy on PYTHONPATH (inherited by the worker
processes of run_n.py) and then runs the script unchanged:

    python portable.py run_n.py --workers 7        # do not use python -I here: workers must inherit PYTHONPATH
    python portable.py aggregate_n.py
    python portable.py dp_n2.py --zone DK1
"""
import os, runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
os.environ["PYTHONPATH"] = os.pathsep.join([str(LIAB), str(HERE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
sys.path[:0] = [str(LIAB), str(HERE)]
os.chdir(HERE)
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(str(HERE / script), run_name="__main__")
