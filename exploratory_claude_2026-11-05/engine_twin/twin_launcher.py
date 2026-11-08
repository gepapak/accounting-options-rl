"""Run the audit engine's unmodified evaluation.py with the read-only step tracer installed. cwd must be the engine folder.
Usage: TWIN_TRACE_OUT=<file.npz> python twin_launcher.py <evaluation.py arguments>"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENG = Path.cwd()
assert (ENG / "evaluation.py").is_file(), "run from the engine folder"
sys.path.insert(0, str(HERE))
import twin_trace  # noqa: E402

sys.path[0] = str(ENG)  # as if evaluation.py had been started directly from the engine folder
sys.path.append(str(HERE))
import environment  # noqa: E402

twin_trace.install(environment.RenewableMultiAgentEnv, os.environ["TWIN_TRACE_OUT"])
sys.argv = ["evaluation.py"] + sys.argv[1:]
runpy.run_path(str(ENG / "evaluation.py"), run_name="__main__")
