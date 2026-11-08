"""Write PREREGISTRATION_HASHES_T.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_T.md", "RUN_LOG_T.md", "envs_t.py", "spr_t.py", "orpo_t.py", "run_t.py", "aggregate_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study T registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_T.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
