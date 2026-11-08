"""Write PREREGISTRATION_HASHES_U.txt: SHA-256 and UTC time of every registered file (paths relative to this folder)."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_U.md", "RUN_LOG_U.md", "run_u.py", "aggregate_u.py", "../study_t/envs_tn.py", "../study_t/spr_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study U registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_U.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
