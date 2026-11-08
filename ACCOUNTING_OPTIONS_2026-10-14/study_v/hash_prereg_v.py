"""Write PREREGISTRATION_HASHES_V.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_V.md", "RUN_LOG_V.md", "rewards_v.py", "run_v.py", "aggregate_v.py", "../study_t/envs_t.py", "../study_t/spr_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study V registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_V.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
