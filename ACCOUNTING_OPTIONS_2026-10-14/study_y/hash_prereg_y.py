"""Write PREREGISTRATION_HASHES_Y.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_Y.md", "RUN_LOG_Y.md", "run_y.py", "aggregate_y.py", "../study_t/envs_t.py", "../study_t/spr_t.py",
         "../study_x/envs_x.py", "../study_v/rewards_v.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study Y registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_Y.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
