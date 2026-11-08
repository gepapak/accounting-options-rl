"""Write PREREGISTRATION_HASHES_W.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_W.md", "RUN_LOG_W.md", "aggregate_w.py", "../study_v/run_v.py", "../study_v/rewards_v.py", "../study_t/envs_t.py",
         "../study_t/spr_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study W registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_W.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
