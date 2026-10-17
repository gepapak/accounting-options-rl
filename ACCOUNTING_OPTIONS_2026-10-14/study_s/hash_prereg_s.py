"""Write PREREGISTRATION_HASHES_S.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_S.md", "RUN_LOG_S.md", "envs_s.py", "spr_s.py", "run_s.py", "aggregate_s.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study S registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_S.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
