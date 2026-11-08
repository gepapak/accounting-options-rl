"""Write PREREGISTRATION_HASHES_RE.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_RE.md", "RUN_LOG_RE.md", "envs_re.py", "reach_re.py", "reach_re.json", "run_re.py", "aggregate_re.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study RE registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_RE.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
