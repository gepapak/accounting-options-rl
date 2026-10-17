"""Write PREREGISTRATION_HASHES_Q.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_Q.md", "RUN_LOG_Q.md", "run_q.py", "aggregate_q.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study Q registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_Q.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
