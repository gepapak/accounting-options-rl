"""Write PREREGISTRATION_HASHES_P.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_P.md", "RUN_LOG_P.md", "run_p.py", "aggregate_p.py", "design_p.py", "design_p.json"]
here = Path(__file__).resolve().parent
lines = [f"# Study P registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_P.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
