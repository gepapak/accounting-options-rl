"""Write PREREGISTRATION_HASHES_O.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_O.md", "RUN_LOG_O.md", "envs_o.py", "run_o.py", "aggregate_o.py", "dp_o.py", "calibrate_o.py",
         "dp_o_DK1.json", "dp_o_DK2.json", "calibration_o.json"]
here = Path(__file__).resolve().parent
lines = [f"# Study O registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_O.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
