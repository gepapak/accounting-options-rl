"""Write PREREGISTRATION_HASHES_N.txt: SHA-256 and UTC time of every registered file."""
import hashlib, datetime
from pathlib import Path
FILES = ["PREREGISTRATION_N.md", "RUN_LOG_N.md", "envs_n.py", "run_n.py", "aggregate_n.py", "dp_n.py", "dp_n_fee.py", "dp_n2.py",
         "dp_n2_DK1.json", "dp_n2_DK2.json", "dp_n_DK1.json", "dp_n_DK2.json"]
here = Path(__file__).resolve().parent
lines = [f"# Study N registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_N.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
