"""Write PREREGISTRATION_HASHES_Z.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_Z.md", "THEORY_Z.md", "RUN_LOG_Z.md", "envs_z.py", "rewards_z.py", "run_z.py", "aggregate_z.py", "test_envs_z.py",
         "witness_z.py", "witness_z.json", "../study_t/envs_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study Z registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_Z.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
