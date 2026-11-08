"""Write PREREGISTRATION_HASHES_ZB.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_ZB.md", "THEORY_ZB.md", "RUN_LOG_ZB.md", "envs_zb.py", "rewards_zb.py", "run_zb.py", "aggregate_zb.py",
         "test_envs_zb.py", "witness_zb.py", "witness_zb.json", "chain_zb.py", "../study_t/envs_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study Z, Part B registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_ZB.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
