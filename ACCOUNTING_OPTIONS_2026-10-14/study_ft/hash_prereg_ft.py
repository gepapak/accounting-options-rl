"""Write PREREGISTRATION_HASHES_FT.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_FT.md", "RUN_LOG_FT.md", "envs_ft.py", "rewards_ft.py", "run_ft.py", "aggregate_ft.py", "test_envs_ft.py", "dp_lambda.py", "dp_lambda.json", "../study_t/envs_t.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study FT registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_FT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
