"""Write PREREGISTRATION_HASHES_EXT.txt: SHA-256 and UTC time of every registered file."""
import datetime
import hashlib
from pathlib import Path

FILES = ["PREREGISTRATION_EXT.md", "RUN_LOG_EXT.md", "envs_ext.py", "run_ext.py", "scan_ext.py", "aggregate_ext.py", "test_envs_ext.py", "scripted_ext.py", "scripted_ext2.py",
         "vendor/SOURCE_MANIFEST.json", "vendor/LICENSE.txt", "vendor/gym_trading_env/__init__.py", "vendor/gym_trading_env/environments.py", "vendor/gym_trading_env/utils/history.py", "vendor/gym_trading_env/utils/portfolio.py"]
here = Path(__file__).resolve().parent
lines = [f"# Study EXT registered files, hashed {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}"]
for f in FILES:
    lines.append(f"{hashlib.sha256((here / f).read_bytes()).hexdigest()}  {f}")
(here / "PREREGISTRATION_HASHES_EXT.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
