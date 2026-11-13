"""Hash the protocol, rules and code of the reward-diagram campaign before the full evaluation."""
import datetime
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["PROTOCOL_SCAN.md", "SCAN_JOBS.txt", "RUN_LOG_SCAN.md", "scan_launcher.py", "run_scan.py", "analyze_scan.py", "reward_diagram.py", "../engine_twin_2026-11-05/twin_trace.py", "../engine_twin_2026-11-05/run_scarcity_tail_campaign.py"]
lines = [f"# hashed {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} (UTC), before the full evaluation"]
for f in FILES:
    lines.append(f"{hashlib.sha256((HERE / f).read_bytes()).hexdigest()}  {f}")
(HERE / "PROTOCOL_SCAN_HASHES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
