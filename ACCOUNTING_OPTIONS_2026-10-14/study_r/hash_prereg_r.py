"""Freeze Study R once; refuse an existing manifest or any full-training output."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
manifest = HERE / "PREREGISTRATION_HASHES_R.txt"
if manifest.exists() or list((HERE / "results_r/jobs").glob("*.json")):
    raise RuntimeError("Registration/output already exists; never replace it")
files = {HERE / name for name in ("PREREGISTRATION_R.md", "RUN_LOG_R.md", "envs_r.py", "run_r.py",
         "aggregate_r.py", "predict_r.py", "predictions_r.json", "validate_r.py", "validation_r.json", "hash_prereg_r.py")}
for relative in json.loads((HERE / "predictions_r.json").read_text())["input_sha256"]:
    files.add(ROOT / relative)
files.update(ROOT / relative for relative in ("LIABILITY_STUDY_2026-09-29/envs_b.py",
             "LIABILITY_STUDY_2026-09-29/run_b.py", "GENERALIZATION_STUDY_2026-09-29/envs.py"))
files.update((HERE / "development_smoke/jobs").glob("*.json"))
log = (HERE / "RUN_LOG_R.md").read_bytes()
marker = b"## After hashing"
index = log.index(marker) + len(marker)
prefix_length = index
while prefix_length < len(log) and log[prefix_length:prefix_length + 1] in (b"\r", b"\n"):
    prefix_length += 1
lines = [f"# Study R frozen UTC {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
         f"# RUN_LOG_PREFIX_BYTES {prefix_length}"]
for file in sorted(files):
    content = file.read_bytes()
    if file.name == "RUN_LOG_R.md":
        content = content[:prefix_length]
    lines.append(f"{hashlib.sha256(content).hexdigest()}  {file.relative_to(ROOT).as_posix()}")
manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
