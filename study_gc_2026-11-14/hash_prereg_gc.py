"""Study GC registration: write PREREGISTRATION_HASHES_GC.txt (SHA-256 and UTC time of every registered file), or check it.

    python hash_prereg_gc.py            # once, before the full run; refuses to overwrite an existing manifest
    python hash_prereg_gc.py --verify   # any time later

The run log is append-only. At registration it ends with the line "## After hashing"; its digest covers the log up to and including that
line, so later entries can be appended below it. Every other file is hashed in full.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "PREREGISTRATION_HASHES_GC.txt"
FILES = ["PREREGISTRATION_GC.md", "RUN_LOG_GC.md", "envs_gc.py", "run_gc.py", "aggregate_gc.py", "test_envs_gc.py", "reach_gc.py",
         "reach_gc.json", "hash_prereg_gc.py"]
LOG, MARKER = "RUN_LOG_GC.md", b"## After hashing"


def digest(name: str) -> str:
    data = (HERE / name).read_bytes()
    if name == LOG:
        i = data.index(MARKER)
        j = data.find(b"\n", i)
        data = data[: j + 1 if j >= 0 else len(data)]
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        bad = 0
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            h, name = line.split(maxsplit=1)
            ok = digest(name) == h
            bad += not ok
            print(("ok        " if ok else "MISMATCH  ") + name)
        print("all registered files match" if not bad else f"{bad} registered file(s) differ")
        return 1 if bad else 0
    if MANIFEST.exists():
        print(f"{MANIFEST.name} exists; the registration is not overwritten")
        return 1
    if not (HERE / LOG).read_bytes().rstrip().endswith(MARKER):
        print(f"{LOG} must end with the line '{MARKER.decode()}' when it is hashed")
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [f"# Study GC registered files, hashed {stamp}"] + [f"{digest(f)}  {f}" for f in FILES]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
