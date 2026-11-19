"""Write (default) or verify (--verify) the SHA-256 hashes of Study XE's registered files. RUN_LOG_XE.md is hashed only up to its line
'## After hashing'. Writing refuses to overwrite an existing hash file."""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["PREREGISTRATION_XE.md", "run_xe.ps1", "hash_prereg_xe.py", "aggregate_xe.py", "engines_pilot.py", "rl_engine_env.py",
         "rl_engine_pilot.py", "reading_engines.py", "census_mtsim.py", "eurusd_mags.npy", "crypto_mags.npy",
         "../study_gc_2026-11-14/envs_gc.py", "RUN_LOG_XE.md"]
OUT = HERE / "PREREGISTRATION_HASHES_XE.txt"


def digest(rel):
    p = (HERE / rel).resolve()
    data = p.read_bytes()
    if rel == "RUN_LOG_XE.md":
        data = data.split(b"## After hashing")[0]
    return hashlib.sha256(data).hexdigest()


def main():
    if "--verify" in sys.argv:
        bad = 0
        for line in OUT.read_text().splitlines()[1:]:
            h, rel = line.split("  ", 1)
            ok = digest(rel) == h
            bad += not ok
            print(f"{'ok' if ok else 'CHANGED':<8}  {rel}")
        print("all registered files match" if not bad else f"{bad} file(s) changed")
        sys.exit(1 if bad else 0)
    if OUT.exists():
        sys.exit(f"{OUT.name} exists; refusing to overwrite (use --verify)")
    lines = [f"# Study XE registered files, hashed {datetime.now(timezone.utc).isoformat(timespec='seconds')}"]
    lines += [f"{digest(f)}  {f}" for f in FILES]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
