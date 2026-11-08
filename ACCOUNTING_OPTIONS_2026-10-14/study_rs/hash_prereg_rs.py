"""Study RS registration hashes.

    python hash_prereg_rs.py           # writes PREREGISTRATION_HASHES_RS.txt (refuses to overwrite)
    python hash_prereg_rs.py --verify  # recomputes every hash and compares; prints only

RUN_LOG_RS.md is hashed up to and including the line "## After hashing (not part of the registered prefix)"; text appended after it is not
part of the registration.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "## After hashing (not part of the registered prefix)"
FILES = ["PREREGISTRATION_RS.md", "RUN_LOG_RS.md", "envs_rs.py", "run_rs.py", "aggregate_rs.py", "test_envs_rs.py", "witness_rs.py",
         "witness_rs.json", "hash_prereg_rs.py",
         "../github/accounting-options-rl/ACCOUNTING_OPTIONS_2026-10-14/study_t/envs_t.py",
         "../github/accounting-options-rl/ACCOUNTING_OPTIONS_2026-10-14/study_t/spr_t.py",
         "../github/accounting-options-rl/ACCOUNTING_OPTIONS_2026-10-14/study_v/rewards_v.py",
         "../github/accounting-options-rl/LIABILITY_STUDY_2026-09-29/envs_b.py",
         "../github/accounting-options-rl/LIABILITY_STUDY_2026-09-29/run_b.py",
         "../github/accounting-options-rl/GENERALIZATION_STUDY_2026-09-29/envs.py",
         "../github/accounting-options-rl/GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK1.pkl",
         "../github/accounting-options-rl/GENERALIZATION_STUDY_2026-09-29/data_panel/panel_DK2.pkl"]
OUT = HERE / "PREREGISTRATION_HASHES_RS.txt"


def digest(rel: str) -> str:
    data = (HERE / rel).read_bytes()
    if rel == "RUN_LOG_RS.md":
        i = data.find(MARK.encode("utf-8"))
        assert i >= 0, f"{rel}: marker line missing"
        j = data.find(b"\n", i)
        data = data[: (j + 1) if j >= 0 else len(data)]
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        bad = 0
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            h, rel = line.split("  ", 1)
            ok = digest(rel) == h
            bad += not ok
            print(("ok      " if ok else "CHANGED ") + rel)
        print("all registered files unchanged" if not bad else f"{bad} file(s) changed")
        return int(bad > 0)
    if OUT.exists():
        print(f"{OUT.name} exists; not overwritten")
        return 1
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines = [f"# Study RS registered files, hashed {stamp}"] + [f"{digest(rel)}  {rel}" for rel in FILES]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
