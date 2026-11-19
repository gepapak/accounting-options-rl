"""Study XE, after the run (NOT registered, like verify_completed in Studies GC and DR). Independent checks, by a separate code path,
of the run's completeness and of the decisions computed by aggregate_xe.py:
  1. registered hashes; 2. every expected PPO run present, with the registered steps, actions, seeds, test seed and tag;
  3. XE1 and the forgiveness part of XE2 recomputed from the census; 4. XE5 and XE7 counts recomputed from the raw probes, with exact
  hypergeometric (Fisher) and Mann-Whitney statistics written out by hand; 5. XE6's backtrader equality checked path by path."""
import hashlib
import json
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
problems = []

# 1. hashes
for line in (HERE / "PREREGISTRATION_HASHES_XE.txt").read_text().splitlines()[1:]:
    h, rel = line.split("  ", 1)
    data = (HERE / rel).resolve().read_bytes()
    if rel == "RUN_LOG_XE.md":
        data = data.split(b"## After hashing")[0]
    if hashlib.sha256(data).hexdigest() != h:
        problems.append(f"hash changed: {rel}")
print("1. hashes:", "all match" if not any(p.startswith("hash") for p in problems) else "CHANGED")

# 2. completeness
exp = {("btpy", "_xe5"): range(101, 117), ("bt", "_xe5"): range(101, 117), ("btpy", "_xe7"): range(201, 213), ("bt", "_xe7"): range(201, 213)}
runs = {}
for (eng, tag), seeds in exp.items():
    for s in seeds:
        p = HERE / "results_xe" / f"{eng}__s{s}__500000{tag}.json"
        if not p.exists():
            problems.append(f"missing {p.name}")
            continue
        r = json.loads(p.read_text())
        want_levels = [0.0, 200.0] if tag == "_xe5" else [-1.0, 0.0, 1.0]
        want_test = 20261203 if tag == "_xe5" else 20261204
        if r["steps"] != 507904 or r["levels"] != want_levels or r["test_seed"] != want_test or r["seed"] != s:
            problems.append(f"settings differ in {p.name}")
        if tag == "_xe7" and (r["l_max"] != 1.0 or r["mags"] != "crypto_mags.npy" or r["mag_scale"] != 2.0):
            problems.append(f"direction settings differ in {p.name}")
        runs[(eng, tag, s)] = r
print(f"2. runs present: {len(runs)} of 56; settings as registered: {'yes' if not any('differ' in p for p in problems) else 'NO'}")

# 3. census
c = pd.read_csv(HERE / "xe_census.csv")
low = c[c.L <= 5]
print(f"3. XE1: max |backtesting.py - backtrader| at L <= 5 = {np.abs(low.btpy_rep - low.bt_rep).max():.2e}; "
      f"XE2 part: mean forgiven at L 200 = {100 * (c[c.L == 200].btpy_rep - c[c.L == 200].btpy_book).mean():+.2f} pts, "
      f"min over paths = {100 * (c[c.L == 200].btpy_rep - c[c.L == 200].btpy_book).min():+.2f} (must be >= 0)")
print(f"   ledger at L = 1 (sample drift): {100 * c[c.L == 1].ledger.mean():+.3f}%; expected -fees = -0.040%")

# 4. XE5 and XE7 recomputed
EQLOW = ("0.05", "0.1", "0.25")
LEFT = ("0.25", "0.5", "0.75")
ALLE = ("0.05", "0.1", "0.25", "0.5", "1", "2")


def learner(r):
    vals = [r["probe"][f"{e}|{l}"] for e in EQLOW for l in LEFT]
    return sum(v == 200.0 for v in vals) * 2 >= len(vals)


k = {e: sum(learner(runs[(e, "_xe5", s)]) for s in range(101, 117) if (e, "_xe5", s) in runs) for e in ("btpy", "bt")}
n1 = n2 = 16
K, N = k["btpy"] + k["bt"], n1 + n2
p_f = sum(comb(K, x) * comb(N - K, n1 - x) for x in range(k["btpy"], min(K, n1) + 1)) / comb(N, n1)
print(f"4. XE5: learners backtesting.py {k['btpy']} of 16, backtrader {k['bt']} of 16; exact one-sided hypergeometric p = {p_f:.2e}")


def dir_index(r):
    vals = [r["probe"][f"{e}|{l}"] for e in ALLE for l in LEFT]
    return (sum(v < 0 for v in vals) - sum(v > 0 for v in vals)) / len(vals), sum(v < 0 for v in vals) * 2 >= len(vals), sum(v > 0 for v in vals) * 2 >= len(vals)


d = {e: [dir_index(runs[(e, "_xe7", s)]) for s in range(201, 213) if (e, "_xe7", s) in runs] for e in ("btpy", "bt")}
a, b = [x[0] for x in d["btpy"]], [x[0] for x in d["bt"]]
U = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
print(f"   XE7: backtesting.py short-majority {sum(x[1] for x in d['btpy'])}, long-majority {sum(x[2] for x in d['btpy'])} of 12; "
      f"backtrader short {sum(x[1] for x in d['bt'])}, long {sum(x[2] for x in d['bt'])} of 12; U = {U} of 144 "
      f"(mean direction index {np.mean(a):+.2f} vs {np.mean(b):+.2f})")

# 5. XE6 backtrader equality path by path
worst = 0.0
for (eng, tag, s), r in runs.items():
    if eng == "bt" and tag == "_xe5":
        for cash in ("10000", "1000"):
            worst = max(worst, float(np.max(np.abs(np.array(r["tests"][cash]["draws_reported"]) - np.array(r["tests"][cash]["draws_booked"])))))
print(f"5. XE6: backtrader max |reported - booked| over all test paths = {worst:.1e} points")
print("problems:", problems if problems else "none")
