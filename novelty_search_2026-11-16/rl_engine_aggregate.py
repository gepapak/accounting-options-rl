"""Aggregate the PPO-through-engine pilots (exploratory). Groups runs by engine, budget and variant tag. Per group: the learned leverage
by equity (probe), the low-equity summaries, and the test returns from 10,000 and 1,000 USD, reported against booked, with a bootstrap
interval for reported minus booked over test paths pooled across seeds."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
EQ = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LEFT = (0.25, 0.5, 0.75)
res = [json.loads(p.read_text()) for p in sorted((HERE / "rl_engine_results").glob("*.json"))]
if not res:
    sys.exit("no results yet")
groups = defaultdict(list)
for r in res:
    groups[(r.get("tag", ""), r["steps"], r["engine"])].append(r)
rng = np.random.default_rng(11)
for (tag, steps, eng), rr in sorted(groups.items()):
    levels = rr[0].get("levels", [0, 1, 4, 16, 64, 200])
    top = max(levels)
    print(f"\n=== {'backtesting.py' if eng == 'btpy' else 'backtrader'}{(' [' + tag + ']') if tag else ''}: {len(rr)} seeds, {steps} steps, "
          f"actions {levels}")
    print("learned leverage at a flat account (mean over three episode positions), by equity x 10,000 USD:")
    print("  seed  " + "  ".join(f"{e:>6g}" for e in EQ) + "   low-equity mean / share at the largest level")
    gamblers, shorters, longers = 0, 0, 0
    for r in sorted(rr, key=lambda z: z["seed"]):
        row = [np.mean([r["probe"][f"{e:g}|{l:g}"] for l in LEFT]) for e in EQ]
        allp = [r["probe"][f"{e:g}|{l:g}"] for e in EQ for l in LEFT]
        low = [r["probe"][f"{e:g}|{l:g}"] for e in EQ if e <= 0.25 for l in LEFT]
        share = float(np.mean([x >= top for x in low]))
        gamblers += share >= 0.5
        extra = ""
        if min(levels) < 0:
            s_short, s_long = float(np.mean([x < 0 for x in allp])), float(np.mean([x > 0 for x in allp]))
            shorters += s_short >= 0.5
            longers += s_long >= 0.5
            extra = f"   short {s_short:.2f} / long {s_long:.2f} of all probes"
        print(f"  {r['seed']:>4}  " + "  ".join(f"{x:6.1f}" for x in row) + f"   {np.mean(low):6.1f} / {share:.2f}{extra}")
    if min(levels) < 0:
        print(f"  seeds short at half or more of all probes: {shorters} of {len(rr)}; long: {longers} of {len(rr)}")
    else:
        print(f"  seeds choosing the largest leverage at half or more of the low-equity probes: {gamblers} of {len(rr)}")
    for cash in ("10000", "1000"):
        rep = np.concatenate([r["tests"][cash]["draws_reported"] for r in rr])
        book = np.concatenate([r["tests"][cash]["draws_booked"] for r in rr])
        d = rep - book
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
        per_seed = [(r["tests"][cash]["reported_pct"], r["tests"][cash]["booked_pct"], r["tests"][cash]["hit_zero"]) for r in rr]
        print(f"  tests from {int(cash):,} USD ({len(rep)} paths): reported {rep.mean():+7.2f}%, booked {book.mean():+7.2f}%, "
              f"reported - booked {d.mean():+6.2f} [{np.quantile(bs, .025):+.2f}, {np.quantile(bs, .975):+.2f}]; per seed rep/book/zero: "
              + "; ".join(f"{a:+.1f}/{b:+.1f}/{z:.2f}" for a, b, z in per_seed))
