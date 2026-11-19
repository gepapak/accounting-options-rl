"""Study XE decisions XE1 to XE7, as registered. Reads the study's own outputs and writes VERDICTS_XE.md.

Inputs (all in this folder):
  xe_census.csv          engines_pilot.py 2000 <census seed> xe_census.csv
  xe_census_mtsim.csv    census_mtsim.py 300 <mtsim seed> xe_census_mtsim.csv      (venv-mtsim)
  reading_xe.json        reading_engines.py
  results_xe/*_xe5.json  rl_engine_pilot.py, two actions {0, 200}, 16 seeds per engine
  results_xe/*_xe7.json  rl_engine_pilot.py, actions {-1, 0, 1}, margin 1, crypto x2, 12 seeds per engine
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
EQ = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LEFT = (0.25, 0.5, 0.75)
B = 4000
rng = np.random.default_rng(20261205)
lines, verdicts = [], {}


def say(s):
    print(s)
    lines.append(s)


def mat(df, col, levs):
    return np.stack([df[df.L == L].sort_values("path")[col].to_numpy() for L in levs])


# ---------------- XE1 to XE3: the trade census ----------------
c = pd.read_csv(HERE / "xe_census.csv")
levs = sorted(c.L.unique())
P = c.path.nunique()
lowL = [L for L in levs if L <= 5]
d_low = max(float(np.abs(c[c.L == L].btpy_rep - c[c.L == L].bt_rep).max()) for L in lowL)
verdicts["XE1"] = d_low < 1e-9
say(f"XE1 agreement inside the range: max |backtesting.py - backtrader| over L <= 5 and {P} paths = {d_low:.3e} (need < 1e-9) -> "
    f"{'SUPPORTED' if verdicts['XE1'] else 'NOT SUPPORTED'}")

g = c[c.L == 200].sort_values("path")
idx = rng.integers(0, len(g), size=(B, len(g)))
d1 = (g.btpy_rep - g.bt_rep).to_numpy()
d2 = (g.btpy_rep - g.btpy_book).to_numpy()
lb1, lb2 = np.quantile(d1[idx].mean(axis=1), 0.025), np.quantile(d2[idx].mean(axis=1), 0.025)
ledger_err = float(np.abs(g.bt_rep - g.ledger).max())
verdicts["XE2"] = lb1 > 0 and lb2 > 0 and ledger_err < 1e-8
say(f"XE2 divergence past the kink (L = 200): backtesting.py - backtrader {100 * d1.mean():+.2f} pts (lower bound {100 * lb1:+.2f}); "
    f"reported - own trade records {100 * d2.mean():+.2f} (lower bound {100 * lb2:+.2f}); max |backtrader - ledger| {ledger_err:.1e} -> "
    f"{'SUPPORTED' if verdicts['XE2'] else 'NOT SUPPORTED'}")

m_btpy, m_bt = mat(c, "btpy_rep", levs), mat(c, "bt_rep", levs)
idx = rng.integers(0, P, size=(B, P))
pick_btpy = np.array(levs)[m_btpy[:, idx].mean(axis=2).argmax(axis=0)]
pick_bt = np.array(levs)[m_bt[:, idx].mean(axis=2).argmax(axis=0)]
s1, s2 = float(np.mean(pick_btpy == 200)), float(np.mean(pick_bt != 200))
verdicts["XE3"] = s1 >= 0.95 and s2 >= 0.60
say(f"XE3 optimizer choice: backtesting.py picks L = 200 in {s1:.1%} of resamples (need >= 95%); backtrader does not pick 200 in {s2:.1%} "
    f"(need >= 60%) -> {'SUPPORTED' if verdicts['XE3'] else 'NOT SUPPORTED'}")

# ---------------- XE4: the reading ----------------
rd = json.loads((HERE / "reading_xe.json").read_text())
CASH = 10_000.0
ok = True
for r in rd:
    if "error" in r:
        ok = False
        continue
    past = r["query"].startswith("past")
    forgiving = r["engine"].startswith("backtesting.py")
    if past and forgiving:
        ok &= r["forgiven"] > 1e-6 * CASH
    else:
        ok &= abs(r["forgiven"]) < 1e-6 * CASH
mt = pd.read_csv(HERE / "xe_census_mtsim.csv")
m200 = mt[mt.L == 200]
orig_forgiven = float((m200[m200.cond == "ORIG"].rep - m200[m200.cond == "ORIG"].book).mean())
fl_err = float((m200[m200.cond == "FLPATCH"].rep - m200[m200.cond == "FLPATCH"].book).abs().max())
mt_err_all = float((mt[mt.cond == "FLPATCH"].rep - mt[mt.cond == "FLPATCH"].book).abs().max())
verdicts["XE4"] = ok and orig_forgiven > 0 and mt_err_all < 1e-9
say(f"XE4 reading: one-query classification {'correct' if ok else 'WRONG'} for backtesting.py (levered and unlevered short), backtrader, "
    f"vectorbt; gym-mtsim as published forgives {100 * orig_forgiven:+.2f} pts at L = 200, floor-deleted max |reported - booked| "
    f"{mt_err_all:.1e} -> {'SUPPORTED' if verdicts['XE4'] else 'NOT SUPPORTED'}")


# ---------------- XE5, XE6: PPO through the engines, two actions ----------------
def load(tag):
    out = {}
    for p in sorted((HERE / "results_xe").glob(f"*{tag}.json")):
        r = json.loads(p.read_text())
        out.setdefault(r["engine"], []).append(r)
    return out


res5 = load("_xe5")
learners = {}
for eng, rr in res5.items():
    learners[eng] = sum(np.mean([r["probe"][f"{e:g}|{l:g}"] >= max(r["levels"]) for e in EQ if e <= 0.25 for l in LEFT]) >= 0.5 for r in rr)
nb, nt = len(res5.get("btpy", [])), len(res5.get("bt", []))
kb, kt = learners.get("btpy", 0), learners.get("bt", 0)
p5 = fisher_exact([[kb, nb - kb], [kt, nt - kt]], alternative="greater")[1] if nb and nt else float("nan")
verdicts["XE5"] = nb == 16 and nt == 16 and kb >= 9 and kt <= 1 and p5 < 0.01
say(f"XE5 PPO learns the engine (two actions): backtesting.py {kb} of {nb}, backtrader {kt} of {nt}, Fisher one-sided p = {p5:.2e} "
    f"(need >= 9 of 16, <= 1 of 16, p < 0.01) -> {'SUPPORTED' if verdicts['XE5'] else 'NOT SUPPORTED'}")

ok6 = True
for cash in ("10000", "1000"):
    rb = res5.get("btpy", [])
    seed_gaps = np.array([np.mean(np.array(r["tests"][cash]["draws_reported"]) - np.array(r["tests"][cash]["draws_booked"])) for r in rb])
    if len(seed_gaps):
        bs = seed_gaps[rng.integers(0, len(seed_gaps), size=(B, len(seed_gaps)))].mean(axis=1)
        lb = float(np.quantile(bs, 0.025))
    else:
        lb = float("nan")
    bt_err = max([float(np.max(np.abs(np.array(r["tests"][cash]["draws_reported"]) - np.array(r["tests"][cash]["draws_booked"]))))
                  for r in res5.get("bt", [])] or [float("nan")])
    ok6 &= lb > 0 and bt_err < 1e-6
    say(f"   XE6 from {int(cash):,} USD: backtesting.py seed-mean reported - booked {seed_gaps.mean() if len(seed_gaps) else float('nan'):+.2f} pts "
        f"(seed-cluster lower bound {lb:+.2f}); backtrader max |reported - booked| {bt_err:.1e} points")
verdicts["XE6"] = ok6
say(f"XE6 reported against booked for the learned policies -> {'SUPPORTED' if ok6 else 'NOT SUPPORTED'}")

# ---------------- XE7: direction at the default settings ----------------
res7 = load("_xe7")
idx_dir, maj = {}, {}
for eng, rr in res7.items():
    vals, shorts, longs = [], 0, 0
    for r in rr:
        allp = [r["probe"][f"{e:g}|{l:g}"] for e in EQ for l in LEFT]
        s, lg = float(np.mean([x < 0 for x in allp])), float(np.mean([x > 0 for x in allp]))
        vals.append(s - lg)
        shorts += s >= 0.5
        longs += lg >= 0.5
    idx_dir[eng], maj[eng] = np.array(vals), (shorts, longs, len(rr))
if "btpy" in idx_dir and "bt" in idx_dir:
    p7 = mannwhitneyu(idx_dir["btpy"], idx_dir["bt"], alternative="greater").pvalue
    sb, lb_, nb7 = maj["btpy"]
    st, lt, nt7 = maj["bt"]
    verdicts["XE7"] = nb7 == 12 and nt7 == 12 and sb >= 7 and lb_ <= 1 and p7 < 0.05
    say(f"XE7 direction chosen by the accounting: backtesting.py short-majority {sb} of {nb7}, long-majority {lb_}; backtrader short {st}, "
        f"long {lt} of {nt7}; direction index MW one-sided p = {p7:.4f} (need >= 7 short, <= 1 long, p < 0.05) -> "
        f"{'SUPPORTED' if verdicts['XE7'] else 'NOT SUPPORTED'}")
else:
    verdicts["XE7"] = False
    say("XE7: results missing -> NOT SUPPORTED")

say("")
say("Supported: " + ", ".join(k for k, v in verdicts.items() if v) + f" ({sum(verdicts.values())} of {len(verdicts)})")
(HERE / "VERDICTS_XE.md").write_text("# Study XE verdicts\n\n" + "\n".join(f"- {s}" if s else "" for s in lines) + "\n", encoding="utf-8")
