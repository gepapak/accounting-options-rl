"""Scores the same-engine control (results_ctl/) against the four predictions written in README.md before the run (06:29, 17 Nov).
The learner and gap definitions are copied from the registered Study XE aggregator (aggregate_xe.py, XE5 and XE6).
Everything after the four predictions is labelled post hoc."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, fisher_exact

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
XE = HERE.parent / "study_xe_2026-11-16"
EQ = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
LEFT = (0.25, 0.5, 0.75)
SEEDS = list(range(301, 309))
EXPECT = dict(steps=507904, levels=[0.0, 200.0], tag="_ctl", l_max=200.0, mags="eurusd_mags.npy", mag_scale=1.0, test_seed=20261211)

res = {}
for p in sorted((HERE / "results_ctl").glob("*_ctl.json")):
    r = json.loads(p.read_text())
    bad = {k: r[k] for k, v in EXPECT.items() if r[k] != v}
    assert not bad, (p.name, bad)
    res.setdefault(r["engine"], {})[r["seed"]] = r
assert all(sorted(res[e]) == SEEDS for e in ("btpy", "btpyfl")), "missing runs"


def share(r):          # XE5: share of the nine low-equity probes (equity <= 0.25) at the largest level
    return float(np.mean([r["probe"][f"{e:g}|{l:g}"] >= max(r["levels"]) for e in EQ if e <= 0.25 for l in LEFT]))


def grid(r):           # all 18 probes, equity rising left to right, three time points each: '#' = 200x, '.' = flat
    return " ".join("".join("#" if r["probe"][f"{e:g}|{l:g}"] >= 200 else "." for l in LEFT) for e in EQ)


def gap(r, cash):
    t = r["tests"][cash]
    return np.array(t["draws_reported"]) - np.array(t["draws_booked"])


print("Probes: equity 0.05 0.1 0.25 0.5 1 2 (three time points each); '#' = 200x, '.' = flat")
print(f"{'seed':>4} | {'published engine':<26} {'share':>5} {'gap 10k':>8} {'gap 1k':>7} | {'floor removed':<26} {'share':>5} {'max|rep-book|':>13}")
learn = {"btpy": {}, "btpyfl": {}}
for s in SEEDS:
    a, b = res["btpy"][s], res["btpyfl"][s]
    learn["btpy"][s], learn["btpyfl"][s] = share(a) >= 0.5, share(b) >= 0.5
    err = max(float(np.abs(gap(b, c)).max()) for c in ("10000", "1000"))
    print(f"{s:>4} | {grid(a):<26} {share(a):5.2f} {gap(a, '10000').mean():8.2f} {gap(a, '1000').mean():7.2f} | {grid(b):<26} {share(b):5.2f} {err:13.1e}")

kp, kc = sum(learn["btpy"].values()), sum(learn["btpyfl"].values())
ctl_err = max(float(np.abs(gap(res["btpyfl"][s], c)).max()) for s in SEEDS for c in ("10000", "1000"))
g10 = np.array([gap(res["btpy"][s], "10000").mean() for s in SEEDS])
g1 = np.array([gap(res["btpy"][s], "1000").mean() for s in SEEDS])
rng = np.random.default_rng(20261212)
lb10 = float(np.quantile(g10[rng.integers(0, 8, size=(10000, 8))].mean(axis=1), 0.025))
lb1 = float(np.quantile(g1[rng.integers(0, 8, size=(10000, 8))].mean(axis=1), 0.025))

pred = [("P1 published engine: at least 5 of 8 learners", kp >= 5, f"{kp} of 8"),
        ("P2 floor removed: at most 1 of 8 learners", kc <= 1, f"{kc} of 8"),
        ("P3 floor removed: reported = booked on every test path (< 1e-6 points)", ctl_err < 1e-6, f"max difference {ctl_err:.1e} points"),
        ("P4 published engine: reported - booked above 15 points from 10,000 USD", g10.mean() > 15, f"seed mean {g10.mean():.2f} (seed-cluster lower bound {lb10:.2f})")]
print("\nPredictions written before the run:")
for name, ok, val in pred:
    print(f"  {name}: {val} -> {'MET' if ok else 'NOT MET'}")
print(f"  {sum(ok for _, ok, _ in pred)} of 4 met")

p_f = fisher_exact([[kp, 8 - kp], [kc, 8 - kc]], alternative="greater")[1]
d_pc = sum(learn["btpy"][s] and not learn["btpyfl"][s] for s in SEEDS)
d_cp = sum(learn["btpyfl"][s] and not learn["btpy"][s] for s in SEEDS)
p_sign = binomtest(d_pc, d_pc + d_cp, 0.5, alternative="greater").pvalue
print("\nPost hoc:")
print(f"  Fisher one-sided p (published {kp}/8 against floor removed {kc}/8): {p_f:.4f}")
print(f"  paired seeds (same initial weights and training paths): {d_pc} pairs learn only with the floor, {d_cp} only without; exact sign test p = {p_sign:.4f}")
print(f"  reported - booked from 1,000 USD: seed mean {g1.mean():.2f} (lower bound {lb1:.2f})")
lrn = [s for s in SEEDS if learn["btpy"][s]]
bk10 = np.mean([np.mean(res["btpy"][s]["tests"]["10000"]["draws_booked"]) for s in lrn])
rp10 = np.mean([np.mean(res["btpy"][s]["tests"]["10000"]["draws_reported"]) for s in lrn])
print(f"  the {len(lrn)} learners from 10,000 USD: reported {rp10:+.2f}%, booked {bk10:+.2f}%")

xe = [json.loads(p.read_text()) for p in sorted((XE / "results_xe").glob("*_xe5.json"))]
kx = {e: sum(share(r) >= 0.5 for r in xe if r["engine"] == e) for e in ("btpy", "bt")}
nx = {e: sum(r["engine"] == e for r in xe) for e in ("btpy", "bt")}
p_pool = fisher_exact([[kx["btpy"] + kp, nx["btpy"] + 8 - kx["btpy"] - kp], [kc, 8 - kc]], alternative="greater")[1]
print(f"  with the registered Study XE (published {kx['btpy']}/{nx['btpy']}, backtrader {kx['bt']}/{nx['bt']}): published engine "
      f"{kx['btpy'] + kp}/{nx['btpy'] + 8} against floor removed {kc}/8, Fisher one-sided p = {p_pool:.2e}")

(HERE / "score_ctl.json").write_text(json.dumps(dict(
    learners_published=kp, learners_floor_removed=kc, control_max_abs_gap=ctl_err, gap10_seed_mean=float(g10.mean()), gap10_lower=lb10,
    gap1_seed_mean=float(g1.mean()), gap1_lower=lb1, predictions={n: bool(o) for n, o, _ in pred}, fisher_p=float(p_f),
    sign_test_p=float(p_sign), pooled_with_xe_p=float(p_pool), shares={e: {s: share(res[e][s]) for s in SEEDS} for e in res}), indent=1))
