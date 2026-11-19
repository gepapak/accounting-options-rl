"""Test D: the law against results that already exist (nothing is re-run except sign draws for the Nordic ratios).
D1  the registered Study XE census and the 16 November pilot census (backtesting.py, capped loss with closure, 2 bp a side)
D2  what the trained PPO agents of Study XE over-report (reported minus booked), per learner
D3  gym-mtsim 2.0.0 as published: a stop-out at a margin level of 0.2, then the balance floored at zero (a buffer before the floor)
D4  the paper's Nordic ladder: zero floor over terminal limited liability from zero equity, 24 zone-years, against 2 - 1.2533 kappa
D1 and D4's formulas are in PREDICTIONS.md / PREDICTIONS_v2.md. D2 and D3 use the same law; their measured values were known."""
import glob
import json
import sys
from math import log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

import law

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
P1 = json.loads((HERE / "predictions.json").read_text())
P2 = json.loads((HERE / "predictions_v2.json").read_text())
eur = np.load(HERE / "eurusd_mags.npy")
s_e = P1["inputs"]["s_eurusd"]
out = {}

# ---------------------------------------------------------------- D1
print("D1. backtesting.py census: forgiven = reported minus the engine's own trade records, % of the starting cash")
print(f"{'L':>4} | {'predicted':>9} | {'XE (registered)':>16} {'ratio':>6} | {'pilot 16 Nov':>12} {'ratio':>6} | ruin pred / XE / pilot")
xe = pd.read_csv(ROOT / "study_xe_2026-11-16" / "xe_census.csv")
pi = pd.read_csv(ROOT / "novelty_search_2026-11-16" / "engines_pilot.csv")
d1 = []
for pr in P2["D1"]:
    L = pr["L"]
    a, b = xe[xe.L == L], pi[pi.L == L]
    fa, fb = (a.btpy_rep - a.btpy_book).mean(), (b.btpy_rep - b.btpy_book).mean()
    d1.append(dict(L=L, pred=pr["forgiven"], xe=float(fa), pilot=float(fb), r_xe=float(fa / pr["forgiven"]), r_pilot=float(fb / pr["forgiven"])))
    print(f"{L:>4} | {100 * pr['forgiven']:9.3f} | {100 * fa:16.3f} {fa / pr['forgiven']:6.3f} | {100 * fb:12.3f} {fb / pr['forgiven']:6.3f} | "
          f"{pr['p_ruin']:.3f} / {a.btpy_zero.mean():.3f} / {b.btpy_zero.mean():.3f}")
out["D1"] = d1

# ---------------------------------------------------------------- D2
print("\nD2. Trained PPO agents of the registered Study XE (two actions, flat or 200x): reported minus booked on 200 test paths")
pred200 = next(r["forgiven"] for r in P2["D1"] if r["L"] == 200)
gaps = []
for f in sorted(glob.glob(str(ROOT / "study_xe_2026-11-16" / "results_xe" / "btpy__s*_xe5.json"))):
    r = json.loads(Path(f).read_text())
    probes = [r["probe"][k] for k in r["probe"]]
    share = float(np.mean([p == 200.0 for p in probes]))
    g10 = float(np.mean(np.array(r["tests"]["10000"]["draws_reported"]) - np.array(r["tests"]["10000"]["draws_booked"])))
    g1 = float(np.mean(np.array(r["tests"]["1000"]["draws_reported"]) - np.array(r["tests"]["1000"]["draws_booked"])))
    gaps.append(dict(seed=r["seed"], share_200=share, gap_10k=g10, gap_1k=g1, zero_10k=r["tests"]["10000"]["hit_zero"]))
every = [g for g in gaps if g["share_200"] == 1.0]
m10 = np.mean([g["gap_10k"] for g in every])
print(f"agents that bet 200x at every probe: {len(every)} of {len(gaps)}; their mean reported - booked from 10,000 USD: {m10:.2f} points "
      f"(range {min(g['gap_10k'] for g in every):.2f} to {max(g['gap_10k'] for g in every):.2f}); the law: {100 * pred200:.2f}; ratio {m10 / (100 * pred200):.3f}")
print(f"their share of test paths ending at zero: {np.mean([g['zero_10k'] for g in every]):.3f}; the law's P(ruin): "
      f"{next(r['p_ruin'] for r in P2['D1'] if r['L'] == 200):.3f}")
out["D2"] = dict(n_everywhere=len(every), gap_mean=float(m10), pred=100 * pred200, agents=gaps)

# ---------------------------------------------------------------- D3
print("\nD3. gym-mtsim 2.0.0 as published (stop-out at margin level 0.2, then the balance floored): forgiven, % of the balance")
over = law.overshoot_log_samples(eur)                      # stationary overshoot of the log price below a far barrier
rho_l = float(over.mean())
FEE, CONTRACT, P_E, N_M = 0.0003, 100_000.0, 1.2, 85       # the package's forex fee (price units), contract size, bars monitored
mt = [pd.read_csv(ROOT / "study_xe_2026-11-16" / "xe_census_mtsim.csv"), pd.read_csv(ROOT / "novelty_search_2026-11-16" / "census_mtsim.csv")]
d3 = []
print(f"{'L':>4} | {'predicted':>9} {'P(balance<0)':>12} | {'XE census':>10} {'pilot':>8} | measured share at zero (XE, pilot)")
for L in (50, 100, 200):
    per_unit = CONTRACT * (P_E / L + FEE)
    v = 10_000.0 / per_unit                                  # lots (ignoring the 0.01 step)
    leff = v * CONTRACT * P_E / 10_000.0                     # notional over balance
    e1 = 1.0 - v * CONTRACT * FEE / 10_000.0                 # equity after the fee
    margin = v * CONTRACT * P_E / L / 10_000.0
    stop_eq = 0.2 * margin                                   # equity at which the stop-out fires
    K = 1.0 - (e1 - stop_eq) / leff                          # price ratio at the stop-out barrier
    p_stop = law.p_min_below(-log(K) + rho_l, -0.5 * s_e ** 2, s_e, N_M)
    loss_beyond = leff * K * (1.0 - np.exp(-over))           # loss beyond the stop-out equity, in units of the balance
    forg = p_stop * float(np.maximum(loss_beyond - stop_eq, 0.0).mean())
    p_zero = p_stop * float((loss_beyond > stop_eq).mean())
    meas = []
    for d in mt:
        g = d[(d.L == L) & (d.cond == "ORIG")]
        meas.append((float((g.rep - g.book).mean()), float(g.zero.mean())))
    d3.append(dict(L=L, pred=forg, p_zero=p_zero, xe=meas[0][0], pilot=meas[1][0]))
    print(f"{L:>4} | {100 * forg:9.2f} {p_zero:12.3f} | {100 * meas[0][0]:10.2f} {100 * meas[1][0]:8.2f} | {meas[0][1]:.3f}, {meas[1][1]:.3f}")
both = pd.concat(mt)
g = both[(both.L == 200) & (both.cond == "ORIG")]
f = (g.rep - g.book).to_numpy()
rng = np.random.default_rng(3)
bs = f[rng.integers(0, len(f), size=(4000, len(f)))].mean(axis=1)
print(f"L = 200, both censuses pooled (600 paths): {100 * f.mean():.2f} [{100 * np.quantile(bs, .025):.2f}, {100 * np.quantile(bs, .975):.2f}]")
out["D3"] = d3

# ---------------------------------------------------------------- D4
print("\nD4. Nordic ladder: zero floor over terminal limited liability from zero equity, fee-free (placebo signs on real hourly spreads)")
REPO = ROOT if (ROOT / "ACCOUNTING_OPTIONS_2026-10-14").exists() else ROOT / "github" / "accounting-options-rl"   # repository, or workspace
PRE, POST = REPO / "ACCOUNTING_OPTIONS_2026-10-14" / "data_esett_pre", REPO / "data" / "esett"
ZONES = ["DK1", "DK2", "FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5"]
old = pd.read_csv(REPO / "ACCOUNTING_OPTIONS_2026-10-14" / "ladder12.csv")


def hourly(z):                                               # as in ladder12.py
    a = pd.read_csv(PRE / f"esett_EXP14_{z}_pre.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    b = pd.read_csv(POST / f"esett_EXP14_{z}.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    d = pd.concat([a, b]); d["t"] = pd.to_datetime(d.timestampUTC, utc=True); d = d.drop_duplicates("t").dropna()
    d["h"] = d.t.dt.floor("h"); return d.groupby("h").imblSpotDifferencePrice.mean()


rng = np.random.default_rng(20261117)
rows = []
for z in ZONES:
    s = hourly(z)
    for per, (a, b) in {"2024": ("2024-01-01", "2025-01-01"), "post-reform year": ("2025-03-04", "2026-03-04")}.items():
        mag = np.abs(s[(s.index >= pd.Timestamp(a, tz="UTC")) & (s.index < pd.Timestamp(b, tz="UTC"))].values)
        V = float((mag ** 2).sum())
        kappa = float((mag ** 3).sum() / V ** 1.5)
        zf, term = [], []
        for _ in range(40):                                  # 20,000 sign draws
            Lc = np.cumsum(rng.choice([-1.0, 1.0], size=(500, len(mag))) * mag, axis=1)
            zf.append(np.maximum(0.0, -Lc.min(axis=1))); term.append(np.maximum(-Lc[:, -1], 0.0))
        zf, term = np.concatenate(zf), np.concatenate(term)
        ratio = float(zf.mean() / term.mean())
        se = ratio * sqrt((zf.std() / sqrt(len(zf)) / zf.mean()) ** 2 + (term.std() / sqrt(len(term)) / term.mean()) ** 2)
        o = old[(old.zone == z) & (old.period == per)].ZF_over_TERM_feefree.iloc[0]
        rows.append(dict(zone=z, period=per, hours=len(mag), kappa=kappa, n_eff=1.0 / kappa ** 2, pred=2.0 - 1.2533 * kappa, measured=ratio,
                         se=float(se), table=float(o)))
df = pd.DataFrame(rows)
df.to_csv(HERE / "test_d4_nordic.csv", index=False)
print(df.assign(err=df.pred - df.measured).round(4).to_string(index=False))
from scipy.stats import spearmanr, pearsonr  # noqa: E402

rho_s = spearmanr(df.pred, df.measured).correlation
mae = float((df.pred - df.measured).abs().mean())
print(f"\n24 zone-years: Spearman {rho_s:.3f} (criterion >= 0.8), Pearson {pearsonr(df.pred, df.measured)[0]:.3f}, mean absolute error {mae:.4f} "
      f"(criterion <= 0.05), largest error {float((df.pred - df.measured).abs().max()):.4f}; mean standard error of a measured ratio {df.se.mean():.4f}")
print("against the paper's table (2,000 draws): mean absolute difference of the recomputed ratios", f"{float((df.measured - df.table).abs().mean()):.4f}")
print("D4:", "CRITERIA MET" if (rho_s >= 0.8 and mae <= 0.05) else "CRITERIA NOT MET")
out["D4"] = dict(spearman=float(rho_s), mae=mae, ok=bool(rho_s >= 0.8 and mae <= 0.05))
(HERE / "test_d_existing.json").write_text(json.dumps(out, indent=1))
