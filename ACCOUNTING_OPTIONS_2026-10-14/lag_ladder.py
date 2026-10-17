"""Accounting/settlement 'liability ladder' on real Danish spreads (exploratory).

A participant holds a constant 1 MW exposure; per settlement period P&L x_t = eps_t*|s_t|*dt - fee*dt,
eps_t independent fair signs (placebo: no edge). Starting equity e (EUR per MW).
Rules (subsidy = E[payoff] - E[booked ledger at closure]):
  FL      full liability: payoff L_T                                   -> 0
  LL0     capped loss, closed at first L_t <= 0                         -> overshoot only
  LAG_W   closure detected with lag W: closed at sigma0 + W, payoff L^+  -> zombie window
  TERM    terminal limited liability: payoff L_T^+
  ZF      zero floor, reflected, continues                              -> lookback
Diffusion prediction (no fee): ZF = 2 x TERM for every starting equity (reflection principle).
"""
import numpy as np, pandas as pd, argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=2000)
ap.add_argument("--fee", type=float, default=0.10)   # EUR/MWh
ap.add_argument("--seed", type=int, default=20261014)
ap.add_argument("--out", default="ladder.csv")
a = ap.parse_args()
panel = pd.read_csv("panel.csv.gz", parse_dates=["time_utc"])
rng = np.random.default_rng(a.seed)
LAGS_H = [0, 24, 168, 504]           # 0, 1 day, 1 week, 3 weeks (official ~3-week invoicing/settlement)
rows = []
for (zone, regime), g in panel.groupby(["zone", "regime"]):
    g = g.sort_values("time_utc")
    dt = g.resolution_h.iloc[0]
    mag = (g.spread_eur.abs().values * dt)            # EUR per MW per period
    n_h = int(round(8760 / dt))                      # one-year horizon in periods
    if len(mag) < n_h: n_h = len(mag)                # R2 is 14 months; all others longer
    mag = mag[:n_h]; T_h = n_h * dt
    fee = a.fee * dt
    sigma_T = np.sqrt(np.sum(mag ** 2))              # sd of booked P&L over the horizon (placebo)
    for z in [0.0, 0.1, 0.25, 0.5, 1.0]:             # starting equity in units of sigma_T
        e = z * sigma_T
        acc = {k: [] for k in ["FL", "LL0"] + [f"LAG{w}" for w in LAGS_H[1:]] + ["TERM", "ZF"]}
        for b in range(0, a.draws, 250):
            m = min(250, a.draws - b)
            eps = rng.choice([-1.0, 1.0], size=(m, n_h))
            x = eps * mag - fee
            L = e + np.cumsum(x, axis=1)
            booked_T = L[:, -1]
            acc["FL"].append(np.zeros(m))
            neg = L <= 0
            hit = neg.any(axis=1)
            s0 = np.where(hit, neg.argmax(axis=1), n_h)          # first passage index
            # LL0: closed at s0, payoff 0 -> subsidy = -L_s0 (overshoot)
            idx = np.minimum(s0, n_h - 1)
            Ls0 = L[np.arange(m), idx]
            acc["LL0"].append(np.where(hit, -Ls0, 0.0))
            for w in LAGS_H[1:]:
                wp = int(round(w / dt))
                tc = np.minimum(s0 + wp, n_h - 1)
                Ltc = L[np.arange(m), tc]
                acc[f"LAG{w}"].append(np.where(hit, np.maximum(-Ltc, 0.0), 0.0))
            acc["TERM"].append(np.maximum(-booked_T, 0.0))
            runmin = np.minimum.accumulate(L, axis=1)
            F = np.maximum(0.0, -runmin[:, -1])
            acc["ZF"].append(F)
        for k, v in acc.items():
            v = np.concatenate(v)
            rows.append(dict(zone=zone, regime=regime, rule=k, e_over_sigmaT=z, sigma_T_eur_per_MW=sigma_T,
                             horizon_h=T_h, subsidy_mean_eur_per_MW=v.mean(), subsidy_se=v.std() / np.sqrt(len(v)),
                             subsidy_over_sigmaT=v.mean() / sigma_T))
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
pv = df.pivot_table(index=["zone", "regime", "e_over_sigmaT"], columns="rule", values="subsidy_over_sigmaT")
pv = pv[["FL", "LL0", "LAG24", "LAG168", "LAG504", "TERM", "ZF"]]
pv["ZF/TERM"] = pv.ZF / pv.TERM
pd.set_option("display.width", 220)
print(pv.round(4).to_string())
from scipy.stats import norm
print("\nDiffusion (no fee) TERM/sigma_T by z:", {z: round(norm.pdf(z) - z * norm.sf(z), 4) for z in [0, 0.1, 0.25, 0.5, 1.0]})
print(df.groupby(["zone", "regime"]).sigma_T_eur_per_MW.first().round(0).to_string())
