"""EXPLORATORY: level-indexed collateral versus the volatility-driven limited-liability option, monthly 2019-2026.

kappa_3w(fee): buffer per MW of open imbalance exposure above which a one-window (3 weeks ~ Nordic invoicing and
settlement lag) gamble has non-positive value for a limited-liability holder (no edge, cost 'fee' per MWh);
estimated each month from the trailing 3 months of hourly spreads with fair random hourly signs.
Level-indexed rules per MW of consumption (both scale with a price LEVEL, not with spread volatility):
  DK_C1   Energinet Forskrift C1 (28 Jan 2021, consumption BRPs): 5 % of previous month's consumption x the year's
          forward price -> 0.05 * 730 MWh * F; F proxied by the calendar year's mean day-ahead price.
  ESETT_V eSett Standard Formula volume term: m (V1+V2) P with m = 3/7, V1 = 7 days of consumption (168 MWh per MW),
          P = 7-day mean imbalance price with negatives set to zero -> 72 * mean(max(imbalance price, 0)).
Coverage = collateral / kappa (>= 1: the rule deters the one-window gamble for an exposure equal to consumption).
"""
import numpy as np, pandas as pd
rng = np.random.default_rng(31)
h = pd.read_csv("panel_hourly.csv.gz", parse_dates=["time_utc"])
h["month"] = h.time_utc.dt.to_period("M")
h["year"] = h.time_utc.dt.year
W = 504; FEES = (0.1, 0.5); DRAWS = 150
def kappa(spread, fee):
    a = np.abs(spread); n = len(a)
    starts = np.arange(0, n - W, 24)
    if len(starts) < 10: return np.nan, np.nan
    idx = starts[:, None] + np.arange(W)[None, :]
    M = a[idx]
    Y = np.concatenate([(rng.choice([-1.0, 1.0], size=M.shape) * M).sum(axis=1) for _ in range(DRAWS)])
    c = fee * W
    f = lambda k: np.mean(np.maximum(-k - Y, 0.0)) - c
    if f(0.0) <= 0: return 0.0, Y.std()
    lo, hi = 0.0, float(np.max(-Y))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi), Y.std()
rows = []
for zone, g in h.groupby("zone"):
    g = g.sort_values("time_utc").reset_index(drop=True)
    F_year = g.groupby("year").spot_eur.mean()
    months = sorted(g.month.unique())
    for i, m in enumerate(months):
        if i < 2: continue
        win = g[g.month.isin(months[i - 2:i + 1])]
        cur = g[g.month == m]
        r = dict(zone=zone, month=str(m), level_imb=cur.imb_eur.mean(), level_da=cur.spot_eur.mean(),
                 spread_sd=win.spread_eur.std(), regime=cur.regime.iloc[-1])
        for fee in FEES:
            k, sd = kappa(win.spread_eur.values, fee)
            r[f"kappa_{fee}"] = k; r["sigma_3w"] = sd
        r["DK_C1_per_MW"] = 0.05 * 730 * F_year.loc[cur.year.iloc[0]]
        r["ESETT_V_per_MW"] = (3 / 7) * 168 * np.maximum(cur.imb_eur.values, 0).mean()
        for fee in FEES:
            for rule in ("DK_C1", "ESETT_V"):
                r[f"cov_{rule}_{fee}"] = r[f"{rule}_per_MW"] / r[f"kappa_{fee}"] if r[f"kappa_{fee}"] > 0 else np.inf
        rows.append(r)
df = pd.DataFrame(rows); df.to_csv("coverage_monthly.csv", index=False)
pd.set_option("display.width", 220)
cols = ["level_imb", "spread_sd", "kappa_0.1", "kappa_0.5", "DK_C1_per_MW", "ESETT_V_per_MW", "cov_DK_C1_0.1", "cov_ESETT_V_0.1", "cov_DK_C1_0.5", "cov_ESETT_V_0.5"]
d2 = df.replace(np.inf, np.nan)
print(d2.groupby(["zone", "regime"])[cols].median().round(2).to_string())
print("\nshare of months with coverage < 1 (fee 0.1):")
print(d2.assign(dk=d2["cov_DK_C1_0.1"] < 1, es=d2["cov_ESETT_V_0.1"] < 1).groupby(["zone", "regime"])[["dk", "es"]].mean().round(2).to_string())
