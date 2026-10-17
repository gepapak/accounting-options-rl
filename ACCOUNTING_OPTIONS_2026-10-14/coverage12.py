"""EXPLORATORY: twelve Nordic zones, 2021-11 .. 2026-09 (eSett Open Data EXP14, hourly means).
For each zone and month: the buffer kappa_3w per MW that deters a 3-week (Nordic settlement lag) limited-liability
gamble on the imbalance spread (fair random hourly signs, cost FEE EUR/MWh), from the trailing three months;
and collateral per MW of consumption under
  LEVEL  eSett Standard Formula volume term, m(V1+V2)P = (3/7)*168 MWh*P, P = mean max(imbalance price, 0) of the month
         (Denmark: 0 EUR at eSett; its Energinet C1 consumption rule is evaluated in coverage.py)
  VOL    a volatility-indexed alternative: kappa estimated from the three months BEFORE the month (causal)
Coverage = collateral / kappa_3w (target kappa uses the month and the two before it, as in coverage.py)."""
import numpy as np, pandas as pd, sys
from pathlib import Path
FEE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
W = 504; DRAWS = 120
rng = np.random.default_rng(12)
HERE = Path(__file__).resolve().parent
PRE = HERE / "data_esett_pre"; POST = HERE.parent / "data" / "esett"   # fetch_esett_pre.py; ../data/fetch_esett.py
ZONES = ["DK1", "DK2", "FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5"]
REFORM = pd.Timestamp("2025-03-04", tz="UTC")

def hourly(zone):
    a = pd.read_csv(PRE / f"esett_EXP14_{zone}_pre.csv.gz", usecols=["timestampUTC", "imblSalesPrice", "imblSpotDifferencePrice"])
    b = pd.read_csv(POST / f"esett_EXP14_{zone}.csv.gz", usecols=["timestampUTC", "imblSalesPrice", "imblSpotDifferencePrice"])
    d = pd.concat([a, b]); d["t"] = pd.to_datetime(d.timestampUTC, utc=True)
    d = d.drop_duplicates("t").sort_values("t").dropna()
    d["h"] = d.t.dt.floor("h")
    h = d.groupby("h").agg(imb=("imblSalesPrice", "mean"), spread=("imblSpotDifferencePrice", "mean")).reset_index()
    h["month"] = h.h.dt.tz_convert(None).dt.to_period("M")
    return h

def kappa(spread, fee):
    a = np.abs(spread); n = len(a)
    starts = np.arange(0, n - W, 24)
    if len(starts) < 10: return np.nan
    M = a[starts[:, None] + np.arange(W)[None, :]]
    Y = np.concatenate([(rng.choice([-1.0, 1.0], size=M.shape) * M).sum(axis=1) for _ in range(DRAWS)])
    c = fee * W
    f = lambda k: np.mean(np.maximum(-k - Y, 0.0)) - c
    if f(0.0) <= 0: return 0.0
    lo, hi = 0.0, float(np.max(-Y))
    for _ in range(50):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)

rows = []
for zone in ZONES:
    h = hourly(zone); months = sorted(h.month.unique())
    kap = {}
    for i, m in enumerate(months):
        if i >= 2:
            kap[m] = kappa(h[h.month.isin(months[i - 2:i + 1])].spread.values, FEE)
    for i, m in enumerate(months):
        if i < 3 or m not in kap: continue
        cur = h[h.month == m]
        prev3 = h[h.month.isin(months[i - 3:i])]
        kv = kappa(prev3.spread.values, FEE)
        level = (3 / 7) * 168 * np.maximum(cur.imb.values, 0).mean()
        rows.append(dict(zone=zone, month=str(m), post=m.to_timestamp() >= REFORM.tz_convert(None).to_period("M").to_timestamp(),
                         level_imb=cur.imb.mean(), spread_sd=cur.spread.std(), kappa=kap[m], LEVEL=level, VOL=kv))
df = pd.DataFrame(rows)
df.to_csv(f"coverage12_fee{FEE}.csv", index=False)
pd.set_option("display.width", 230)
df["cov_LEVEL"] = df.LEVEL / df.kappa.replace(0, np.nan)
g = df.groupby(["zone", "post"])
summ = pd.DataFrame({"months": g.size(), "spread_sd": g.spread_sd.median(), "level": g.level_imb.median(), "kappa": g.kappa.median(),
                     "LEVEL_EUR": g.LEVEL.mean(), "cov_LEVEL_med": g.cov_LEVEL.median(),
                     "LEVEL<1": g.cov_LEVEL.apply(lambda s: (s < 1).mean())}).round(2)
print(f"fee {FEE} EUR/MWh"); print(summ.to_string())
# efficiency frontier: volatility-indexed rule a * kappa_prev3 (causal) versus the level-indexed eSett volume term
fr = []
for zset, zl in [("FI+SE+NO", [z for z in ZONES if not z.startswith("DK")]), ("DK", ["DK1", "DK2"])]:
    d = df[df.zone.isin(zl)]
    for post in (False, True):
        e = d[d.post == post]
        fr.append(dict(zones=zset, post=post, rule="LEVEL (eSett volume term)", mean_collateral=e.LEVEL.mean(), fail=(e.LEVEL < e.kappa).mean()))
        for a in (1.0, 1.25, 1.5, 2.0, 3.0):
            fr.append(dict(zones=zset, post=post, rule=f"VOL x{a:g}", mean_collateral=(a * e.VOL).mean(), fail=(a * e.VOL < e.kappa).mean()))
F = pd.DataFrame(fr); F.to_csv(f"frontier12_fee{FEE}.csv", index=False)
print("share of zone-months in which collateral per MW < kappa (not deterred), and mean collateral per MW")
print(F.round(3).to_string(index=False))
