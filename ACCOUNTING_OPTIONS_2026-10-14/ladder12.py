"""EXPLORATORY: the accounting-option ladder in the twelve Nordic zones, one pre-reform year (2024) versus the first
post-reform year (2025-03-04 .. 2026-03-03); hourly mean spreads (eSett EXP14), placebo hourly signs, 1 MW, fee 0.10,
from zero equity. Also the fee-free ratio ZF/TERM (Theorem 2: between 1 and 2, = 2 in the diffusion limit)."""
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
PRE = HERE / "data_esett_pre"; POST = HERE.parent / "data" / "esett"   # fetch_esett_pre.py; ../data/fetch_esett.py
ZONES = ["DK1", "DK2", "FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5"]
rng = np.random.default_rng(2024); DRAWS = 2000
def hourly(z):
    a = pd.read_csv(PRE / f"esett_EXP14_{z}_pre.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    b = pd.read_csv(POST / f"esett_EXP14_{z}.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    d = pd.concat([a, b]); d["t"] = pd.to_datetime(d.timestampUTC, utc=True); d = d.drop_duplicates("t").dropna()
    d["h"] = d.t.dt.floor("h"); return d.groupby("h").imblSpotDifferencePrice.mean()
def ladder(mag, fee):
    n = len(mag); acc = {k: [] for k in ("LL", "LAG_3w", "TERM", "ZF")}
    for b in range(0, DRAWS, 250):
        m = 250; L = np.cumsum(rng.choice([-1.0, 1.0], size=(m, n)) * mag - fee, axis=1)
        neg = L <= 0; hit = neg.any(1); s0 = np.where(hit, neg.argmax(1), n - 1); r = np.arange(m)
        acc["LL"].append(np.where(hit, np.maximum(-L[r, s0], 0), 0)); acc["LAG_3w"].append(np.where(hit, np.maximum(-L[r, np.minimum(s0 + 504, n - 1)], 0), 0))
        acc["TERM"].append(np.maximum(-L[:, -1], 0)); acc["ZF"].append(np.maximum(0, -np.minimum.accumulate(L, 1)[:, -1]))
    return {k: float(np.concatenate(v).mean()) for k, v in acc.items()}
rows = []
for z in ZONES:
    s = hourly(z)
    for per, (a, b) in {"2024": ("2024-01-01", "2025-01-01"), "post-reform year": ("2025-03-04", "2026-03-04")}.items():
        mag = np.abs(s[(s.index >= pd.Timestamp(a, tz="UTC")) & (s.index < pd.Timestamp(b, tz="UTC"))].values)
        r = ladder(mag, 0.10); r0 = ladder(mag, 0.0)
        rows.append(dict(zone=z, period=per, hours=len(mag), sigma_year=float(np.sqrt((mag ** 2).sum())), **r, ZF_over_TERM_feefree=r0["ZF"] / r0["TERM"]))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv("ladder12.csv", index=False)
pd.set_option("display.width", 200)
print(df.pivot(index="zone", columns="period", values=["LAG_3w", "TERM", "ZF", "ZF_over_TERM_feefree"]).round(2).reindex(ZONES).to_string())
