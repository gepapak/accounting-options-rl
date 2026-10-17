"""EXPLORATORY: spread tails before and after the mFRR EAM go-live (4 March 2025), twelve Nordic zones, eSett EXP14.
Pre: 2023-01-01 .. 2025-03-03 (quarter-hour imbalance settlement already in place); post: 2025-03-04 .. 2026-09-27.
Statistics of the imbalance-minus-spot spread per settlement period (EUR/MWh): median |s|, standard deviation,
quarter-hours with |s| > 1,000 per 10,000, and the maximum |s|. Writes tails12.csv."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PRE, POST = HERE / "data_esett_pre", HERE.parent / "data" / "esett"
ZONES = ["DK1", "DK2", "FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5"]
rows = []
for z in ZONES:
    a = pd.read_csv(PRE / f"esett_EXP14_{z}_pre.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    b = pd.read_csv(POST / f"esett_EXP14_{z}.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    for period, d in (("2023-01..2025-03", a[a.timestampUTC >= "2023-01-01"]), ("post-reform", b)):
        s = d.imblSpotDifferencePrice.dropna().to_numpy()
        rows.append(dict(zone=z, period=period, n=len(s), median_abs=float(np.median(np.abs(s))), sd=float(s.std()),
                         per10k_over_1000=float((np.abs(s) > 1000).mean() * 1e4), max_abs=float(np.abs(s).max())))
df = pd.DataFrame(rows)
df.to_csv(HERE / "tails12.csv", index=False)
print(df.round(2).to_string(index=False))
