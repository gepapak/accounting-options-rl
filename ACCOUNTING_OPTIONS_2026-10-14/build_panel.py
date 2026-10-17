"""Build a Danish imbalance-spread panel (EUR/MWh) from local official Energinet files.

Inputs: data/energinet/ of this repository (Energinet, Energi Data Service, CC BY 4.0).
Pre-reform (2019-11 .. 2025-03-04): hourly ImbalancePriceDKK (RegulatingBalancePowerdata)
  converted with the hour's DKK/EUR ratio from Elspotprices, minus SpotPriceEUR.
Post-reform (2025-03-04 .. 2026-08-17): quarter-hour ImbalancePriceEUR - SpotPriceEUR
  from ImbalancePrice_*_post_golive (data/energinet/ of this repository).
Output: panel_hourly.csv.gz with columns time_utc, zone, spread_eur, regime, resolution_h.
Post-reform quarter-hours are kept at 15 min (resolution_h = 0.25).
"""
import pandas as pd, numpy as np
from pathlib import Path
P7 = str(Path(__file__).resolve().parents[1] / "data" / "energinet") + "/"   # Elspotprices, RegulatingBalancePowerdata
ZIP = P7                                                                   # ImbalancePrice post go-live

spot = pd.read_csv(P7 + "Elspotprices_2019_2025.csv.gz", parse_dates=["HourUTC"])
spot["fx"] = spot.SpotPriceDKK / spot.SpotPriceEUR.replace(0, np.nan)
fx_med = spot.fx.median()
spot["fx"] = spot.fx.where(spot.fx.between(7.3, 7.6), fx_med)
reg = pd.read_csv(P7 + "RegulatingBalancePowerdata_2019_2025.csv.gz", parse_dates=["HourUTC"])
pre = reg.merge(spot[["HourUTC", "PriceArea", "SpotPriceEUR", "fx"]], on=["HourUTC", "PriceArea"], how="inner")
pre["imb_eur"] = pre.ImbalancePriceDKK / pre.fx
pre["spot_eur"] = pre.SpotPriceEUR
pre["spread_eur"] = pre.imb_eur - pre.spot_eur
pre = pre.rename(columns={"HourUTC": "time_utc", "PriceArea": "zone"})
pre = pre[pre.time_utc < pd.Timestamp("2025-03-04 11:15")]
pre["resolution_h"] = 1.0

posts = []
for z in ["DK1", "DK2"]:
    d = pd.read_csv(ZIP + f"ImbalancePrice_{z}_post_golive.csv.gz")
    d["time_utc"] = pd.to_datetime(d.TimeUTC, utc=True).dt.tz_localize(None)
    d["zone"] = z
    d["imb_eur"] = d.ImbalancePriceEUR
    d["spot_eur"] = d.SpotPriceEUR
    d["spread_eur"] = d.ImbalancePriceEUR - d.SpotPriceEUR
    d["resolution_h"] = 0.25
    posts.append(d[["time_utc", "zone", "imb_eur", "spot_eur", "spread_eur", "resolution_h"]])
post = pd.concat(posts)
panel = pd.concat([pre[["time_utc", "zone", "imb_eur", "spot_eur", "spread_eur", "resolution_h"]], post], ignore_index=True)
panel = panel.dropna(subset=["spread_eur"]).sort_values(["zone", "time_utc"]).reset_index(drop=True)

def regime(t):
    if t < pd.Timestamp("2021-11-01"): return "R1 pre-single-price (2019-11..2021-10)"
    if t < pd.Timestamp("2023-01-01"): return "R2 crisis (2021-11..2022-12)"
    if t < pd.Timestamp("2025-03-04 11:15"): return "R3 post-crisis hourly (2023-01..2025-03)"
    return "R4 15-min mFRR EAM (2025-03..2026-08)"
panel["regime"] = panel.time_utc.map(regime)
panel.to_csv("panel.csv.gz", index=False)
print(panel.groupby(["zone", "regime"]).agg(n=("spread_eur", "size"), start=("time_utc", "min"), end=("time_utc", "max"),
      mean=("spread_eur", "mean"), sd=("spread_eur", "std"), mean_abs=("spread_eur", lambda s: s.abs().mean())).round(2).to_string())
print("fx median", round(fx_med, 4))
