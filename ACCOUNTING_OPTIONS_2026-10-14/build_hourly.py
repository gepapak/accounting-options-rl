"""Hourly version of the panel for cross-regime comparability: post-reform quarter-hours are averaged to hourly
mean imbalance price, day-ahead price and spread (a constant 1 MW position over an hour earns the hourly mean spread)."""
import pandas as pd
p = pd.read_csv("panel.csv.gz", parse_dates=["time_utc"])
p["hour"] = p.time_utc.dt.floor("h")
h = p.groupby(["zone", "hour"]).agg(imb_eur=("imb_eur", "mean"), spot_eur=("spot_eur", "mean"), spread_eur=("spread_eur", "mean"),
                                     n=("spread_eur", "size"), regime=("regime", "first")).reset_index()
h = h[(h.n == 1) | (h.n == 4)].rename(columns={"hour": "time_utc"}).drop(columns="n")
h.to_csv("panel_hourly.csv.gz", index=False)
print(h.groupby(["zone", "regime"]).agg(n=("spread_eur", "size"), sd=("spread_eur", "std"), mean_abs=("spread_eur", lambda s: s.abs().mean()),
      level=("imb_eur", "mean"), dayahead=("spot_eur", "mean")).round(1).to_string())
