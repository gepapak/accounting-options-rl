"""EXPLORATORY: Danish imbalance spreads 2019-2026 on the accounting-option ladder, and collateral comparisons.

Per MW of constant open imbalance exposure, placebo signs (no edge), fee FEE EUR/MWh, horizon one year.
Rules: LL (capped loss), LAG_W (closure W hours after the ledger first reaches zero; Nordic invoicing/settlement
~3 weeks after delivery per eSett), TERM (terminal limited liability), ZF (simulator zero floor).
Collateral references (sources in FACTS.md): Denmark 0 EUR eSett collateral, Energinet credit limit / security
DKK 1,000,000 (current BRP page) or DKK 2,000,000 (Forskrift C1, 2021, trading BRPs); eSett Standard Formula
C = 3(S1+S2) + m(V1+V2)P, minimum 40,000 EUR (FI/SE/NO), applied here to Danish prices as a stylized comparison.
"""
import numpy as np, pandas as pd
FX = 7.4508
FEE = 0.10
DRAWS = 3000
rng = np.random.default_rng(20261014)
panel = pd.read_csv("panel_hourly.csv.gz", parse_dates=["time_utc"]); panel["resolution_h"] = 1.0
ic = pd.read_csv("ic_collateral_hourly.csv")
LAGS = {"LAG_1d": 24, "LAG_1w": 168, "LAG_3w": 504}
out = []
for (zone, regime), g in panel.groupby(["zone", "regime"]):
    g = g.sort_values("time_utc"); dt = g.resolution_h.iloc[0]
    n = int(round(8760 / dt)); mag = g.spread_eur.abs().values[:n] * dt; n = len(mag); fee = FEE * dt
    sig = np.sqrt((mag ** 2).sum())
    kap = {f: float(ic[(ic.zone == zone) & (ic.regime == regime) & (ic.window == "3w") & (ic.fee == f)].kappa_eur_per_MW.iloc[0]) for f in (0.1, 0.5, 1.0)}
    for e_name, e in [("e=0", 0.0), ("e=kappa3w", kap[0.1])]:
        acc = {k: [] for k in ["LL", *LAGS, "TERM", "ZF"]}
        for b in range(0, DRAWS, 250):
            m = min(250, DRAWS - b)
            L = e + np.cumsum(rng.choice([-1.0, 1.0], size=(m, n)) * mag - fee, axis=1)
            neg = L <= 0; hit = neg.any(axis=1); s0 = np.where(hit, neg.argmax(axis=1), n - 1)
            D = lambda idx: np.maximum(-L[np.arange(m), idx], 0.0)
            acc["LL"].append(np.where(hit, D(s0), 0.0))
            for k, wh in LAGS.items():
                acc[k].append(np.where(hit, D(np.minimum(s0 + int(round(wh / dt)), n - 1)), 0.0))
            acc["TERM"].append(np.maximum(-L[:, -1], 0.0))
            acc["ZF"].append(np.maximum(0.0, -np.minimum.accumulate(L, axis=1)[:, -1]))
        row = dict(zone=zone, regime=regime, start=e_name, sigma_year_eur_per_MW=sig)
        row.update({k: float(np.concatenate(v).mean()) for k, v in acc.items()})
        out.append(row)
    # eSett Standard Formula, steady state, 1 MW constant sales-based short imbalance (stylized on Danish prices)
    imb = g.imb_eur.values * dt; wp = int(round(168 / dt)); nw = len(imb) // wp
    weekly = imb[: nw * wp].reshape(nw, wp).sum(axis=1)                    # EUR per MW-week of imbalance purchases
    S2 = np.mean(np.abs(weekly))
    Ppos = np.mean(np.maximum(g.imb_eur.values, 0.0))
    C_ss = 3 * S2 + (3 / 7) * 168 * Ppos
    out[-1].update(dict(kappa3w_fee0p1=kap[0.1], kappa3w_fee0p5=kap[0.5], kappa3w_fee1p0=kap[1.0], eSett_steady_per_MW=C_ss,
                        eSett_steady_over_kappa=C_ss / kap[0.1],
                        DK_cap_MW_DKK1M=1e6 / FX / kap[0.1], DK_cap_MW_DKK2M=2e6 / FX / kap[0.1], eSett_min_cap_MW=40000 / kap[0.1]))
df = pd.DataFrame(out)
df.to_csv("market_ladder.csv", index=False)
pd.set_option("display.width", 250)
z0 = df[df.start == "e=0"].set_index(["zone", "regime"])[["sigma_year_eur_per_MW", "LL", "LAG_1d", "LAG_1w", "LAG_3w", "TERM", "ZF"]]
print("Subsidy per MW-year from zero equity (EUR), fee 0.1 EUR/MWh:"); print(z0.round(0).to_string())
zk = df[df.start == "e=kappa3w"].set_index(["zone", "regime"])
print("\nSubsidy per MW-year from the IC buffer kappa_3w (EUR):"); print(zk[["LL", "LAG_1d", "LAG_1w", "LAG_3w", "TERM", "ZF"]].round(0).to_string())
print("\nCollateral comparisons per MW:"); print(zk[["kappa3w_fee0p1", "kappa3w_fee0p5", "kappa3w_fee1p0", "eSett_steady_per_MW", "eSett_steady_over_kappa", "DK_cap_MW_DKK1M", "DK_cap_MW_DKK2M", "eSett_min_cap_MW"]].round(1).to_string())
