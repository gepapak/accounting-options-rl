"""Incentive-compatible collateral per MW of open imbalance exposure (exploratory).

A participant with buffer b (equity incl. posted collateral) holds exposure q (MW) over the unsettled
window W; P&L Y_W per MW is the placebo (fair-sign) sum of real spread magnitudes over W hours.
One-window gambling incentive per MW at buffer b:  G(b) = -fee*W + E[(-b/q - Y_W)^+].
kappa_W(fee) = smallest buffer per MW with G <= 0  (=> IC requires collateral >= kappa_W * q).
Reported for W = 1 h, 1 day, 1 week, 3 weeks (official Nordic invoicing/settlement ~3 weeks after delivery)
and fees 0.1 / 0.5 / 1.0 EUR/MWh, by zone and regime; Gaussian benchmark uses sigma_W.
Also: steady-state eSett imbalance term 3*S2 for a constant 1 MW position, S2 = E|weekly sum| (per MW).
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
rng = np.random.default_rng(7)
panel = pd.read_csv("panel_hourly.csv.gz", parse_dates=["time_utc"]); panel["resolution_h"] = 1.0
WS = {"1h": 1, "1d": 24, "1w": 168, "3w": 504}
FEES = [0.1, 0.5, 1.0]
DRAWS = 400
rows = []
def psi(z):  # E[(-z - N)^+] for standard normal
    return norm.pdf(z) - z * norm.sf(z)
for (zone, regime), g in panel.groupby(["zone", "regime"]):
    dt = g.resolution_h.iloc[0]
    mag = g.spread_eur.abs().values * dt
    for wname, wh in WS.items():
        wp = int(round(wh / dt))
        nwin = len(mag) // wp
        if nwin < 5: continue
        M = mag[: nwin * wp].reshape(nwin, wp)
        # placebo sums: DRAWS sign draws per window
        Y = np.concatenate([(rng.choice([-1.0, 1.0], size=M.shape) * M).sum(axis=1) for _ in range(DRAWS)])
        sd = Y.std()
        for fee in FEES:
            c = fee * wh
            f = lambda k: np.mean(np.maximum(-k - Y, 0.0)) - c
            kmax = np.max(-Y)
            kap = brentq(f, 0.0, kmax) if f(0.0) > 0 and f(kmax) < 0 else (0.0 if f(0.0) <= 0 else np.nan)
            zg = brentq(lambda z: psi(z) - c / sd, -5, 40) if c / sd < psi(0) else 0.0
            rows.append(dict(zone=zone, regime=regime, window=wname, window_h=wh, fee=fee, n_windows=nwin,
                             sigma_W=sd, zombie_call_per_MW=np.mean(np.maximum(Y, 0)),
                             q999_loss=-np.quantile(Y, 0.001), kappa_eur_per_MW=kap, kappa_over_sigma=kap / sd,
                             kappa_gauss_over_sigma=zg, tail_factor=kap / (zg * sd) if zg > 0 else np.nan))
    # steady-state eSett imbalance component for a constant 1 MW position: 3 * E|weekly sum|
    wp = int(round(168 / dt)); nwin = len(mag) // wp
    M = mag[: nwin * wp].reshape(nwin, wp)
    S2 = np.mean(np.abs(np.concatenate([(rng.choice([-1.0, 1.0], size=M.shape) * M).sum(axis=1) for _ in range(DRAWS)])))
    rows.append(dict(zone=zone, regime=regime, window="eSett_3xS2_steady", window_h=np.nan, fee=np.nan,
                     kappa_eur_per_MW=3 * S2))
df = pd.DataFrame(rows)
df.to_csv("ic_collateral_hourly.csv", index=False)
pd.set_option("display.width", 230)
k = df[df.window == "3w"].pivot_table(index=["zone", "regime"], columns="fee", values=["kappa_eur_per_MW", "kappa_over_sigma", "tail_factor"])
print(k.round(2).to_string())
s = df[df.window == "eSett_3xS2_steady"].set_index(["zone", "regime"]).kappa_eur_per_MW
w3 = df[(df.window == "3w") & (df.fee == 0.1)].set_index(["zone", "regime"])
print("\nsigma_3w, zombie call per MW (3w), eSett 3xS2 steady per MW, 3xS2/sigma_3w")
print(pd.DataFrame({"sigma_3w": w3.sigma_W, "zombie_call": w3.zombie_call_per_MW, "eSett_3xS2": s,
                    "ratio_to_sigma": s / w3.sigma_W, "kappa_fee0.1": w3.kappa_eur_per_MW}).round(0).to_string())
lagdep = df[(df.fee == 0.1) & df.window.isin(list(WS))].pivot_table(index=["zone", "regime"], columns="window_h", values="kappa_eur_per_MW")
print("\nkappa (EUR per MW) vs window, fee 0.1:\n", lagdep.round(0).to_string())
