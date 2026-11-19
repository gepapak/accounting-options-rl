"""D4b (POST HOC, written after D4's simple formula failed its accuracy criterion): the Nordic zero-floor / terminal ratio from the
law without the first-order shortcut. For bar moves treated as draws from the zone-year's own hourly magnitudes with fair signs,
E[max] = sqrt(2 V / pi) - rho and E[S^+] = sqrt(V / (2 pi)), so
    ZF / TERM = 2 - sqrt(2 pi) rho / sqrt(V),
with rho the overshoot constant of the hourly spreads (Siegmund's integral) and V the year's sum of squared spreads.
Measured ratios: test_d4_nordic.csv (20,000 sign draws on the chronological magnitudes)."""
import sys
from math import log, pi, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT if (ROOT / "ACCOUNTING_OPTIONS_2026-10-14").exists() else ROOT / "github" / "accounting-options-rl"   # repository, or workspace
PRE, POST = REPO / "ACCOUNTING_OPTIONS_2026-10-14" / "data_esett_pre", REPO / "data" / "esett"


def rho_siegmund_log(mags, n_u=40000, u_lo=1e-4, u_hi=3e4):
    """Siegmund's integral on a log-spaced grid (fast for thousands of magnitudes). Returns rho in the units of mags."""
    m = np.asarray(mags, float)
    m = m[m > 0]
    s = sqrt(float(np.mean(m ** 2)))
    x = m / s
    u = np.exp(np.linspace(log(u_lo), log(u_hi), n_u))
    cf = np.zeros_like(u)
    for i in range(0, len(x), 256):
        cf += np.cos(np.outer(u, x[i:i + 256])).sum(axis=1)
    cf /= len(x)
    g = np.log(np.maximum(2.0 * (1.0 - cf), 1e-300) / u ** 2) / u ** 2
    integral = float(np.sum(0.5 * (g[1:] * u[1:] + g[:-1] * u[:-1]) * np.diff(np.log(u))))      # int g du = int g u dlog u
    head = g[0] * u_lo                                                                          # g is flat near zero
    tail = (log(2.0) - 2.0 * (log(u_hi) + 1.0)) / u_hi
    return -(integral + head + tail) / pi * s


def hourly(z):
    a = pd.read_csv(PRE / f"esett_EXP14_{z}_pre.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    b = pd.read_csv(POST / f"esett_EXP14_{z}.csv.gz", usecols=["timestampUTC", "imblSpotDifferencePrice"])
    d = pd.concat([a, b]); d["t"] = pd.to_datetime(d.timestampUTC, utc=True); d = d.drop_duplicates("t").dropna()
    d["h"] = d.t.dt.floor("h"); return d.groupby("h").imblSpotDifferencePrice.mean()


# checks of the log-grid integral on laws with known constants
qs = (np.arange(4000) + 0.5) / 4000
from statistics import NormalDist  # noqa: E402
g = np.abs(np.array([NormalDist().inv_cdf(0.5 + q / 2) for q in qs])); e = -np.log(1.0 - qs)
print(f"checks: Gaussian {rho_siegmund_log(g) / sqrt(np.mean(g ** 2)):.4f} (0.5826), Laplace {rho_siegmund_log(e) / sqrt(np.mean(e ** 2)):.4f} (0.7071), "
      f"EURUSD {rho_siegmund_log(np.load(HERE / 'eurusd_mags.npy')) / sqrt(np.mean(np.load(HERE / 'eurusd_mags.npy') ** 2)):.4f} (0.6180 on the linear grid)")

df = pd.read_csv(HERE / "test_d4_nordic.csv")
rows = []
for z in df.zone.unique():
    s = hourly(z)
    for per, (a, b) in {"2024": ("2024-01-01", "2025-01-01"), "post-reform year": ("2025-03-04", "2026-03-04")}.items():
        mag = np.abs(s[(s.index >= pd.Timestamp(a, tz="UTC")) & (s.index < pd.Timestamp(b, tz="UTC"))].values)
        V = float((mag ** 2).sum())
        rho = rho_siegmund_log(mag)
        rows.append(dict(zone=z, period=per, rho_over_s=rho / sqrt(V / len(mag)), pred_b=2.0 - sqrt(2.0 * pi) * rho / sqrt(V)))
out = df.merge(pd.DataFrame(rows), on=["zone", "period"])
out["err_b"] = out.pred_b - out.measured
out.to_csv(HERE / "test_d4b_nordic.csv", index=False)
print(out[["zone", "period", "kappa", "rho_over_s", "pred", "pred_b", "measured", "se", "err_b"]].round(4).to_string(index=False))
from scipy.stats import spearmanr  # noqa: E402

print(f"\nD4b (post hoc): Spearman {spearmanr(out.pred_b, out.measured).correlation:.3f}, mean absolute error {out.err_b.abs().mean():.4f}, "
      f"largest {out.err_b.abs().max():.4f}, mean signed error {out.err_b.mean():+.4f}; share within two standard errors "
      f"{float((out.err_b.abs() <= 2 * out.se).mean()):.2f} (mean standard error {out.se.mean():.4f})")
