"""Exploratory theory check: is the loss a floor-and-stop engine forgives (backtesting.py's rule) the overshoot of a discretely
monitored barrier, and does the Broadie-Glasserman-Kou (1997) constant predict it?

A position of L x equity is held from bar 1. Ruin is the first close with equity <= 0. The engine reports -100% there; the ledger books
the close-to-close loss, which overshoots the barrier. For a Gaussian random walk, the mean overshoot of a discretely monitored barrier
is about beta * sigma_step, with beta = -zeta(1/2)/sqrt(2 pi) = 0.5826 (Broadie, Glasserman and Kou 1997; Siegmund 1985). In equity
units sigma_step = L * sigma_bar (the bar return's standard deviation), so the predicted forgiveness per path is
P(ruin) x 0.5826 x L x sigma_bar. Compared on fair placebo paths with EURUSD and crypto magnitudes, and on Gaussian bars (where the
constant should be close)."""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
BETA = 0.5826
N, NB = 20_000, 88


def sim(draw, L, rng):
    r = draw(rng, (N, NB - 1))                                   # bar returns, fair
    p = np.cumprod(np.c_[np.ones(N), 1 + r], axis=1)
    eq = 1 + L * (p[:, 1:] - 1)                                   # equity, units of initial equity, no fees
    hit = eq <= 0
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    ruined = first >= 0
    over = np.zeros(N)
    over[ruined] = -eq[ruined, first[ruined]]                     # forgiven at the stop: how far below zero the close landed
    return ruined.mean(), over.mean(), r.std()


eur = np.load(HERE / "eurusd_mags.npy")
cry = np.load(HERE / "crypto_mags.npy")
markets = {
    "EURUSD mags": lambda rng, sh: rng.choice([-1, 1], size=sh) * eur[rng.integers(0, len(eur), size=sh)],
    "crypto 2021 mags": lambda rng, sh: rng.choice([-1, 1], size=sh) * cry[rng.integers(0, len(cry), size=sh)],
    "Gaussian, sd 0.3%": lambda rng, sh: rng.normal(0, 0.003, size=sh),
}
rng = np.random.default_rng(7)
print("forgiven per path (units of initial equity): measured vs predicted P(ruin) x 0.5826 x L x sigma_bar")
print(f"{'market':<20} {'L':>4} {'P(ruin)':>8} {'measured':>9} {'predicted':>9} {'ratio':>6}")
for name, draw in markets.items():
    levs = (2, 5, 10) if "crypto" in name else (20, 50, 100, 200)
    for L in levs:
        pr, ov, sd = sim(draw, L, rng)
        pred = pr * BETA * L * sd
        print(f"{name:<20} {L:>4} {pr:8.3f} {ov:9.4f} {pred:9.4f} {ov / pred if pred > 0 else float('nan'):6.2f}")
