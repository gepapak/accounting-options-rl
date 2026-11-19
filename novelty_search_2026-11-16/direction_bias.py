"""Exploratory, no engine and no learner: at backtesting.py's DEFAULT margin (1, no leverage), does the zero floor make a short more
attractive than a long on a fair market? A 1x long can lose at most the account (the price cannot go below zero), but a 1x short can
lose more, and the floor forgives the excess. Fair placebo paths: 88 bars, magnitudes = gym-mtsim's bundled EURUSD daily moves x k
(k = 1, 4, 8, 16, capped at 90%) or the bundled BTC/ETH/BCH daily moves (x1, x2), fair signs. Rule (as in backtesting.py): equity set to 0
and the run stopped at the first close with equity <= 0. Fee 2 bp per side. Reports the mean reported return of the 1x long and the 1x
short, and the booked (ledger) means, over 20,000 paths each."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
FEE, N, NB = 0.0002, 20_000, 88
eur = np.load(HERE / "eurusd_mags.npy")
crypto = None
try:
    import pandas as pd
    cl = pd.read_csv(HERE / "real_closes_gym_mtsim_2021.csv")
    rets = np.concatenate([np.abs(np.diff(cl[s].dropna().to_numpy()) / cl[s].dropna().to_numpy()[:-1]) for s in ("BTCUSD", "ETHUSD", "BCHUSD")])
    crypto = rets[rets > 0]
except Exception as ex:
    print("crypto magnitudes unavailable:", ex)


def run(mags, k, side, rng):
    m = np.minimum(k * mags[rng.integers(0, len(mags), size=(N, NB - 1))], 0.9)
    s = rng.choice([-1.0, 1.0], size=(N, NB - 1))
    p = np.cumprod(np.c_[np.ones(N), 1.0 + s * m], axis=1)            # price path, p0 = 1
    units = 0.9999 / (1 + FEE)                                          # 1x of the cash (cash = 1)
    pnl = side * units * (p[:, 1:] - p[:, [0]])                         # entry at bar 0's close
    eq = 1 - units * FEE + pnl
    ledger = eq[:, -1] - units * p[:, -1] * FEE - 1
    hit = (eq <= 0).any(axis=1)
    reported = np.where(hit, -1.0, ledger)
    return 100 * reported.mean(), 100 * ledger.mean(), 100 * hit.mean(), 100 * reported.std() / np.sqrt(N)


rng = np.random.default_rng(20261116)
cases = [("EURUSD x1", eur, 1), ("EURUSD x4", eur, 4), ("EURUSD x8", eur, 8), ("EURUSD x16", eur, 16)]
if crypto is not None:
    cases += [("BTC/ETH/BCH 2021 x1", crypto, 1), ("BTC/ETH/BCH 2021 x2", crypto, 2)]
print("1x positions held 87 bars; mean returns in % of cash (20,000 fair paths each); SE of the reported mean in brackets")
print(f"{'market':<22} {'long reported':>14} {'short reported':>15} {'short - long':>12} | {'long booked':>11} {'short booked':>12} | short reaches 0")
for name, mags, k in cases:
    lr, lb, lh, lse = run(mags, k, +1.0, rng)
    sr, sb, sh, sse = run(mags, k, -1.0, rng)
    print(f"{name:<22} {lr:+9.2f} [{lse:.2f}] {sr:+9.2f} [{sse:.2f}] {sr - lr:+11.2f} | {lb:+11.2f} {sb:+12.2f} | {sh:6.1f}%")
