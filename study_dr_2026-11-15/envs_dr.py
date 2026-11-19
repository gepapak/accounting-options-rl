"""Study DR environments (PREREGISTRATION_DR.md): Study GC's forex arm (`../study_gc_2026-11-14/envs_gc.py`, imported read-only; nothing in
that folder is changed), with one change per arm, made in the training distribution or the account:

    V2, V4, V8, V16   leverage 100 (the package default); in training, every episode's placebo magnitudes are multiplied by
                      k ~ log-uniform[1, k_hi] (k_hi = 2, 4, 8, 16), capped at 0.9. The agent does not observe k.
    L30, L500         nominal magnitudes (k = 1: Study GC's training paths, draw for draw); the account's leverage is 30 (the EU retail
                      cap for major currency pairs) or 500 (common at offshore retail brokers), fixed for the whole run and the tests.

Everything else is Study GC's forex arm: EURUSD from gym-mtsim's bundled forex file, the fee of its registered `forex-hedge-v0`
(0.0003), 10,000 USD, stop-out level 0.2, hedge mode, one order at a time, window 2, the flat five-number observation, the published
action box and reward (divided by 100), training balances log-uniform in [0.05, 2] x 10,000 USD. Conditions ORIG (gym-mtsim as
published) and FLPATCH (the published tick() without its three balance-floor lines), as in Study GC.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

GC_DIR = Path(__file__).resolve().parent.parent / "study_gc_2026-11-14"
if str(GC_DIR) not in sys.path:
    sys.path.insert(0, str(GC_DIR))
import envs_gc as G  # noqa: E402

MARKET = "forex"
ARMS = {
    "V2": dict(k_hi=2.0, leverage=100.0),
    "V4": dict(k_hi=4.0, leverage=100.0),
    "V8": dict(k_hi=8.0, leverage=100.0),
    "V16": dict(k_hi=16.0, leverage=100.0),
    "L30": dict(k_hi=1.0, leverage=30.0),
    "L500": dict(k_hi=1.0, leverage=500.0),
}
K_CAP = 0.9                                   # a scaled bar never moves the price by more than 90%
K_STRESS, STRESS_SEED = 4.0, G.TEST_SEED + 4  # the stressed test bank (the pilot's)
EGRID, BALANCE, N_TEST = G.EGRID, G.BALANCE, G.N_TEST


def scaled_placebo(rng: np.random.Generator, k: float) -> np.ndarray:
    """Study GC's placebo path with every drawn magnitude multiplied by k (capped at K_CAP). For k = 1 it draws exactly what
    envs_gc.placebo_closes draws from the same generator state."""
    b = G.bundle(MARKET)
    n = len(b["index"])
    mags = np.minimum(k * b["mags"][rng.integers(0, len(b["mags"]), size=n - 1)], K_CAP)
    signs = rng.choice([-1.0, 1.0], size=n - 1)
    return b["p0"] * np.cumprod(np.r_[1.0, 1.0 + signs * mags])


def nominal_bank() -> np.ndarray:
    """Study GC's forex test bank (300 paths, generator seed 20261114)."""
    return G.test_bank(MARKET)


def stressed_bank() -> np.ndarray:
    """300 paths with every magnitude times 4 (generator seed 20261118), the pilot's stressed bank."""
    rng = np.random.default_rng(STRESS_SEED)
    return np.stack([scaled_placebo(rng, K_STRESS) for _ in range(N_TEST)])


class DREnv(G.GCEnv):
    """Study GC's wrapper for one Study DR arm."""

    def __init__(self, arm: str, condition: str, train: bool, seed: int):
        assert arm in ARMS and condition in ("ORIG", "FLPATCH")
        super().__init__(MARKET, condition, train=train, seed=seed, audit=False)
        self.arm, self.k_hi, self.leverage = arm, ARMS[arm]["k_hi"], ARMS[arm]["leverage"]
        self.env.unwrapped.original_simulator.leverage = self.leverage     # MtEnv copies the original simulator at every reset
        self.last_k = 1.0

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        self.last_k = 1.0
        if "path" not in opts and self.train and self.k_hi > 1.0:
            self.last_k = float(np.exp(self.rng.uniform(0.0, np.log(self.k_hi))))
            opts["path"] = scaled_placebo(self.rng, self.last_k)
        return super().reset(seed=seed, options=opts)


def vmax(balance: float, price: float, leverage: float) -> float:
    """envs_gc.vmax_closed at a given leverage: the largest volume the margin check accepts from a flat account."""
    m, info = G.MARKETS[MARKET], G.bundle(MARKET)["info"]
    per_unit = info.trade_contract_size * (price / leverage + m["fee"])
    return float(np.floor(max(balance, 0.0) / per_unit / info.volume_step + 1e-9) * info.volume_step)


def exposure(predict, arm: str, e: float) -> float:
    """Study GC's exposure probe (envs_gc.exposure) at the arm's leverage; at leverage 100 it is envs_gc.exposure itself. The share of
    the margin-feasible maximum that the deterministic policy opens from a flat account with equity e x 10,000 USD at the bundle's
    first close; 0 if it holds or the order would be refused."""
    lev = ARMS[arm]["leverage"]
    a = np.asarray(predict(np.array([[e, e, 0.0, 0.0, 0.0]], dtype=np.float32)), dtype=float).ravel()
    if 1.0 / (1.0 + np.exp(-a[1])) > 0.5:
        return 0.0
    info, p0 = G.bundle(MARKET)["info"], G.bundle(MARKET)["p0"]
    v = float(np.clip(abs(a[2]), info.volume_min, info.volume_max))
    v = round(v / info.volume_step) * info.volume_step                 # the environment's own rounding
    feasible = vmax(e * BALANCE, p0, lev)
    if feasible < info.volume_min:
        return 0.0
    return float(v / feasible) if v <= feasible + 1e-9 else 0.0
