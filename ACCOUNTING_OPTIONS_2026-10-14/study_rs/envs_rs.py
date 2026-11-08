"""Study RS environments (PREREGISTRATION_RS.md): Study T's edge-market sleeve with the REAL sign of every price move.

Study T (study_t/envs_t.EdgeEnvT) and the twin-reward studies after it use real Danish spread magnitudes |s_t| with an independent fair sign per
quarter and a simulated signal, so the sign fairness that SPR needs holds by construction. Study RS keeps the sleeve, fixed sizing from K0, the
liquidity cap, the leverage grid, the costs fee*|q| + kappa*q^2, the zero floor, the observation, the reward modes and the twin hook
`flip_next` (all inherited unchanged from EdgeEnvT), and replaces the simulated parts with the market's own data:

    move    the real spread s_t (imbalance price minus day-ahead price, EUR/MWh): magnitude |s_t| and sign(s_t) in {-1, 0, +1}
    signal  z_t = sign(s_{t-LAG}), the realized spread LAG = 8 quarters (two hours) earlier, in {-1, 0, +1}

    kind 'edge'     the real signal of the same quarter
    kind 'placebo'  the real moves of the window with the signal of a different window at least two episodes away (same marginal law,
                    uninformative about the moves)

Settings: RS_DK1 (DK1, kappa 6) and RS_DK2 (DK2, kappa 6), Study A's train and test splits.

zero_starts=True (training environments of the registered runs): half of the episodes whose start equity is not given explicitly start at
zero reported equity, exactly as study_v/rewards_v.EdgeEnvZ (Study V Part C). Every test episode passes its start equity explicitly.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np

HERE = Path(__file__).resolve().parent
ACC = HERE.parent / "github" / "accounting-options-rl" / "ACCOUNTING_OPTIONS_2026-10-14"
STUDY_T = ACC / "study_t"
STUDY_V = ACC / "study_v"
for _p in (str(STUDY_V), str(STUDY_T)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from envs_t import E0_TRAIN, EPISODE, FEE, K0, LEV, Q1, EdgeEnvT, load_market  # noqa: E402,F401

LAG = 8
SETTINGS = {
    "RS_DK1": dict(region="DK1", kappa=6.0),
    "RS_DK2": dict(region="DK2", kappa=6.0),
}
PLACEBO_MIN_SHIFT = 2 * EPISODE


@lru_cache(maxsize=None)
def market_arrays(setting: str, split: str):
    """(|s|, sign(s), z, cap) for the setting's region and split; read-only arrays."""
    m = load_market(SETTINGS[setting]["region"], split, "RM")
    s = np.asarray(m.spread, dtype=float)
    abs_s, eps = np.abs(s), np.sign(s)
    z = np.zeros_like(s)
    z[LAG:] = np.sign(s[:-LAG])                  # the first LAG quarters have no signal; episodes start at t >= 200
    cap = np.asarray(m.cap_mwh, dtype=float)
    for a in (abs_s, eps, z, cap):
        a.flags.writeable = False
    return abs_s, eps, z, cap


class EdgeEnvRS(EdgeEnvT):
    """EdgeEnvT with real signs and the lagged real signal. Only the data, the placebo signal and the optional zero starts differ; step and
    _obs are EdgeEnvT's."""

    def __init__(self, setting: str, split: str, kind: str, mode: str, seed: int = 0, zero_starts: bool = False):
        gym.Env.__init__(self)
        self.zero_starts = zero_starts
        assert mode in ("reported", "booked") and kind in ("edge", "placebo") and setting in SETTINGS
        self.setting = setting
        self.abs_s, self.eps, self.zz, self.cap = market_arrays(setting, split)
        self.kappa = SETTINGS[setting]["kappa"]
        self.n = len(self.abs_s)
        self.kind = "edge"                       # EdgeEnvT.reset then takes the real moves and signal of the window
        self.rs_kind, self.mode = kind, mode
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(4,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self.flip_next = False

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        if self.zero_starts and "e0" not in opts and self.rng.random() < 0.5:     # the draw order of rewards_v.EdgeEnvZ
            opts["e0"] = 0.0
        obs, info = super().reset(seed=seed, options=opts)
        if self.rs_kind == "placebo":
            while True:
                t0s = int(self.rng.integers(200, self.n - EPISODE))
                if abs(t0s - self.t0) >= PLACEBO_MIN_SHIFT:
                    break
            self.z_ep = self.zz[t0s:t0s + EPISODE]
            obs = self._obs()
        return obs, info
