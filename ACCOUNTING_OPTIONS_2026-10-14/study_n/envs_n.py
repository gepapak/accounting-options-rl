"""Study N environments: the Study B/F placebo sleeve with two new accounting rules, each with a matched control.

Z  zombie window (settlement lag W quarter-hours). When the booked ledger L first reaches <= 0 the breach is detected
   W quarter-hours later; trading continues meanwhile; at detection the sleeve closes.
     ZFG  forgiven: once insolvent, the simulator reports the owner's claim max(L, 0); at closure the deficit is
          forgiven (pilot 1 paid the forgiveness as a lump sum at detection instead; same terminal payoff)
     ZBK  booked:   reported equity = L throughout
   Observation: [L/K0 clipped to [-3, 5], k/W] where k = quarter-hours left in the window (0 when solvent).
T  terminal limited liability (horizon H quarter-hours). The ledger may go negative; trading continues to the horizon.
     TFG  forgiven: reported equity at the horizon = max(L_H, 0)   (Merton's call)
     TBK  booked:   reported equity at the horizon = L_H
   Observation: [L/K0 clipped to [-3, 5], tau/H] where tau = quarter-hours to the horizon.
Everything else is Study F: placebo (sign-randomized) Danish spreads, fixed sizing from K0 = 20,000 EUR, leverage
levels +-{0, 0.25, 1, 4, 16} x 0.25 MWh, liquidity cap 25 % of balancing volume, fee FEE_N = 1.0 EUR/MWh (Study H level), reward = change
in reported equity in % of K0. Within each setting the two arms differ ONLY in the closing bonus.
"""
from __future__ import annotations
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
LIAB = Path(r"C:/Users/gepap/Desktop/Thesis/Paper1/rl-trading-reality-check-main/LIABILITY_STUDY_2026-09-29")
sys.path.insert(0, str(LIAB))
from envs_b import K0, LEV, Q1, load_market  # noqa: E402,F401

FEE_N = 1.0          # EUR/MWh (Study H level; Study F's 0.10 left booked controls unlearnable in pilot 1)

W_LAG = 96          # 1 day of quarter-hours
H_TERM = 192        # 2 days of quarter-hours
EP_Z = 672          # zombie-setting training episodes: at most one week
ARMS = ("ZFG", "ZBK", "TFG", "TBK")


class LagEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, market, arm: str, train: bool, seed: int = 0):
        super().__init__()
        assert arm in ARMS
        self.m, self.arm, self.train = market, arm, train
        self.setting, self.forgiven = arm[0], arm[1:] == "FG"
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(np.array([-3.0, 0.0], np.float32), np.array([5.0, 1.0], np.float32))
        self.rng = np.random.default_rng(seed)

    # ---- state helpers
    def _obs(self):
        c = (self.k / W_LAG) if self.setting == "Z" else (self.tau / H_TERM)
        return np.array([np.clip(self.L / K0, -3.0, 5.0), c], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = options or {}
        r = self.rng
        if self.train:
            if r.random() < 0.5:
                L = K0 * float(np.exp(r.uniform(np.log(0.05), np.log(2.0))))
                insolvent = False
            else:
                L = -K0 * float(10 ** r.uniform(-2.0, 0.0))
                insolvent = True
            if self.setting == "Z":
                self.k = int(r.integers(1, W_LAG + 1)) if insolvent else 0
                self.tau, self.left = 0, EP_Z
            else:
                self.k, self.tau = 0, int(r.integers(1, H_TERM + 1))
                self.left = self.tau
            self.t = int(r.integers(200, self.m.n - max(EP_Z, H_TERM) - 1))
        else:
            L = K0 * float(o.get("e0", 1.0))
            self.k = 0
            self.t = int(o.get("t0", 200))
            self.tau = int(o.get("tau0", H_TERM)) if self.setting == "T" else 0
            self.left = int(o.get("steps", self.m.n - self.t)) if self.setting == "Z" else self.tau
        self.L, self.L0 = L, L
        self.closed, self.forgiven_amt, self.zombie_steps, self.zombie_max = False, 0.0, 0, 0
        return self._obs(), {}

    def step(self, action):
        m, t = self.m, self.t
        lev = float(LEV[int(action)])
        q = float(np.clip(lev * Q1, -m.cap_mwh[t], m.cap_mwh[t]))
        pnl = q * m.spread[t] - FEE_N * abs(q)
        if self.setting == "Z" and self.k > 0:
            self.zombie_steps += 1
            self.zombie_max += int(abs(lev) == LEV.max())
        before = self.reported()
        self.L += pnl
        terminated = False
        if self.setting == "Z":
            if self.k > 0:
                self.k -= 1
                if self.k == 0:                      # detection: the sleeve closes
                    terminated, self.closed = True, True
            elif self.L <= 0.0:
                self.k = W_LAG                       # breach; detected W quarter-hours later
            reward = self.reported() - before        # ZFG reports the owner's claim max(L, 0) once insolvent
            if terminated and self.forgiven:
                self.forgiven_amt = max(-self.L, 0.0)
        else:
            reward = pnl
        if self.setting == "T":
            self.tau -= 1
            if self.tau == 0:
                terminated = True
                if self.forgiven:
                    bonus = max(-self.L, 0.0)
                    self.forgiven_amt += bonus
                    reward += bonus
        self.t += 1
        self.left -= 1
        truncated = (not terminated) and (self.left <= 0 or self.t >= m.n - 1)
        return self._obs(), float(reward / K0 * 100.0), terminated, truncated, {"pnl": pnl}

    def reported(self):
        if self.setting == "Z":
            insolvent_window = self.k > 0 or self.closed
            return max(self.L, 0.0) if (self.forgiven and insolvent_window) else self.L
        return self.L + self.forgiven_amt
