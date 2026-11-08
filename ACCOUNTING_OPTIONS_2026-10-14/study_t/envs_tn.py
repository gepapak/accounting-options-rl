"""Study T, Part C environments: Study N's settlement-lag and terminal-liability environments (study_n/envs_n.py) with twin support.

A faithful copy of `LagEnv` (the registered environment; study_n/envs_n.py hard-codes the analysis machine's path to the liability study,
so it is copied rather than imported). The only addition is `flip_next`, which makes the next step use the opposite sign of its exogenous
spread (the counterfactual used by SPR and TIR). `tests` in `unit_test_envs_tn.py` check that the copy reproduces the registered
environment step for step.

    arms  ZFG / ZBK (zombie window, deficit forgiven / booked), TFG / TBK (terminal limited liability, forgiven / booked)
    reward in percent of K0; observation [L/K0 clipped to [-3, 5], k/W (zombie) or tau/H (terminal)]
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

LIAB = Path(__file__).resolve().parents[2] / "LIABILITY_STUDY_2026-09-29"
sys.path.insert(0, str(LIAB))
from envs_b import K0, LEV, Q1, load_market  # noqa: E402,F401

FEE_N = 1.0
W_LAG = 96
H_TERM = 192
EP_Z = 672
ARMS = ("ZFG", "ZBK", "TFG", "TBK")


class LagEnvT(gym.Env):
    metadata: dict = {}

    def __init__(self, market, arm: str, train: bool, seed: int = 0):
        super().__init__()
        assert arm in ARMS
        self.m, self.arm, self.train = market, arm, train
        self.setting, self.forgiven = arm[0], arm[1:] == "FG"
        self.action_space = gym.spaces.Discrete(len(LEV))
        self.observation_space = gym.spaces.Box(np.array([-3.0, 0.0], np.float32), np.array([5.0, 1.0], np.float32))
        self.rng = np.random.default_rng(seed)
        self.flip_next = False

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
        self.flip_next = False
        return self._obs(), {}

    def step(self, action):
        m, t = self.m, self.t
        lev = float(LEV[int(action)])
        q = float(np.clip(lev * Q1, -m.cap_mwh[t], m.cap_mwh[t]))
        spread = m.spread[t] * (-1.0 if self.flip_next else 1.0)
        self.flip_next = False
        pnl = q * spread - FEE_N * abs(q)
        if self.setting == "Z" and self.k > 0:
            self.zombie_steps += 1
            self.zombie_max += int(abs(lev) == LEV.max())
        before = self.reported()
        self.L += pnl
        terminated = False
        if self.setting == "Z":
            if self.k > 0:
                self.k -= 1
                if self.k == 0:
                    terminated, self.closed = True, True
            elif self.L <= 0.0:
                self.k = W_LAG
            reward = self.reported() - before
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
        return self._obs(), float(reward / K0 * 100.0), terminated, truncated, {"pnl": pnl, "cost": FEE_N * abs(q)}

    def reported(self):
        if self.setting == "Z":
            insolvent_window = self.k > 0 or self.closed
            return max(self.L, 0.0) if (self.forgiven and insolvent_window) else self.L
        return self.L + self.forgiven_amt
