"""Tests of LineRewardWrapper on a small simulator with two floored cash writes: its reward equals the booked P&L on every step, the transition stays the simulator's own, and the adaptive mode returns the reported reward where nothing is forgiven. Exit code 0 = all checks pass.

    python test_line_wrapper.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reward_diagram import LineRewardWrapper  # noqa: E402

FEE = 0.1


class Toy(gym.Env):
    """A gross P&L and then a fee, each written with its own floor at zero; the reward is the change of cash."""

    def __init__(self, seed: int, cash0: float):
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-1e12, 1e12, shape=(1,), dtype=np.float64)
        self.moves = np.random.default_rng(seed).normal(0.0, 1.0, 400)
        self.cash0 = cash0
        self.scale = 1.0

    def reset(self, *, seed=None, options=None):
        self.t, self.cash = 0, float(self.cash0)
        return np.array([self.cash]), {}

    def step(self, action):
        q = 2.0 * (int(action) - 1)
        gross = q * self.moves[self.t] * self.scale
        cost = FEE * abs(q)
        pre = self.cash
        cash = max(pre + gross, 0.0)
        self.cash = max(cash - cost, 0.0)
        self.t += 1
        self.scale = 1.0
        return np.array([self.cash]), self.cash - pre, False, self.t >= len(self.moves), dict(own_cost=cost, booked=q * self.moves[self.t - 1] - cost)


def set_line(clone, t):
    clone.scale = t


def main() -> int:
    failures = 0
    steps = forgiven = 0
    for seed in range(4):
        for cash0 in (0.0, 1.0, 50.0):
            for adaptive in (False, True):
                env = LineRewardWrapper(Toy(seed, cash0), set_line, adaptive=adaptive)
                plain = Toy(seed, cash0)
                env.reset()
                plain.reset()
                actions = np.random.default_rng(seed + 100).integers(0, 3, 300)
                for a in actions:
                    o, r, _, _, info = env.step(a)
                    op, rp, _, _, _ = plain.step(a)
                    steps += 1
                    failures += int(abs(float(o[0]) - float(op[0])) > 1e-12)                       # the transition is the simulator's own
                    failures += int(abs(info["reported_reward"] - rp) > 1e-12)
                    failures += int(abs(r - info["booked"]) > 1e-6)                                # the reward is the booked P&L
                    forgiven += int(abs(rp - info["booked"]) > 1e-9)
                    if adaptive and abs(rp - info["booked"]) <= 1e-9 and info["line_gross"] is None:
                        failures += int(abs(r - rp) > 1e-12)                                       # nothing forgiven: the reported reward passes through
    print(f"steps {steps}, of which the reward forgave something on {forgiven}; failures {failures}")
    if forgiven == 0 or failures:
        print("FAILED")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
