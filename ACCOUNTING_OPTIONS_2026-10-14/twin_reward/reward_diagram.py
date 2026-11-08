"""The reward diagram: a simulator step's reported reward read as a function of the market input.

Freeze one pre-step state and action and move the exogenous input along a line, input(t) = reference + t * (input - reference): t = 1 is the real
step, t = -1 the sign twin of Studies S to Z, t = 0 no move. phi(t) is the reported reward on a clone at t. The trainer needs a clone that
accepts a changed input, and the own cost wherever the account cannot pay it: no ledger, no payoff formula, no position, no account structure.

For a simulator whose gross P&L is linear in t and whose accounting is a program of floored cash writes, phi is the payoff diagram of the option
book that the accounting hands the agent (manuscript section "The reward diagram"):

    phi(t) = booked(t) + credit(t),   booked(t) = t * g - C,   credit(t) >= 0 and convex in t,

so phi is convex and piecewise affine. Reading it:

    no kink on [a, b]             no floor binds anywhere between a and b (two credit-free points certify the whole interval between them)
    kink position and slope jump  distance to a hidden floor in multiples of the realized move, and the exposure behind it
    flat tail at level -e         a hard floor; e is the loss the floor lets through: the hidden balance, or the isolated margin
    opposite tail slopes          accounts of opposite exposure, each floored; the net gross is their sum
    tails of one sign             part of the gross stays unfloored (a mark-to-market bucket, partial forgiveness); the gross is the steeper one
    concave kink                  not a floor: a contract cap or a clip in the gross, which no forgiving accounting produces
    jump                          the reward is replaced on a branch, for example a bankruptcy step that is not rewarded

`read_diagram` turns a scan into these quantities. `tail_gross` and `line_estimate` give the booked P&L. `LineRewardWrapper` trains on it.
Everything here is reward-only: the ledger is never read. Try the direct calculation first where the payoff formula is known and trusted.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np


def geometric_grid(t_max: float = 1024.0, base: float = 2.0) -> np.ndarray:
    """0, +-1, +-base, +-base^2, ..., +-t_max, sorted."""
    pos = [1.0]
    while pos[-1] * base <= t_max * (1.0 + 1e-12):
        pos.append(pos[-1] * base)
    pos = np.array(pos)
    return np.concatenate((-pos[::-1], [0.0], pos))


def scan(query: Callable[[float], float], ts: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """phi on the grid ts (any order); returns (sorted ts, phi). `query(t)` is one reward query on a fresh clone."""
    ts = np.sort(np.asarray(ts, dtype=float))
    return ts, np.array([float(query(float(t))) for t in ts])


@dataclass
class Diagram:
    ts: np.ndarray
    phi: np.ndarray
    slopes: np.ndarray                     # slope on each interval of the grid
    convex_kinks: list = field(default_factory=list)   # (node t, slope jump > 0)
    concave_kinks: list = field(default_factory=list)  # (node t, slope jump < 0)
    s_plus: float = 0.0                    # largest stable slope moving outward on t > 0
    s_minus: float = 0.0                   # largest stable slope moving outward on t < 0
    level_left: Optional[float] = None     # value of a flat tail on the left, if there is one
    level_right: Optional[float] = None    # value of a flat tail on the right, if there is one
    tol: float = 0.0

    @property
    def gross(self) -> float:
        return tail_gross(self.s_plus, self.s_minus)

    @property
    def hidden_level(self) -> Optional[float]:
        """The loss a hard floor lets through: minus the level of a flat tail below phi(0), or None. Reads the hidden balance or margin."""
        i0 = int(np.argmin(np.abs(self.ts)))
        cands = [v for v in (self.level_left, self.level_right) if v is not None and v <= self.phi[i0] + self.tol]
        return -min(cands) if cands else None


SLOPE_REL = 1e-6      # two slopes count as equal within this relative difference (rewards are differences of large balances, so rounding scales with them)


def read_diagram(ts, phi, rel_tol: float = 1e-9, abs_tol: float = 1e-6) -> Diagram:
    """Read a scan. ts must be sorted; the grid should contain 0 and points on both sides. Kinks are located to the grid node at which the slope
    changes; `kink_position` locates a kink exactly when each side has an affine piece. A slope is *stable* where two consecutive intervals agree."""
    ts = np.asarray(ts, dtype=float)
    phi = np.asarray(phi, dtype=float)
    assert np.all(np.diff(ts) > 0), "the grid must be sorted"
    s = np.diff(phi) / np.diff(ts)
    tol = rel_tol * float(np.max(np.abs(phi[np.abs(ts) <= 16.0]))) + abs_tol if np.any(np.abs(ts) <= 16.0) else abs_tol
    # a slope difference is significant if it moves the value over the shorter neighbouring interval by more than a tolerance local to that neighbourhood,
    # and exceeds SLOPE_REL of the slopes themselves
    dt = np.minimum(np.diff(ts)[:-1], np.diff(ts)[1:])
    jump = np.diff(s)
    local = rel_tol * np.maximum.reduce([np.abs(phi[:-2]), np.abs(phi[1:-1]), np.abs(phi[2:])]) + abs_tol
    sig = (np.abs(jump) * dt > local) & (np.abs(jump) > SLOPE_REL * np.maximum(np.abs(s[:-1]), np.abs(s[1:])))
    d = Diagram(ts=ts, phi=phi, slopes=s, tol=tol)
    d.convex_kinks = [(float(ts[i + 1]), float(jump[i])) for i in np.flatnonzero(sig & (jump > 0))]
    d.concave_kinks = [(float(ts[i + 1]), float(jump[i])) for i in np.flatnonzero(sig & (jump < 0))]

    def stable_largest(idx):
        """Largest-magnitude slope that two consecutive intervals agree on, moving outward along idx (interval indices)."""
        best = 0.0
        for a, b in zip(idx[:-1], idx[1:]):
            len_min = min(ts[a + 1] - ts[a], ts[b + 1] - ts[b])
            if abs(s[a] - s[b]) <= SLOPE_REL * max(abs(s[a]), abs(s[b])) + tol / len_min and abs(s[b]) > abs(best):
                best = float(s[b])
        return best

    i0 = int(np.argmin(np.abs(ts)))
    right = list(range(i0, len(s)))
    left = list(range(i0 - 1, -1, -1))
    d.s_plus = stable_largest(right) if len(right) > 1 else (float(s[right[0]]) if right else 0.0)
    d.s_minus = stable_largest(left) if len(left) > 1 else (float(s[left[0]]) if left else 0.0)
    if len(s) >= 2:
        if abs(s[0]) * (ts[1] - ts[0]) <= tol and abs(s[1]) * (ts[2] - ts[1]) <= tol:
            d.level_left = float(phi[0])
        if abs(s[-1]) * (ts[-1] - ts[-2]) <= tol and abs(s[-2]) * (ts[-2] - ts[-3]) <= tol:
            d.level_right = float(phi[-1])
    return d


def tail_gross(s_plus: float, s_minus: float) -> float:
    """The net gross from the tail slopes: their sum if they have opposite signs (accounts of opposite exposure, each floored), else the steeper one
    (one exposure, possibly with an unfloored part). Not identified with an unfloored channel and floored accounts of the opposite sign."""
    if s_plus * s_minus < 0.0:
        return s_plus + s_minus
    return s_plus if abs(s_plus) >= abs(s_minus) else s_minus


def far_tail_gross(query: Callable[[float], float], T: float) -> float:
    """Four queries at +-T and +-2T. Exact once T lies beyond every kink; within (E + 2A) / T of the net gross in the worst case (no unfloored channel)."""
    a_plus = (query(2.0 * T) - query(T)) / T
    a_minus = (query(-T) - query(-2.0 * T)) / T
    return tail_gross(a_plus, a_minus)


def minimax_estimate(query: Callable[[float], float], cost: float, Lambda: float) -> float:
    """The endpoint estimator (phi(L) - phi(-L)) / L - C: exact minimax over coherent account programs, error at most (E + A + 2 eta) / L, and no
    algorithm querying the whole of [-L, L] can guarantee less."""
    return (query(Lambda) - query(-Lambda)) / Lambda - cost


def line_estimate(query: Callable[[float], float], cost: float, r: Optional[float] = None, width_tol: float = 1e-9,
                  grid: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 1024.0)) -> tuple[float, int, Diagram | None]:
    """Adaptive reading of the booked P&L of one step: returns (estimate, reward queries used, diagram or None).

    Stage 1, the twin: phi(1) and phi(-1). If the bracket's width phi(1) + phi(-1) + 2C vanishes nothing is forgiven on either branch, and phi(1) is the
    booked P&L. Stage 2, only where the width is positive: the scan outward until a stable slope is found on some side, then gross - C."""
    tol = width_tol * (1.0 + abs(cost))
    r1 = float(query(1.0)) if r is None else float(r)
    rm = float(query(-1.0))
    n = 2 if r is None else 1
    if abs(r1 + rm + 2.0 * cost) <= tol * 1.0:
        return r1, n, None
    ts = [-1.0, 0.0, 1.0]
    phis = {-1.0: rm, 1.0: r1}
    phis[0.0] = float(query(0.0))
    n += 1
    for t in grid:
        phis[t] = float(query(t))
        phis[-t] = float(query(-t))
        n += 2
        ts = sorted(phis)
        d = read_diagram(np.array(ts), np.array([phis[x] for x in ts]))
        if t >= 4.0 and (d.s_plus != 0.0 or d.s_minus != 0.0):
            return d.gross - cost, n, d
    return d.gross - cost, n, d


def kink_position(t_left: float, phi_left: float, slope_left: float, t_right: float, phi_right: float, slope_right: float) -> float:
    """Exact position of a kink between two affine pieces known by a point and a slope on each side."""
    return (phi_right - phi_left + slope_left * t_left - slope_right * t_right) / (slope_left - slope_right)


class LineRewardWrapper(gym.Wrapper):
    """Train on the tail-slope reward (TSR): far_tail_gross - C, with four queries per step (or `adaptive=True`: one query at t = -1, and the four more
    only where the twin's width is not zero). Reward-only: the ledger is never read. Requires a callable `set_line(clone, t)` that moves the clone's next
    input along the line, and `own_cost(env, action, info)` (default `info["own_cost"]`). The transition stays the simulator's own: only the reward changes."""

    def __init__(self, env: gym.Env, set_line: Callable, own_cost: Optional[Callable] = None, T: float = 1e9, adaptive: bool = False):
        super().__init__(env)
        self._set_line = set_line
        self._own_cost = own_cost or (lambda env, action, info: info["own_cost"])
        self.T = float(T)
        self.adaptive = adaptive

    def step(self, action):
        pre = copy.deepcopy(self.env)                     # the state before the step; every query starts from a copy of it

        def q(t: float) -> float:
            clone = copy.deepcopy(pre)
            self._set_line(clone, t)
            return float(clone.step(action)[1])

        obs, r, terminated, truncated, info = self.env.step(action)
        r = float(r)
        c = float(self._own_cost(self.env, action, info))
        if self.adaptive and abs(r + q(-1.0) + 2.0 * c) <= 1e-9 * (1.0 + abs(r) + abs(c)):
            return obs, r, terminated, truncated, dict(info, reported_reward=r, line_gross=None)    # nothing is forgiven on either branch
        g_hat = far_tail_gross(q, self.T)
        return obs, g_hat - c, terminated, truncated, dict(info, reported_reward=r, line_gross=g_hat)
