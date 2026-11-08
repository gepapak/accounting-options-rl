"""Study EXT environments: the official Gym-Trading-Env, unmodified, driven only through its public constructor, its data feed and its documented custom reward.

The source in vendor/gym_trading_env is byte-identical to commit a86fc03d29759374fae2db7df4d878485808a730 (checked against SOURCE_MANIFEST.json at import). This harness does not
subclass or patch it. It sets the price array the engine reads (the data feed) and the documented `reward_function`, a linear change of wealth, percent of the initial wealth.

Accounting of the engine (read from its source, and read again from the reward alone in the scan): on a step that takes the portfolio's valuation to zero or below the engine ends the
episode and does not call the reward function, so the reward stays 0: the whole loss of that step is not rewarded. Negative wealth is kept in the portfolio and in the public history.

Market. Daily moves of BTC/USD from the engine's official example file (complete UTC days), magnitudes real, signs simulated afresh in every episode, so that no policy can profit in the
placebo market. `edge`: a signal z observed before acting with P(z = sign of the next move) = P_EDGE. `real`: the observed signs.

Rewards (`reward=`, percent of W0): REP the engine's reward; LEDGER the change of the public wealth, every step; SPR, TIR the twin rewards; TSR the tail-slope reward. C is read from the
reward at no move, phi(0), on a clone: no declared cost and no formula. None of SPR, TIR and TSR reads the wealth.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor"
DATA = HERE / "data" / "BTC_USD-Hourly.csv"
DATA_SHA256 = "b5ccfdef2aa852ded1ea03378c1777757d2c27a81d53e0fc81af0011fed9cccf"


def verify_vendor() -> int:
    man = json.loads((VENDOR / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    n = 0
    for f in man["files"]:
        p = f["path"]
        if not p.startswith("src/"):
            continue
        local = VENDOR / p.replace("src/", "")
        assert hashlib.sha256(local.read_bytes()).hexdigest() == f["sha256"], f"vendored file differs: {p}"
        n += 1
    return n


verify_vendor()
sys.path.insert(0, str(VENDOR))
from gym_trading_env.environments import TradingEnv  # noqa: E402

# The package sets warnings.filterwarnings("error") for the whole process at import. Reset it so that a warning elsewhere (a library's deprecation) does not end a run.
warnings.resetwarnings()
warnings.simplefilter("ignore", category=DeprecationWarning)

W0 = 1000.0
EPISODE = 30
FEE = 0.0001
POSITIONS = (-10, -5, -2, -1, 0, 1, 2, 5, 10)
P_EDGE = 0.60
SPLIT_DATE = "2020-01-01"
T_FAR = 1e9


@lru_cache(maxsize=None)
def daily_returns():
    """Complete UTC days of the official hourly file: arrays of closes and simple daily returns, in time order, with their dates."""
    raw = DATA.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == DATA_SHA256, "the example file differs from the pinned bytes"
    df = pd.read_csv(DATA, usecols=["unix", "close"]).sort_values("unix")
    df["time"] = pd.to_datetime(df["unix"], unit="s", utc=True)
    df["day"] = df["time"].dt.floor("D")
    g = df.groupby("day")
    days = g["close"].agg(["last", "count"])
    days = days[days["count"] == 24]
    close = days["last"].to_numpy(dtype=float)
    ret = close[1:] / close[:-1] - 1.0
    dates = days.index[1:]
    return close, ret, dates


def split_moves(split: str):
    _, ret, dates = daily_returns()
    mask = dates < pd.Timestamp(SPLIT_DATE, tz="UTC") if split == "train" else dates >= pd.Timestamp(SPLIT_DATE, tz="UTC")
    return ret[mask]


def linear_reward(history) -> float:
    """The documented custom reward: the change of portfolio valuation, percent of the initial wealth."""
    return float((history["portfolio_valuation", -1] - history["portfolio_valuation", -2]) / W0 * 100.0)


def _f_wealth(h):
    return float(np.clip(h["portfolio_valuation", -1] / W0, -5.0, 5.0))


def _f_pos(h):
    return float(h["position", -1]) / 10.0


def _f_rem(h):
    return 1.0 - float(h["step", -1]) / EPISODE


def clone_core(core: TradingEnv, price_line: float | None = None) -> TradingEnv:
    """A clone of the engine's state. Equal to copy.deepcopy(core) for every reward and valuation (checked in test_envs_ext.py), but cheap. `price_line` = t moves the next
    price along the line through the realized move: next price = current + t * (next - current)."""
    c = copy.copy(core)
    c._portfolio = copy.copy(core._portfolio)
    h = copy.copy(core.historical_info)
    h.history_storage = core.historical_info.history_storage.copy()
    h.columns = list(core.historical_info.columns)
    c.historical_info = h
    c._obs_array = core._obs_array.copy()
    c.log_metrics = []
    c.verbose = 0
    if price_line is not None:
        p = core._price_array.copy()
        i = core._idx
        p[i + 1] = p[i] + price_line * (p[i + 1] - p[i])
        c._price_array = p
    return c


class ExtEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, split: str, kind: str, reward: str = "REP", seed: int = 0, positions=POSITIONS):
        super().__init__()
        assert kind in ("placebo", "edge", "real") and reward in ("REP", "LEDGER", "SPR", "TIR", "TSR", "WIDTH")
        self.moves = split_moves(split)
        self.signed = self.moves.copy()
        self.abs_m = np.abs(self.moves)
        self.kind, self.reward_mode = kind, reward
        self.positions = list(positions)
        self.rng = np.random.default_rng(seed)
        n = len(self.moves)
        df = pd.DataFrame({"close": np.full(n + 1, 100.0), "feature_z": np.zeros(n + 1)}, index=pd.date_range("2000-01-01", periods=n + 1, freq="D"))
        self.core = TradingEnv(df=df, positions=self.positions, dynamic_feature_functions=[_f_wealth, _f_pos, _f_rem], reward_function=linear_reward, windows=None,
                               trading_fees=FEE, borrow_interest_rate=0.0, portfolio_initial_value=W0, initial_position=0, max_episode_duration=EPISODE, verbose=0, render_mode=None)
        self.action_space = gym.spaces.Discrete(len(self.positions))
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(4,), dtype=np.float32)

    def _draw_market(self):
        n = len(self.abs_m)
        if self.kind == "real":
            eps = np.sign(self.moves)
            z = np.sign(np.r_[0.0, self.moves[:-1]])          # yesterday's sign: public, causal
        else:
            eps = self.rng.choice([-1.0, 1.0], size=n)
            if self.kind == "edge":
                z = np.where(self.rng.random(n) < P_EDGE, eps, -eps)
            else:
                z = self.rng.choice([-1.0, 1.0], size=n)
        ret = eps * self.abs_m
        price = 100.0 * np.cumprod(np.r_[1.0, 1.0 + ret])
        self.core._price_array = price
        self.core._obs_array[:, 0] = np.r_[z, 0.0].astype(np.float32)

    def _obs(self, o):
        return np.nan_to_num(np.clip(np.asarray(o, dtype=np.float32), -10.0, 10.0))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._draw_market()
        opts = options or {}
        if "start" in opts:
            # a fixed test window: the engine draws its start from NumPy's global generator, so set it for this draw and restore it
            state = np.random.get_state()
            np.random.seed(int(opts["start"]))
            o, info = self.core.reset()
            np.random.set_state(state)
        else:
            o, info = self.core.reset()
        return self._obs(o), {}

    @property
    def wealth(self) -> float:
        return float(self.core.historical_info["portfolio_valuation", -1])

    def query(self, action: int, t: float) -> float:
        c = clone_core(self.core, price_line=t)
        return float(c.step(int(action))[1])

    def step(self, action):
        action = int(action)
        mode = self.reward_mode
        extras = {}
        if mode in ("SPR", "TIR", "WIDTH"):
            extras["r_twin"] = self.query(action, -1.0)
            extras["c"] = max(-self.query(action, 0.0), 0.0)
        elif mode == "TSR":
            pp, p2, mm, m2 = (self.query(action, t) for t in (T_FAR, 2.0 * T_FAR, -T_FAR, -2.0 * T_FAR))
            ap, am = (p2 - pp) / T_FAR, (mm - m2) / T_FAR
            extras["g"] = (ap + am) if ap * am < 0.0 else (ap if abs(ap) >= abs(am) else am)
            extras["c"] = max(-self.query(action, 0.0), 0.0)
        w_old = self.wealth
        o, r, done, trunc, info = self.core.step(action)
        r = float(r)
        x = (self.wealth - w_old) / W0 * 100.0
        if mode == "REP":
            reward = r
        elif mode == "LEDGER":
            reward = x
        elif mode == "SPR":
            reward = 0.5 * (r - extras["r_twin"]) - extras["c"]
        elif mode == "TIR":
            rt, c = extras["r_twin"], extras["c"]
            reward = np.sign(r - rt) * (max(r, rt) + c) - c if abs(r - rt) > 1e-9 else -c
        elif mode == "TSR":
            reward = extras["g"] - extras["c"]
        elif mode == "WIDTH":
            reward = r + extras["r_twin"] + 2.0 * extras["c"]
        return self._obs(o), float(reward), bool(done), bool(trunc), dict(booked=x, reported=r, bust=bool(done))
