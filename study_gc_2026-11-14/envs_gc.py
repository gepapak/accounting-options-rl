"""Study GC environments (PREREGISTRATION_GC.md): Study G's design inside unmodified gym-mtsim 2.0.0, with the market taken from the
package's registered crypto environment instead of a synthetic forex market.

Nothing in the package is changed. As in Study G, the wrapper controls only the episode (which price path, which starting balance) and
the observation; all accounting, the action box and the reward are gym-mtsim's.

Markets
    crypto  the BTCUSD contract of the bundled crypto file, the fee of the registered `crypto-hedge-v0` (50 USD per coin)
    forex   the EURUSD contract of the bundled forex file, the fee of the registered `forex-hedge-v0` (0.0003)
Account (both, the package defaults): 10,000 USD, leverage 100, stop-out level 0.2, hedge mode; one order at a time, window 2 (Study G).

Prices: a placebo built from the bundled daily bars. A path has the bundle's length and starts at its first close; every bar draws one
of the bundle's simple-return magnitudes with replacement and a fair sign, so the price is a martingale and no position has a positive
expected profit. A fresh path is drawn for every episode.

Conditions
    ORIG     gym-mtsim as published: a balance left negative by a stop-out is set to zero
    FLPATCH  a subclass whose tick() is the published code without the three lines of that floor (Study G's control)
    LINE     the published simulator, but the learner is rewarded by the reading of the published reward on clones of the simulator:
             the reward at a zero move plus its slope in the price (bumps of 0.1%) times the realized move. It reads neither the ledger
             nor the order records and deletes nothing.

The booked profit of a step is taken from the simulator's own order records: starting balance plus the profit of closed orders plus the
profit of open orders (`ledger`). test_envs_gc.py checks that it equals the FLPATCH equity under the same actions and prices.
"""
from __future__ import annotations

import copy
from datetime import timedelta

import gymnasium as gym
import numpy as np
import pandas as pd

import gym_mtsim  # noqa: F401
from gym_mtsim import CRYPTO_DATA_PATH, FOREX_DATA_PATH, MtEnv, MtSimulator

MARKETS = {
    "crypto": dict(path=CRYPTO_DATA_PATH, symbol="BTCUSD", fee=50.0, vol_scale=10.0),
    "forex": dict(path=FOREX_DATA_PATH, symbol="EURUSD", fee=0.0003, vol_scale=1.0),
}
BALANCE, LEVERAGE, STOP_OUT = 10_000.0, 100.0, 0.2
EGRID = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
TEST_SEED, N_TEST = 20261114, 300
_BUNDLE: dict = {}


class MtSimulatorFL(MtSimulator):
    """gym-mtsim's simulator without the zero floor on the balance: tick() is the published method minus its last three lines."""

    def tick(self, delta_time: timedelta=timedelta()) -> None:
        self._check_current_time()

        self.current_time += delta_time
        self.equity = self.balance

        for order in self.orders:
            order.exit_time = self.current_time
            order.exit_price = self.price_at(order.symbol, order.exit_time)['Close']
            self._update_order_profit(order)
            self.equity += order.profit

        while self.margin_level < self.stop_out_level and len(self.orders) > 0:
            most_unprofitable_order = min(self.orders, key=lambda order: order.profit)
            self.close_order(most_unprofitable_order)


def bundle(market: str) -> dict:
    """The bundled contract, bars and return magnitudes of a market (loaded once)."""
    if market not in _BUNDLE:
        m = MARKETS[market]
        base = MtSimulator(symbols_filename=m["path"], hedge=True)
        frame = base.symbols_data[m["symbol"]]
        close = frame["Close"].to_numpy(dtype=float)
        _BUNDLE[market] = dict(info=base.symbols_info[m["symbol"]], frame=frame, index=frame.index, columns=list(frame.columns),
                               p0=float(close[0]), mags=np.abs(close[1:] / close[:-1] - 1.0))
    return _BUNDLE[market]


def placebo_closes(rng: np.random.Generator, market: str) -> np.ndarray:
    """One placebo path of closes: the bundle's length, its first close, its magnitudes with replacement, fair signs."""
    b = bundle(market)
    n = len(b["index"])
    mags = b["mags"][rng.integers(0, len(b["mags"]), size=n - 1)]
    signs = rng.choice([-1.0, 1.0], size=n - 1)
    return b["p0"] * np.cumprod(np.r_[1.0, 1.0 + signs * mags])


def test_bank(market: str) -> np.ndarray:
    """The N_TEST test paths shared by every agent of a market."""
    rng = np.random.default_rng(TEST_SEED)
    return np.stack([placebo_closes(rng, market) for _ in range(N_TEST)])


def path_frame(market: str, close: np.ndarray) -> pd.DataFrame:
    b = bundle(market)
    open_ = np.r_[close[0], close[:-1]]
    new = pd.DataFrame({c: 1.0 for c in b["columns"]}, index=b["index"])
    new["Open"], new["Close"] = open_, close
    if "Low" in new:
        new["Low"] = np.minimum(open_, close)
    if "High" in new:
        new["High"] = np.maximum(open_, close)
    return new


def make_simulator(market: str, condition: str) -> MtSimulator:
    m, b = MARKETS[market], bundle(market)
    cls = MtSimulatorFL if condition == "FLPATCH" else MtSimulator
    sim = cls(unit="USD", balance=BALANCE, leverage=LEVERAGE, stop_out_level=STOP_OUT, hedge=True)
    sim.symbols_info = {m["symbol"]: copy.deepcopy(b["info"])}
    sim.symbols_data = {m["symbol"]: b["frame"].copy()}
    return sim


def vmax_closed(market: str, balance: float, price: float) -> float:
    """Largest volume the margin check accepts from a flat account: balance / (margin per unit + fee per unit), rounded down."""
    m, info = MARKETS[market], bundle(market)["info"]
    per_unit = info.trade_contract_size * (price / LEVERAGE + m["fee"])
    return float(np.floor(max(balance, 0.0) / per_unit / info.volume_step + 1e-9) * info.volume_step)


def light_clone(sim: MtSimulator) -> MtSimulator:
    """A copy of the simulator that owns its orders and shares the price frames and the symbol table (neither is written)."""
    c = copy.copy(sim)
    c.orders = copy.deepcopy(sim.orders)
    c.closed_orders = list(sim.closed_orders)
    c.symbols_data = dict(sim.symbols_data)
    return c


def ledger(sim: MtSimulator, e0: float) -> float:
    return e0 + sum(o.profit for o in sim.closed_orders) + sum(o.profit for o in sim.orders)


class GCEnv(gym.Wrapper):
    """Study G's wrapper around MtEnv, with a fresh placebo path per episode."""

    def __init__(self, market: str, condition: str, train: bool, seed: int, audit: bool = False):
        assert condition in ("ORIG", "FLPATCH", "LINE")
        m = MARKETS[market]
        env = MtEnv(original_simulator=make_simulator(market, condition), trading_symbols=[m["symbol"]], window_size=2,
                    symbol_max_orders=1, fee=m["fee"])
        super().__init__(env)
        self.market, self.condition, self.train, self.audit = market, condition, train, audit
        self.sym, self.vol_scale = m["symbol"], m["vol_scale"]
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10, 10, shape=(5,), dtype=np.float32)
        self.count = dict(steps=0, floor_steps=0, episodes=0, wiped_episodes=0, floored_episodes=0, line_steps=0, line_max_abs_err=0.0)

    # ---- episode --------------------------------------------------------------------------------------------------------------------
    def _flat(self, o) -> np.ndarray:
        return np.clip(np.array([o["equity"][0] / 1e4, o["balance"][0] / 1e4, o["margin"][0] / 1e4,
                                 o["orders"][0, 0, 1] / self.vol_scale, o["orders"][0, 0, 2] / 1e4], dtype=np.float32), -10, 10)

    def _install(self, close: np.ndarray) -> None:
        e = self.env.unwrapped
        e.original_simulator.symbols_data[self.sym] = path_frame(self.market, close)
        e.prices[self.sym] = np.column_stack([close, np.r_[close[0], close[:-1]]])      # columns Close, Open, as MtEnv builds them
        e.signal_features = e._process_data()

    def reset(self, *, seed=None, options=None):
        e = self.env.unwrapped
        opts = dict(options or {})
        self._install(np.asarray(opts["path"], dtype=float) if "path" in opts else placebo_closes(self.rng, self.market))
        if "balance" in opts:
            b = float(opts["balance"])
        elif self.train:
            b = BALANCE * float(np.exp(self.rng.uniform(np.log(0.05), np.log(2.0))))
        else:
            b = BALANCE
        e.original_simulator.balance = e.original_simulator.equity = b
        o, info = self.env.reset()
        self.e0 = self.prev_ledger = b
        self.stats = dict(e0=b, reported=0.0, booked=0.0, floored=0, wiped=0, orders=0, lev_max=0.0, line_max_abs_err=0.0)
        return self._flat(o), info

    # ---- the reading as a reward ----------------------------------------------------------------------------------------------------
    def line_reward(self, a: np.ndarray) -> float:
        e = self.env.unwrapped
        tick = e._current_tick
        t0, t1 = e.time_points[tick], e.time_points[tick + 1]
        shadow = copy.copy(e)                                  # shares everything except the simulator
        shadow.simulator = light_clone(e.simulator)
        eq0 = float(shadow.simulator.equity)
        shadow._apply_action(a)                                # the package's own order handling, on the clone
        post = shadow.simulator
        p0, p1 = float(e.prices[self.sym][tick, 0]), float(e.prices[self.sym][tick + 1, 0])
        frame = post.symbols_data[self.sym]
        col = frame.columns.get_loc("Close")

        def phi(price: float) -> float:
            c = light_clone(post)
            df = frame.copy()
            df.iloc[tick + 1, col] = price
            c.symbols_data[self.sym] = df
            c.tick(t1 - t0)
            return float(c.equity) - eq0

        base = phi(p0)
        if not post.orders:                                    # no open order: the reward does not depend on the price
            return base
        d = 1e-3 * p0
        return base + (phi(p0 + d) - phi(p0 - d)) / (2 * d) * (p1 - p0)

    # ---- one bar --------------------------------------------------------------------------------------------------------------------
    def step(self, action):
        e = self.env.unwrapped
        a = np.asarray(action, dtype=np.float64)
        line = self.line_reward(a) if (self.condition == "LINE" or self.audit) else None
        o, r, term, trunc, info = self.env.step(a)
        sim = e.simulator
        led = ledger(sim, self.e0)
        booked = led - self.prev_ledger
        self.prev_ledger = led
        st, ct = self.stats, self.count
        st["reported"], st["booked"] = float(sim.equity) - self.e0, led - self.e0
        floor = float(r) - booked > 1e-6
        st["floored"] |= int(floor)
        st["wiped"] |= int(sim.balance <= 1e-9)
        st["orders"] = len(sim.closed_orders) + len(sim.orders)
        if sim.equity > 0:
            st["lev_max"] = max(st["lev_max"], float(sim.margin) * float(sim.leverage) / float(sim.equity))
        ct["steps"] += 1
        ct["floor_steps"] += int(floor)
        if line is not None:
            err = abs(line - booked)
            st["line_max_abs_err"] = max(st["line_max_abs_err"], err)
            ct["line_steps"] += 1
            ct["line_max_abs_err"] = max(ct["line_max_abs_err"], err)
        done = term or trunc
        if done:
            ct["episodes"] += 1
            ct["wiped_episodes"] += st["wiped"]
            ct["floored_episodes"] += st["floored"]
        reward = float(line) if self.condition == "LINE" else float(r)
        return self._flat(o), reward / 100.0, term, trunc, ({"stats": dict(st)} if done else {})


def exposure(predict, market: str, e: float) -> float:
    """Study G's exposure: the share of the margin-feasible maximum position that the deterministic policy opens from a flat account
    with equity e x 10,000 USD; 0 if it holds or the order would be refused. `predict` maps a raw observation to the action."""
    a = np.asarray(predict(np.array([[e, e, 0.0, 0.0, 0.0]], dtype=np.float32)), dtype=float).ravel()
    if 1.0 / (1.0 + np.exp(-a[1])) > 0.5:
        return 0.0
    info, p0 = bundle(market)["info"], bundle(market)["p0"]
    v = float(np.clip(abs(a[2]), info.volume_min, info.volume_max))
    v = round(v / info.volume_step) * info.volume_step                 # the environment's own rounding
    feasible = vmax_closed(market, e * BALANCE, p0)
    if feasible < info.volume_min:
        return 0.0
    return float(v / feasible) if v <= feasible + 1e-9 else 0.0
