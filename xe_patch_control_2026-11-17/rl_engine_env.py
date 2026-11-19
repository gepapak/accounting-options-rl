"""Gymnasium environments that let PPO trade THROUGH an unmodified mainstream backtesting engine (exploratory, 16 Nov 2026).

The engine runs one backtest per episode in a worker thread. Its strategy's next() hands the bar's observation and the engine's own
reported equity to the environment and waits for the agent's action, so every fill, fee, margin check and ruin rule is the engine's.
Market: the pilot's fair placebo paths (88 daily bars, EURUSD magnitudes, fair signs), so no policy has an edge. Episodes start at
equity log-uniform in [0.05, 2] x 10,000 USD (the paper's design, so that low-equity states are visited).

Actions (discrete): target leverage in LEVELS (notional over current equity). Observation: equity / 10,000 USD, current leverage / 200,
bars left / 88. Reward: change in the engine's reported equity / 10,000 USD. The agent never sees the ledger.

    BtpyEnv        backtesting.py 0.6.6 (margin = 1/200; at equity <= 0 the engine sets equity to 0 and stops)
    BacktraderEnv  backtrader 1.9.78 (futures-style margin; losses are booked in full and the account may stay negative)
"""
from __future__ import annotations

import queue
import threading
import warnings

warnings.filterwarnings("ignore")
import gymnasium as gym
import numpy as np
import pandas as pd

LEVELS = (0.0, 1.0, 4.0, 16.0, 64.0, 200.0)
L_MAX = 200.0
FEE = 0.0002
REF = 10_000.0
NBARS = 88
P0 = 1.20036


def placebo(rng, mags, scale=1.0):
    m = np.minimum(scale * mags[rng.integers(0, len(mags), size=NBARS - 1)], 0.9)
    s = rng.choice([-1.0, 1.0], size=NBARS - 1)
    return P0 * np.cumprod(np.r_[1.0, 1.0 + s * m])


class _Chan:
    def __init__(self, levels=LEVELS, l_max=L_MAX):
        self.obs = queue.Queue(maxsize=1)
        self.act = queue.Queue(maxsize=1)
        self.levels = tuple(levels)
        self.l_max = float(l_max)


class _EngineEnv(gym.Env):
    """levels: target leverage per action (negative = short). l_max: the engine's leverage limit (backtesting.py margin = 1/l_max).
    mag_scale: the placebo magnitudes are multiplied by it (capped at 90%)."""
    metadata = {"render_modes": []}

    def __init__(self, mags, seed=0, train=True, fixed_cash=None, levels=LEVELS, l_max=L_MAX, mag_scale=1.0):
        super().__init__()
        self.mags, self.train, self.fixed_cash = np.asarray(mags), train, fixed_cash
        self.levels, self.l_max, self.mag_scale = tuple(levels), float(l_max), float(mag_scale)
        self.rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Discrete(len(self.levels))
        self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
        self.ch, self.thread, self.last_eq, self.info_end = None, None, None, {}

    # -- subclass hooks: start the engine in a thread on (path, cash) using self.ch; put ("bar", obs, equity) and ("end", stats) --
    def _start(self, close, cash):
        raise NotImplementedError

    def _abort(self):
        if self.thread is not None and self.thread.is_alive():
            try:
                while True:
                    self.ch.act.get_nowait()
            except queue.Empty:
                pass
            self.ch.act.put(None)
            self.thread.join(timeout=5)
        self.thread = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._abort()
        opts = options or {}
        close = np.asarray(opts["path"]) if "path" in opts else placebo(self.rng, self.mags, self.mag_scale)
        if "cash" in opts:
            cash = float(opts["cash"])
        elif self.fixed_cash is not None:
            cash = float(self.fixed_cash)
        else:
            cash = REF * float(np.exp(self.rng.uniform(np.log(0.05), np.log(2.0))))
        self.cash0 = cash
        self.ch = _Chan(self.levels, self.l_max)
        self._start(close, cash)
        kind, obs, eq = self.ch.obs.get()
        self.last_eq = eq
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        self.ch.act.put(int(action))
        msg = self.ch.obs.get()
        if msg[0] == "bar":
            _, obs, eq = msg
            r = (eq - self.last_eq) / REF
            self.last_eq = eq
            return np.asarray(obs, dtype=np.float32), float(r), False, False, {}
        _, stats = msg
        r = (stats["reported_equity"] - self.last_eq) / REF
        self.thread.join(timeout=5)
        self.thread = None
        info = dict(stats=dict(stats, cash0=self.cash0))
        return np.zeros(3, dtype=np.float32), float(r), True, False, info

    def close(self):
        self._abort()


# ------------------------------------------------------------------ backtesting.py
from backtesting import Backtest, Strategy  # noqa: E402


class _BtpyAgent(Strategy):
    ch = None          # replaced per environment by a subclass made once in BtpyEnv.__init__

    def init(self):
        self.lev = 0.0

    def next(self):
        i = len(self.data) - 1
        eq = float(self.equity)
        lmax = self.ch.l_max
        obs = (eq / REF, self.lev / lmax, (NBARS - 1 - i) / NBARS)
        self.ch.obs.put(("bar", obs, eq))
        a = self.ch.act.get()
        if a is None:
            raise SystemExit
        target = self.ch.levels[a]
        if target == self.lev:
            return
        if target == 0.0:
            self.position.close()
        else:
            frac = max(min(abs(target) / lmax, 0.9999), 1e-6)
            (self.buy if target > 0 else self.sell)(size=frac)       # exclusive_orders: closes the old trade first
        self.lev = target


from functools import partial  # noqa: E402

from backtesting.backtesting import _Broker, _OutOfMoneyError  # noqa: E402


class BookedBroker(_Broker):
    """ADDED 17 Nov 2026 for the same-engine control. backtesting.py 0.6.6's own broker with its balance floor removed: next() is
    the package's method, line for line, except that its two lines `self._cash = 0` and `self._equity[i:] = 0` are replaced by one
    that records the equity the closed trades left (negative). The account still closes at ruin: full liability with closure."""

    def next(self):
        # Reset cached value here due to price change on every bar
        self.__dict__.pop(self.__class__._position_unrealized_pl.func.__name__, None)

        i = self._i = len(self._data) - 1
        self._process_orders()

        # Log account equity for the equity curve
        equity = self.equity
        self._equity[i] = equity

        if equity <= 0:
            assert self.margin_available <= 0
            for trade in self.trades:
                self._close_trade(trade, self.last_price, i)
            self.__dict__.pop(self.__class__._position_unrealized_pl.func.__name__, None)
            self._equity[i:] = self._cash          # was: self._cash = 0; self._equity[i:] = 0
            raise _OutOfMoneyError


class BtpyEnv(_EngineEnv):
    booked = False        # True: use BookedBroker (the control). Everything else is the registered Study XE wrapper, unchanged.

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._cls = type("BtpyAgentBound", (_BtpyAgent,), {})       # one subclass per environment object, made once

    def _start(self, close, cash):
        self._cls.ch = self.ch
        df = pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "High": np.maximum(np.r_[close[0], close[:-1]], close),
                           "Low": np.minimum(np.r_[close[0], close[:-1]], close), "Close": close, "Volume": 1.0})
        ch, cls, no_floor = self.ch, self._cls, self.booked

        def work():
            try:
                bt = Backtest(df, cls, cash=cash, commission=FEE, margin=1.0 / ch.l_max, trade_on_close=True,
                              exclusive_orders=True, finalize_trades=True)
                if no_floor:
                    bt._broker = partial(BookedBroker, *bt._broker.args, **bt._broker.keywords)
                st = bt.run()
                tr = st["_trades"]
                booked = cash + (float(tr["PnL"].sum()) if len(tr) else 0.0)
                ch.obs.put(("end", dict(reported_equity=float(st["Equity Final [$]"]), booked_equity=booked,
                                        hit_zero=bool(float(np.min(st["_equity_curve"]["Equity"])) <= 0.0), trades=int(len(tr)))))
            except SystemExit:
                pass
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()


class BtpyBookedEnv(BtpyEnv):
    """backtesting.py with its balance floor removed (BookedBroker): the same-engine control."""
    booked = True


# ------------------------------------------------------------------ backtrader
import backtrader as btr  # noqa: E402


class _BtAgent(btr.Strategy):
    params = dict(ch=None, acc=None)

    def __init__(self):
        self.lev = 0.0

    def notify_trade(self, trade):
        if trade.isclosed:
            self.p.acc["pnl"] += trade.pnlcomm

    def next(self):
        i = len(self) - 1
        if i >= NBARS - 2:                                          # stop at bar 86, as backtesting.py finalizes
            return
        eq = float(self.broker.getvalue())
        obs = (eq / REF, self.lev / self.p.ch.l_max, (NBARS - 1 - i) / NBARS)
        self.p.ch.obs.put(("bar", obs, eq))
        a = self.p.ch.act.get()
        if a is None:
            raise SystemExit
        target = self.p.ch.levels[a]
        if target == self.lev:
            return
        if self.position:
            self.close()
        if target != 0.0 and eq > 0.0:
            units = np.floor(abs(target) * eq / (self.data.close[0] * (1 + FEE)))
            (self.buy if target > 0 else self.sell)(size=units)
        self.lev = target if eq > 0.0 else 0.0

    def stop(self):
        acc = self.p.acc
        acc["value"] = float(self.broker.getvalue())
        acc["size"] = float(self.position.size) if self.position else 0.0
        acc["price"] = float(self.position.price) if self.position else 0.0
        acc["mark"] = float(self.data.close[0])


class BacktraderEnv(_EngineEnv):
    def _start(self, close, cash):
        ch = self.ch
        df = pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": np.maximum(np.r_[close[0], close[:-1]], close),
                           "low": np.minimum(np.r_[close[0], close[:-1]], close), "close": close, "volume": 1.0},
                          index=pd.date_range("2020-01-01", periods=len(close), freq="D"))

        def work():
            try:
                acc = dict(pnl=0.0)
                c = btr.Cerebro(stdstats=False)
                c.adddata(btr.feeds.PandasData(dataname=df))
                c.broker.setcash(cash)
                c.broker.set_coc(True)
                c.broker.setcommission(commission=FEE, margin=P0 / (2.0 * ch.l_max), mult=1.0, commtype=btr.CommInfoBase.COMM_PERC,
                                       percabs=True, stocklike=False)
                c.addstrategy(_BtAgent, ch=ch, acc=acc)
                c.run()
                booked = cash + acc["pnl"]
                if acc["size"]:
                    booked += acc["size"] * (acc["mark"] - acc["price"]) - abs(acc["size"]) * acc["price"] * FEE
                ch.obs.put(("end", dict(reported_equity=acc["value"], booked_equity=booked, hit_zero=acc["value"] <= 0.0, trades=-1)))
            except SystemExit:
                pass
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
