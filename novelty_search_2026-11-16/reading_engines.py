"""The paper's one-query reading applied to the engines (exploratory): open the largest position at leverage 200 (a 1x short in vectorbt,
which has no leverage) and move the next close once INSIDE the range of ordinary moves (-0.2%) and once PAST the kink (-5%, i.e. a loss
of about ten times the equity at leverage 200; +300% for vectorbt's 1x short). Inside, every engine must report the booked profit;
past the kink, the reported equity reveals the rule. No fees, so the booked profit is exactly position x move."""
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
CASH, P0 = 10_000.0, 1.0


def series(move):
    close = np.array([P0, P0, P0 * (1 + move), P0 * (1 + move)])
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": op, "High": np.maximum(op, close), "Low": np.minimum(op, close), "Close": close, "Volume": 1.0})


def btpy(move, L=200):
    from backtesting import Backtest, Strategy

    class S(Strategy):
        def init(self):
            self.done = False

        def next(self):
            if not self.done:
                self.buy(size=0.9999)
                self.done = True
    st = Backtest(series(move), S, cash=CASH, commission=0.0, margin=1.0 / L, trade_on_close=True, finalize_trades=True).run()
    tr = st["_trades"]
    return float(st["Equity Final [$]"]), CASH + float(tr["PnL"].sum())


def btpy_short_unlevered(move):
    """backtesting.py with margin=1 (no leverage): a 1x short of the full equity, then the next close moves by `move`."""
    from backtesting import Backtest, Strategy

    class S(Strategy):
        def init(self):
            self.done = False

        def next(self):
            if not self.done:
                self.sell(size=0.9999)
                self.done = True
    st = Backtest(series(move), S, cash=CASH, commission=0.0, margin=1.0, trade_on_close=True, finalize_trades=True).run()
    tr = st["_trades"]
    return float(st["Equity Final [$]"]), CASH + float(tr["PnL"].sum())


def backtrader_(move, L=200):
    import backtrader as bt
    acc = {}

    class S(bt.Strategy):
        def next(self):
            if len(self) == 2 and not self.position:
                self.buy(size=np.floor(L * CASH * 0.9999 / self.data.close[0]))

        def stop(self):
            acc["v"] = self.broker.getvalue()
            acc["size"], acc["price"] = self.position.size, self.position.price
    df = series(move).rename(columns=str.lower)
    df.index = pd.date_range("2020-01-01", periods=len(df), freq="D")
    c = bt.Cerebro(stdstats=False)
    c.adddata(bt.feeds.PandasData(dataname=df))
    c.broker.setcash(CASH)
    c.broker.set_coc(True)
    c.broker.setcommission(commission=0.0, margin=P0 / (2.0 * L), mult=1.0, stocklike=False)
    c.addstrategy(S)
    c.run()
    return acc["v"], CASH + acc["size"] * (df.close.iloc[-1] - acc["price"])


def vectorbt_(move):
    import vectorbt as vbt
    c = series(move)["Close"]
    units = np.floor(CASH / c.iloc[1])
    pf = vbt.Portfolio.from_orders(c, size=np.array([np.nan, units, np.nan, np.nan]), size_type="amount", direction="shortonly",
                                   init_cash=CASH, fees=0.0)
    return float(pf.value().iloc[-1]), CASH - units * (c.iloc[-1] - c.iloc[1])


print(f"{'engine':<16} {'query':<22} {'reported':>12} {'booked':>12}  verdict")
for name, fn, inside, past in [("backtesting.py", btpy, -0.002, -0.05), ("backtrader", backtrader_, -0.002, -0.05),
                               ("vectorbt (1x short)", vectorbt_, 0.002, 3.0),
                               ("backtesting.py, no leverage (1x short)", btpy_short_unlevered, 0.002, 3.0)]:
    for label, mv in (("inside the range", inside), ("past the kink", past)):
        try:
            rep, book = fn(mv)
            verdict = "agrees" if abs(rep - book) < 1e-6 * CASH else ("FORGIVES " + f"{rep - book:,.0f}" if rep > book else "differs")
            print(f"{name:<16} {label + f' ({mv:+.1%})':<22} {rep:12,.0f} {book:12,.0f}  {verdict}")
        except Exception as ex:
            print(f"{name:<16} {label:<22} failed: {ex}")
