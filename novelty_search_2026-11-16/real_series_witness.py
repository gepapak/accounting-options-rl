"""Real-data witness (exploratory): the pilot's trade on the 14 real daily series bundled with gym-mtsim 2.0.0 (7 forex pairs, 3
cryptocurrencies, 4 stocks, May to September 2021), long and short, through backtesting.py and backtrader, with the two reference
accountings. Each series is one path, so this shows WHERE the engines part on real prices, not an expectation."""
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
from backtesting import Backtest, Strategy  # noqa: E402
import backtrader as bt  # noqa: E402

CASH, FEE = 10_000.0, 0.0002
LEVS = [1, 2, 5, 10, 20, 50, 100, 200]


class BH(Strategy):
    side = 1

    def init(self):
        self.done = False

    def next(self):
        if not self.done:
            (self.buy if self.side > 0 else self.sell)(size=0.9999)
            self.done = True


class Long(BH):
    side = 1


class Short(BH):
    side = -1


class BtS(bt.Strategy):
    params = dict(units=0.0, acc=None, entry=1, exit=0)

    def next(self):
        if len(self) == self.p.entry + 1:
            (self.buy if self.p.units > 0 else self.sell)(size=abs(self.p.units))
        elif len(self) == self.p.exit + 1:
            self.p.acc["v"] = self.broker.getvalue()
            self.p.acc["mark"] = float(self.data.close[0])


def frame(close, dt=False):
    op = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"Open": op, "High": np.maximum(op, close), "Low": np.minimum(op, close), "Close": close, "Volume": 1.0})
    if dt:
        df = df.rename(columns=str.lower)
        df.index = pd.date_range("2021-05-05", periods=len(close), freq="D")
    return df


def one(close, L, side):
    n = len(close)
    entry, exit_ = 1, n - 2
    pe = close[entry]
    units = np.floor(L * CASH * 0.9999 / (pe * (1 + FEE)))
    path = side * units * (close[entry:exit_ + 1] - pe)
    eq = CASH - units * pe * FEE + path
    ledger = eq[-1] - units * close[exit_] * FEE - CASH
    fstop = -CASH if (eq <= 0).any() else ledger
    st = Backtest(frame(close), Long if side > 0 else Short, cash=CASH, commission=FEE, margin=1.0 / L, trade_on_close=True,
                  finalize_trades=True).run()
    btpy_rep = float(st["Equity Final [$]"]) - CASH
    acc = {}
    c = bt.Cerebro(stdstats=False)
    c.adddata(bt.feeds.PandasData(dataname=frame(close, dt=True)))
    c.broker.setcash(CASH)
    c.broker.set_coc(True)
    c.broker.setcommission(commission=FEE, margin=close[0] / (2.0 * L), mult=1.0, commtype=bt.CommInfoBase.COMM_PERC, percabs=True,
                           stocklike=False)
    c.addstrategy(BtS, units=side * units, acc=acc, entry=entry, exit=exit_)
    c.run()
    bt_rep = acc["v"] - units * acc["mark"] * FEE - CASH
    return dict(ledger=100 * ledger / CASH, floor_stop=100 * fstop / CASH, btpy=100 * btpy_rep / CASH, bt=100 * bt_rep / CASH,
                ruin=bool((eq <= 0).any()))


def main():
    df = pd.read_csv(HERE / "real_closes_gym_mtsim_2021.csv")
    rows = []
    for sym in df.columns:
        close = df[sym].dropna().to_numpy()
        for side in (1, -1):
            for L in LEVS:
                r = one(close, L, side)
                rows.append(dict(symbol=sym, side="long" if side > 0 else "short", L=L, **r))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "real_series_witness.csv", index=False)
    out["gap"] = out.btpy - out.bt
    print("returns in % of the starting cash; one real path per series and side")
    for L, g in out.groupby("L"):
        agree = (g.btpy - g.bt).abs().max()
        print(f"L={L:>3}: trades reaching zero {int(g.ruin.sum()):>2}/28 | max |backtesting.py - backtrader| {agree:8.2f} pts | "
              f"backtesting.py mean {g.btpy.mean():+8.2f}, backtrader mean {g.bt.mean():+8.2f}, ledger mean {g.ledger.mean():+8.2f} | "
              f"opposite signs {int((np.sign(g.btpy) != np.sign(g.bt)).sum()):>2}/28 | backtrader = ledger to {(g.bt - g.ledger).abs().max():.1e}")
    worst = out.loc[out.gap.idxmax()]
    print(f"largest gap: {worst.symbol} {worst.side} L={worst.L}: backtesting.py {worst.btpy:+.1f}%, backtrader {worst.bt:+.1f}%")


if __name__ == "__main__":
    main()
