"""Exploratory pilot (16 Nov 2026): do mainstream backtesting engines agree on a leveraged trade once it reaches a loss beyond equity?

The same trade in each engine: on fair placebo paths (88 daily bars, EURUSD magnitudes from gym-mtsim's bundle, fair signs, so no
policy has an edge), buy the largest position the margin allows at leverage L on the first bar and hold to the end. Fee 2 bp per
side of notional. Engines, unmodified: backtesting.py 0.6.6 (margin = 1/L) and backtrader 1.9.78 (futures-style margin P0/L,
percentage commission). Reference accountings computed here: LEDGER (every loss booked, no stop) and FLOOR+STOP (equity set to 0 and
trading stopped at the first close with equity <= 0). For each engine and L: mean reported return over the paths, mean booked return
from the engine's own trade records, share of paths that reached zero equity, and which L maximizes the mean reported return
(what any optimizer of that metric would pick)."""
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAGS = np.load(HERE / "eurusd_mags.npy")
P0, CASH, FEE, NBARS = 1.20036, 10_000.0, 0.0002, 88
LEVS = [1, 2, 5, 10, 20, 50, 100, 200]
NPATHS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
PATH_SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261116     # the pilot's seed; a registered run passes a fresh one
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "engines_pilot.csv"


def paths(n, seed=None):
    seed = PATH_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        m = MAGS[rng.integers(0, len(MAGS), size=NBARS - 1)]
        s = rng.choice([-1.0, 1.0], size=NBARS - 1)
        out.append(P0 * np.cumprod(np.r_[1.0, 1.0 + s * m]))
    return out


def frame(close):
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": op, "High": np.maximum(op, close), "Low": np.minimum(op, close), "Close": close, "Volume": 1.0}, index=idx)


# ---------- reference accountings (the same trade: units = L * cash / (P_entry * (1 + fee)) at the first close) ----------
ENTRY, EXIT = 1, NBARS - 2           # backtesting.py enters at the close of bar 1 and finalizes at the close of bar 86


def reference(close, L):
    pe = close[ENTRY]
    units = np.floor(L * CASH * 0.9999 / (pe * (1 + FEE)))
    fee_in = units * pe * FEE
    eq = CASH - fee_in + units * (close[ENTRY:EXIT + 1] - pe)       # equity at each close, before the exit fee
    ledger = eq[-1] - units * close[EXIT] * FEE - CASH               # everything booked, held to the end
    hit = np.nonzero(eq <= 0)[0]
    floor_stop = (-CASH if hit.size else ledger)                     # equity 0 from the first close at or below 0
    return ledger / CASH, floor_stop / CASH, bool(hit.size)


# ---------- backtesting.py ----------
# Strategy classes are defined once, at module level: defining them inside the per-run functions made every later run slower
# (backtrader keeps a registry of strategy classes), and a 2,000-path run stalled for hours on 16 November.
from backtesting import Backtest, Strategy  # noqa: E402
import backtrader as bt  # noqa: E402


class BtpyBuyHold(Strategy):
    def init(self):
        self.done = False

    def next(self):
        if not self.done:
            self.buy(size=0.9999)
            self.done = True


class BtBuyHold(bt.Strategy):
    params = dict(L=1.0, acc=None)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.p.acc["pnl"] += trade.pnlcomm

    def next(self):
        acc, L = self.p.acc, self.p.L
        acc["minval"] = min(acc["minval"], self.broker.getvalue())
        if len(self) == ENTRY + 1 and not acc["done"]:            # bar 1, filled at its close (cheat-on-close)
            # the same units as backtesting.py and the reference (margin per unit set low enough that the order is accepted;
            # backtrader has no maintenance-margin liquidation, so the margin only affects acceptance and collateral)
            self.buy(size=np.floor(L * CASH * 0.9999 / (self.data.close[0] * (1 + FEE))))
            acc["done"] = True
        elif len(self) == EXIT + 1:                              # bar 86, where backtesting.py finalizes: read the value
            acc["v_exit"] = self.broker.getvalue()
            if self.position:
                acc["size"], acc["price"] = float(self.position.size), float(self.position.price)
                acc["mark"] = float(self.data.close[0])


def run_btpy(close, L):
    df = frame(close).reset_index(drop=True)         # integer index: the engine then skips its annualized statistics, which overflow
    bt_ = Backtest(df, BtpyBuyHold, cash=CASH, commission=FEE, margin=1.0 / L, trade_on_close=True, finalize_trades=True)
    st = bt_.run()
    tr = st["_trades"]
    booked = float(tr["PnL"].sum()) if len(tr) else 0.0
    return float(st["Equity Final [$]"]) / CASH - 1.0, booked / CASH, float(np.min(st["_equity_curve"]["Equity"])) <= 0.0


# ---------- backtrader ----------
def run_bt(close, L):
    acc = dict(pnl=0.0, minval=np.inf, done=False, size=0.0, price=0.0)
    c = bt.Cerebro(stdstats=False)
    c.adddata(bt.feeds.PandasData(dataname=frame(close)))
    c.broker.setcash(CASH)
    c.broker.set_coc(True)
    c.broker.setcommission(commission=FEE, margin=P0 / (2.0 * L), mult=1.0, commtype=bt.CommInfoBase.COMM_PERC, percabs=True, stocklike=False)
    c.addstrategy(BtBuyHold, L=L, acc=acc)
    c.run()
    # reported: the broker's value at bar 86 less the exit fee backtesting.py charges when it finalizes there;
    # booked: the open position's profit from the engine's own fill price, with entry and exit fees
    value = acc["v_exit"] - (abs(acc["size"]) * acc["mark"] * FEE if acc["size"] else 0.0)
    booked = acc["pnl"]
    if acc["size"]:
        booked += acc["size"] * (acc["mark"] - acc["price"]) - abs(acc["size"]) * (acc["mark"] + acc["price"]) * FEE
    return value / CASH - 1.0, booked / CASH, acc["minval"] <= 0.0


def main():
    ps = paths(NPATHS)
    rows = []
    t0 = time.time()
    for L in LEVS:
        for i, close in enumerate(ps):
            led, fls, hit = reference(close, L)
            r1, b1, z1 = run_btpy(close, L)
            r2, b2, z2 = run_bt(close, L)
            rows.append(dict(L=L, path=i, ledger=led, floor_stop=fls, ref_hit=hit, btpy_rep=r1, btpy_book=b1, btpy_zero=z1, bt_rep=r2, bt_book=b2, bt_zero=z2))
        print(f"  L={L} done ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / OUT_CSV, index=False)
    g = df.groupby("L")
    print(f"\n{NPATHS} fair placebo paths per leverage; returns in % of the starting cash (means over paths)")
    print(" L    reach0 | LEDGER  FLOOR+STOP | backtesting.py: reported  booked  zero | backtrader: reported  booked  zero")
    for L, x in g:
        print(f"{L:>4}  {x.ref_hit.mean():5.1%} | {100 * x.ledger.mean():+7.2f}  {100 * x.floor_stop.mean():+8.2f}  | "
              f"{100 * x.btpy_rep.mean():+8.2f}  {100 * x.btpy_book.mean():+8.2f}  {x.btpy_zero.mean():5.1%} | "
              f"{100 * x.bt_rep.mean():+8.2f}  {100 * x.bt_book.mean():+8.2f}  {x.bt_zero.mean():5.1%}")
    best1 = g.btpy_rep.mean().idxmax()
    best2 = g.bt_rep.mean().idxmax()
    bestl = g.ledger.mean().idxmax()
    print(f"\nleverage that maximizes the mean reported return: backtesting.py L={best1}, backtrader L={best2}; ledger L={bestl}")
    agree = df[df.L <= 2]
    print(f"agreement for L <= 2 (no path reaches zero): max |backtesting.py - backtrader| reported = "
          f"{100 * np.max(np.abs(agree.btpy_rep - agree.bt_rep)):.4f} points; max |backtesting.py - ledger| = {100 * np.max(np.abs(agree.btpy_rep - agree.ledger)):.4f}")
    big = df[df.L >= 50]
    print(f"paths with L >= 50 where the two engines' reported returns differ in sign: {np.mean(np.sign(big.btpy_rep) != np.sign(big.bt_rep)):.1%}")


if __name__ == "__main__":
    main()
