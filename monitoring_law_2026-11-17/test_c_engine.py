"""Test C: the law in unmodified backtesting.py 0.6.6 at finer time steps (predictions_v2.json, section C, written before this ran).

Fine Gaussian paths: 87 days x 16 steps a day, step standard deviation s_eurusd / 4, prices compounded. The same paths are given to the
engine as bars sampled m = 1, 4 or 16 times a day. Strategy: buy the largest position the margin allows (margin = 1/L) at the end of
day 1 and hold; trade_on_close, finalize_trades. The engine checks ruin at every bar close it is given. Forgiven = reported final
equity minus the engine's own trade records, in units of the starting cash.

    <venv-vbt python> -B test_c_engine.py [paths] [workers]
"""
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from math import sqrt
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TQDM_DISABLE"] = "1"
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "predictions_v2.json").read_text())
S_E = json.loads((HERE / "predictions.json").read_text())["inputs"]["s_eurusd"]
M, DAYS, CASH, P0 = 16, 87, 10_000.0, 1.2
LEVS, MS, FEES = (20, 50, 200), (1, 4, 16), (0.0, 0.0002)
CHUNK = 250


class Hold(Strategy):
    entry_i = 1

    def init(self):
        self.done = False

    def next(self):
        if not self.done and len(self.data) - 1 == self.entry_i:
            self.buy(size=0.9999)
            self.done = True


HOLD = {m: type(f"Hold{m}", (Hold,), dict(entry_i=m)) for m in MS}


def frame(close):
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"Open": op, "High": np.maximum(op, close), "Low": np.minimum(op, close), "Close": close, "Volume": 1.0})


def chunk(cid):
    rng = np.random.default_rng([20261117, 3, cid])
    out = []
    for _ in range(CHUNK):
        fine = P0 * np.cumprod(np.r_[1.0, 1.0 + rng.normal(0.0, S_E / sqrt(M), size=DAYS * M)])
        for m in MS:
            df = frame(fine[::M // m])
            for fee in FEES:
                for L in LEVS:
                    st = Backtest(df, HOLD[m], cash=CASH, commission=fee, margin=1.0 / L, trade_on_close=True, finalize_trades=True).run()
                    tr = st["_trades"]
                    rep = float(st["Equity Final [$]"]) / CASH - 1.0
                    book = (float(tr["PnL"].sum()) if len(tr) else 0.0) / CASH
                    out.append((fee, L, m, rep - book, float(st["_equity_curve"]["Equity"].min()) <= 0.0))
    return out


def main():
    n_paths = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = [r for part in ex.map(chunk, range(n_paths // CHUNK)) for r in part]
    df = pd.DataFrame(res, columns=["fee", "L", "m", "forgiven", "ruin"])
    df.to_csv(HERE / "test_c_engine.csv", index=False)
    rng = np.random.default_rng(5)
    ok, rows = True, []
    print(f"backtesting.py 0.6.6, {n_paths} fine Gaussian paths, {time.time() - t0:.0f} s; forgiven in % of the starting cash")
    print(f"{'fee':>5} {'L':>4} {'m':>3} | {'ruin':>6} {'pred':>6} | {'forgiven':>9} {'95% interval':>18} {'pred':>8} {'ratio':>6}")
    for pr in P["C"]:
        g = df[(df.fee == pr["fee"]) & (df.L == pr["L"]) & (df.m == pr["m"])]
        f = g.forgiven.to_numpy()
        bs = f[rng.integers(0, len(f), size=(2000, len(f)))].mean(axis=1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        ratio = f.mean() / pr["forgiven"]
        cell = (abs(ratio - 1) <= 0.10 or lo <= pr["forgiven"] <= hi) and abs(g.ruin.mean() - pr["p_ruin"]) <= 0.03
        ok &= cell
        rows.append(dict(fee=pr["fee"], L=pr["L"], m=pr["m"], ruin=float(g.ruin.mean()), forgiven=float(f.mean()), lo=float(lo), hi=float(hi),
                         pred=pr["forgiven"], ratio=float(ratio), ok=bool(cell)))
        print(f"{1e4 * pr['fee']:>3.0f}bp {pr['L']:>4} {pr['m']:>3} | {g.ruin.mean():6.4f} {pr['p_ruin']:6.4f} | {100 * f.mean():9.3f} "
              f"[{100 * lo:7.3f}, {100 * hi:7.3f}] {100 * pr['forgiven']:8.3f} {ratio:6.3f}{'' if cell else '   <-- criterion not met'}")
    slopes = {}
    for L in LEVS:
        v = [df[(df.fee == 0.0) & (df.L == L) & (df.m == m)].forgiven.mean() for m in MS]
        slopes[L] = float(np.polyfit(np.log(MS), np.log(v), 1)[0])
    print("fee-free log-log slope of the forgiven amount against m:", ", ".join(f"L={L}: {s:+.3f}" for L, s in slopes.items()))
    ok &= all(-0.56 <= s <= -0.42 for s in slopes.values())
    print("C:", "ALL CRITERIA MET" if ok else "NOT ALL CRITERIA MET")
    (HERE / "test_c_engine.json").write_text(json.dumps(dict(rows=rows, slopes=slopes, ok=bool(ok), paths=n_paths), indent=1))


if __name__ == "__main__":
    main()
