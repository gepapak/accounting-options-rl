"""How often does backtesting.py's zero floor bind in the setting of its own machine-learning tutorial? (exploratory)
The tutorial backtests EURUSD with Backtest(data, Strategy, commission=.0002, margin=.05), i.e. 20:1, long and short. Here: the
library's bundled EURUSD sample (backtesting.test.EURUSD), zero-skill strategies that hold a long or short position of the full
margin-allowed size and flip direction at random (mean holding 1 to 50 bars), at the tutorial's settings. Per run: reported final
equity (the engine) against the engine's own trade records; whether equity reached zero."""
import sys
import warnings

warnings.filterwarnings("ignore")
import os

os.environ["TQDM_DISABLE"] = "1"
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.test import EURUSD

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


class Coin(Strategy):
    p_flip = 0.1
    seed = 0

    def init(self):
        self.rng = np.random.default_rng(self.seed)
        self.side = 1 if self.rng.random() < 0.5 else -1
        self.started = False

    def next(self):
        if not self.started or self.rng.random() < self.p_flip:
            if self.started:
                self.side = -self.side
            (self.buy if self.side > 0 else self.sell)(size=0.9999)
            self.started = True


def main():
    data = EURUSD.copy()
    rows = []
    rng = np.random.default_rng(20261116)
    for i in range(N):
        hold = float(np.exp(rng.uniform(0, np.log(50))))
        cls = type(f"Coin{i}", (Coin,), dict(p_flip=1.0 / hold, seed=int(rng.integers(1e9))))
        st = Backtest(data, cls, commission=.0002, margin=.05, exclusive_orders=True, finalize_trades=True).run()
        cash = 10_000.0
        tr = st["_trades"]
        booked = cash + (float(tr["PnL"].sum()) if len(tr) else 0.0)
        eq = st["_equity_curve"]["Equity"]
        rows.append(dict(hold=hold, reported=float(st["Equity Final [$]"]), booked=booked, zero=bool(eq.min() <= 0), bars=len(eq),
                         trades=len(tr)))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "tutorial_reach.csv", index=False)
    z = df[df.zero]
    print(f"bundled EURUSD sample: {len(data)} bars, {str(data.index[0])[:16]} .. {str(data.index[-1])[:16]}")
    print(f"{N} zero-skill long/short strategies at the tutorial's settings (margin .05 = 20:1, commission .0002):")
    print(f"  equity reached zero (the engine set it to 0 and stopped) in {df.zero.mean():.1%} of runs")
    if len(z):
        print(f"  on those runs: reported -100%; the engine's own trade records book a median {100 * (z.booked / 1e4 - 1).median():+.1f}% "
              f"(worst {100 * (z.booked / 1e4 - 1).min():+.1f}%); forgiven median {100 * (z.reported - z.booked).median() / 1e4:.1f}% of the cash")
    print(f"  mean reported {100 * (df.reported.mean() / 1e4 - 1):+.2f}% against mean booked {100 * (df.booked.mean() / 1e4 - 1):+.2f}%")


if __name__ == "__main__":
    main()
