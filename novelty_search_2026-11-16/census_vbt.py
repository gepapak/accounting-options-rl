"""Census, vectorbt (exploratory): the pilot's trade on the pilot's paths, as a short (free vectorbt has no leverage setting; with
fair signs a short and a long have the same distribution). Short the largest position at leverage L on bar 1, cover on bar 86, fee 2 bp
per side. Reported: vectorbt's portfolio value at bar 86. LEDGER: the same short with every loss booked. Writes census_vbt.csv."""
import sys
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engines_pilot as E  # noqa: E402  (paths, constants; importing it does not run the pilot)

import vectorbt as vbt  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def main():
    rows = []
    for i, close in enumerate(E.paths(N)):
        c = pd.Series(close[: E.EXIT + 1])
        pe = close[E.ENTRY]
        for L in E.LEVS:
            units = np.floor(L * E.CASH * 0.9999 / (pe * (1 + E.FEE)))
            size = pd.Series(np.nan, index=c.index)
            size.iloc[E.ENTRY] = units
            size.iloc[E.EXIT] = -units                              # buy back the short at bar 86
            pf = vbt.Portfolio.from_orders(c, size=size.values, size_type="amount", direction="both", init_cash=E.CASH,
                                           fees=E.FEE, allow_partial=False, raise_reject=False)
            rep = float(pf.value().iloc[-1]) / E.CASH - 1.0
            filled = int(pf.orders.count())
            ledger = (-(units * (close[E.EXIT] - pe)) - units * (pe + close[E.EXIT]) * E.FEE) / E.CASH
            eq = E.CASH - units * pe * E.FEE - units * (close[E.ENTRY:E.EXIT + 1] - pe)
            rows.append(dict(L=L, path=i, vbt_rep=rep, ledger=ledger, filled=filled, hit=bool((eq <= 0).any())))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "census_vbt.csv", index=False)
    for L, g in df.groupby("L"):
        print(f"L={L:>3}: reach0 {g.hit.mean():6.1%}  vectorbt reported {100 * g.vbt_rep.mean():+8.2f}  ledger {100 * g.ledger.mean():+8.2f}  "
              f"max |diff| {100 * (g.vbt_rep - g.ledger).abs().max():.2e} points  orders filled per run {g.filled.mean():.2f}")


if __name__ == "__main__":
    main()
