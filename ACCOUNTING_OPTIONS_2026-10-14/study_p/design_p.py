"""Study P design calculation (run before registration, after Study O is complete).

Fits Study O's run-level curve P(L(0.02) = 16) = sigmoid(s * [log10(index) - log10 c50]) on its treatment runs, then
predicts, under the sample-complexity law (c50 proportional to steps^(-1/2)), the probability that a run learns the
exploit for each candidate arm at each Study P budget. Writes design_p.json.

    python design_p.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import optimize

from aggregate_p import CONTROL, load

HERE = Path(__file__).resolve().parent
BUDGETS = (250_000, 4_000_000)


def main() -> int:
    R = load()
    O = R[(R.study == "O") & (R.alpha != CONTROL)]
    x, y = np.log10(O["index"].to_numpy()), (O.L002 == 16).to_numpy(float)
    nll = lambda p: -np.sum(y * p[0] * (x - p[1]) - np.logaddexp(0, p[0] * (x - p[1])))  # noqa: E731
    s, lc50 = optimize.minimize(nll, [5.0, np.log10(0.3)], method="Nelder-Mead").x
    idx = {a: float(np.mean([json.loads((HERE.parent / "study_o" / f"dp_o_{z}.json").read_text())["arms"][f"{a:g}"]["index_pct"]
                             for z in ("DK1", "DK2")])) for a in (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95)}
    out = dict(n_runs=len(O), slope=float(s), c50_1M_pct=float(10 ** lc50), budgets={})
    print(f"Study O fit on {len(O)} treatment runs: slope {s:.2f}, c50(1M) = {10 ** lc50:.3f}%")
    for n in BUDGETS:
        lc = lc50 - 0.5 * np.log10(n / 1e6)
        pred = {f"{a:g}": float(1 / (1 + np.exp(-s * (np.log10(i) - lc)))) for a, i in idx.items()}
        out["budgets"][str(n)] = dict(c50_pct=float(10 ** lc), p_learned=pred)
        print(f"budget {n:>9,}: predicted c50 = {10 ** lc:.3f}% | P(learned) by alpha: " +
              ", ".join(f"{a}: {p:.2f}" for a, p in pred.items()))
    (HERE / "design_p.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
