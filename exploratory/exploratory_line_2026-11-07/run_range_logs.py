"""Passive logs versus one probe past the logged range. See PROTOCOL_RANGE.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_formula_gap as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_range.json"
ENTRY = json.loads((HERE / "results.json").read_text())["summary"]["example_public"]["entry"]


def main():
    env, volume = g.prepared(ENTRY, floor=True)
    env_c, _ = g.prepared(ENTRY, floor=False)
    contract = float(env.simulator.symbols_info[g.SYMBOL].trade_contract_size)
    price0 = float(env.simulator.orders[0].entry_price)
    truth = volume * contract

    rng = np.random.default_rng(20261108)
    rets = rng.uniform(-0.004, 0.004, size=5000)
    gaps, y, x = [], [], []
    for ret in rets:
        rp = g.query(env, price0 * (1.0 + float(ret)))["reward"]
        rc = g.query(env_c, price0 * (1.0 + float(ret)))["reward"]
        gaps.append(abs(rp - rc))
        y.append(rp)
        x.append(price0 * float(ret))
    x = np.asarray(x)
    y = np.asarray(y)
    coef, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
    intercept, slope = float(coef[0]), float(coef[1])

    def at(ret, env_):
        return g.query(env_, price0 * (1.0 + ret))["reward"]

    pub16 = at(-0.16, env)
    ctrl16 = at(-0.16, env_c)
    pred16 = intercept + slope * (price0 * -0.16)

    hits = 0
    rng2 = np.random.default_rng(20261109)
    for ret in rng2.standard_t(3, size=5000) / np.sqrt(3.0) * 0.0015:
        if abs(at(float(ret), env) - at(float(ret), env_c)) > 0.01:
            hits += 1

    summary = {
        "entry": ENTRY,
        "volume": volume,
        "contract": contract,
        "truth_slope": truth,
        "n": 5000,
        "max_abs_gap_inside": float(np.max(gaps)),
        "slope": slope,
        "intercept": intercept,
        "slope_rel_err": abs(slope - truth) / truth,
        "pred_at_minus_16pct": pred16,
        "public_at_minus_16pct": pub16,
        "control_at_minus_16pct": ctrl16,
        "abs_pred_minus_public": abs(pred16 - pub16),
        "abs_pred_minus_control": abs(pred16 - ctrl16),
        "abs_public_minus_control": abs(pub16 - ctrl16),
        "rule1": bool(np.max(gaps) <= 0.01),
        "rule2": bool(abs(slope - truth) / truth <= 1e-3),
        "rule3": bool(abs(pred16 - pub16) >= 5000.0 and abs(pred16 - ctrl16) <= 1.0),
        "rule4": bool(abs(pub16 - ctrl16) >= 5000.0),
        "exploratory_student_t_hits": int(hits),
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
