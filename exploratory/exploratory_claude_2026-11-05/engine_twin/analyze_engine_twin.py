r"""Per-run results of the engine's sign twin, from the step traces written by twin_trace.py.

Definitions (all in DKK per step; run totals as percentage points of the initial sleeve E0):
  C        own cash flow, same on both branches: access fee + roll cost + transaction cost + funding - battery flow
  x        booked P&L: gross - C
  r, r'    reported change of the real and of the twin branch
  credit   r - x and r' - (-gross - C); both must be non-negative for the bracket's premise to hold
  bracket  [-r' - 2C, r], width r + r' + 2C
  TIR      sign(gross)(max(r, r') + C) - C where r != r', else -C
  SPR      (r - r')/2 - C, the bracket's midpoint
  direct   position x move - C, from public quantities and the payoff formula of the protocol in force
  other    the same with the payoff formula of the other protocol generation

Flags, each at the audit's threshold of 5 points: reconciliation |reported - booked|, width, |reported - sum of TIR|.
Cost-shift test: twin_reward.single_floor_fit on the pairs at the shifted costs, per informative step.

Usage: python analyze_engine_twin.py [--traces traces] [--out results] [--print] [names...]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for root in (HERE.parent / "github" / "accounting-options-rl" / "ACCOUNTING_OPTIONS_2026-10-14", HERE.parent.parent / "ACCOUNTING_OPTIONS_2026-10-14"):
    if (root / "twin_reward" / "twin_reward.py").is_file():
        sys.path.insert(0, str(root / "twin_reward"))
from twin_reward import single_floor_fit  # noqa: E402

TOL_DKK = 1e-3            # per-step tolerance: cash reaches about 1e10 DKK, binary64 noise is far below this
EQUAL_TOL = 1e-6          # r and r' count as different above this
FLAG_PP = 5.0
LARGE_STEP_ERROR_PP = 0.01
DECISION_FREQ = 6
MWH_CONVERSION_PRICE = 500.0
ACTIVE_CAP = 0.001        # config.mtm_price_return_cap_{min,max}, only for the runs with the cap switched on


def is_percent(variant: str) -> bool:
    return (variant.startswith("L7") and variant != "L7_minus_percent_mtm") or variant.startswith("L8")


def volume_payoff(c: dict) -> np.ndarray:
    """Horizon formula: a contract opened at a decision step t pays position/500 x (settlement_t - entry_t) at t + 6."""
    n = len(c["t"])
    out = np.zeros(n)
    pos, s, e = c["position_post"], c["settlement"], c["entry"]
    for t in range(DECISION_FREQ, n - DECISION_FREQ, DECISION_FREQ):
        out[t + DECISION_FREQ] = pos[t] / MWH_CONVERSION_PRICE * (s[t] - e[t])
    return out


def percent_payoff(c: dict, cap: float | None = None) -> np.ndarray:
    """Legacy formula: the position held over t-1 -> t earns the percentage move of the spot price (cap off in the audit's runs)."""
    p = c["spot"]
    prev = np.r_[p[0], p[:-1]]
    ret = (p - prev) / np.maximum(np.abs(prev), 1e-6)
    if cap is not None:
        ret = np.clip(ret, -cap, cap)
    return c["position_pre"] * ret


def analyze(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    columns = [str(n) for n in z["columns"]]
    rows = z["rows"]
    c = {name: rows[:, i] for i, name in enumerate(columns)}
    e0 = float(z["e0"])
    variant, agent, seed, region = path.stem.split("__")
    pp = 100.0 / e0

    positive = c["access"] + c["roll"] + c["transaction"] + c["funding"]
    own = positive - c["battery"]
    gross = c["gross"]
    x = gross - own
    r, rt = c["r"], c["r_twin"]
    credit = r - x
    credit_twin = rt + gross + own
    width = r + rt + 2 * own
    lower = -rt - 2 * own
    outside = (x > r + TOL_DKK) | (x < lower - TOL_DKK)

    differs = np.abs(r - rt) > EQUAL_TOL
    sign = np.where(gross >= 0, 1.0, -1.0)
    tir = np.where(differs, sign * (np.maximum(r, rt) + own) - own, -own)
    spr = (r - rt) / 2 - own
    tir_wrong = differs & (np.abs(tir - x) > TOL_DKK)
    tir_outside = (tir > r + TOL_DKK) | (tir < lower - TOL_DKK)

    percent = is_percent(variant)
    cap = ACTIVE_CAP if variant.endswith("_capon") else None
    right = percent_payoff(c, cap) if percent else volume_payoff(c)
    other = volume_payoff(c) if percent else percent_payoff(c)

    # the exact cost-shift test on every informative step
    mult = z["cost_multipliers"]
    n_k = len(mult)
    informative = ~np.isnan(c["r_k0"])
    flagged = np.zeros(len(r), dtype=bool)
    for i in np.flatnonzero(informative):
        pairs = [(mult[k] * positive[i] - c["battery"][i], c[f"r_k{k}"][i], c[f"rt_k{k}"][i]) for k in range(n_k)]
        flagged[i] = not single_floor_fit(pairs)[0]
    large_tir = informative & (np.abs(tir - x) * pp > LARGE_STEP_ERROR_PP)
    with_credit = informative & (credit > TOL_DKK)

    reported, booked = r.sum() * pp, x.sum() * pp
    tir_total, spr_total = tir.sum() * pp, spr.sum() * pp
    return dict(
        name=path.stem, variant=variant, agent=agent, seed=int(seed.replace("seed", "")), region=region,
        mode="percent" if percent else "horizon", steps=int(len(r)), e0=e0,
        reported_pct=reported, booked_pct=booked, gap_pp=reported - booked,
        credit_pp=credit.sum() * pp, credit_twin_pp=credit_twin.sum() * pp,
        width_pp=width.sum() * pp, lower_pct=lower.sum() * pp,
        width_over_credit=float(width.sum() / credit.sum()) if credit.sum() * pp > 1e-6 else float("nan"),
        contains_run=bool(lower.sum() * pp - 1e-6 <= booked <= reported + 1e-6),
        steps_outside=int(outside.sum()), steps_credit=int((credit > TOL_DKK).sum()), steps_credit_twin=int((credit_twin > TOL_DKK).sum()),
        min_credit=float(credit.min()), min_credit_twin=float(credit_twin.min()), max_abs_parity=float(np.abs(c["parity"]).max()),
        unknown_writes=int(c["unknown_writes"].sum()), liquidations=int(c["liquidated"].sum()),
        floors_real=int(c["n_floor_real"].sum()), floors_twin=int(c["n_floor_twin"].sum()),
        tir_pct=tir_total, tir_error_pp=tir_total - booked, spr_pct=spr_total, spr_error_pp=spr_total - booked,
        steps_identified=int(differs.sum()), steps_tir_wrong=int(tir_wrong.sum()), steps_tir_outside=int(tir_outside.sum()),
        max_abs_step_tir_error_pp=float(np.abs(tir - x).max() * pp), max_abs_step_spr_error_pp=float(np.abs(spr - x).max() * pp),
        direct_pct=(right - own).sum() * pp, direct_error_pp=(right - gross).sum() * pp,
        direct_max_abs_step_error_pp=float(np.abs(right - gross).max() * pp),
        other_formula_pct=(other - own).sum() * pp, other_formula_error_pp=(other - gross).sum() * pp,
        own_total_pp=own.sum() * pp, battery_pp=c["battery"].sum() * pp, gross_pp=gross.sum() * pp,
        shift_informative=int(informative.sum()), shift_flagged=int(flagged.sum()),
        shift_with_credit=int(with_credit.sum()), shift_flagged_with_credit=int((flagged & with_credit).sum()),
        shift_large_tir=int(large_tir.sum()), shift_flagged_large_tir=int((flagged & large_tir).sum()),
        flag_reconcile=bool(abs(reported - booked) > FLAG_PP), flag_width=bool(width.sum() * pp > FLAG_PP),
        flag_tir_gap=bool(abs(reported - tir_total) > FLAG_PP),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(HERE / "traces"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    paths = sorted(Path(a.traces).glob("*.npz")) if not a.names else [Path(a.traces) / f"{n}.npz" for n in a.names]
    res = [analyze(p) for p in paths]
    if a.to_stdout:
        print(json.dumps(res, indent=1))
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "engine_twin_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    print(f"{len(res)} runs -> {out / 'engine_twin_runs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
