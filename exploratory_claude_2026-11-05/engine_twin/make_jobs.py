"""The job list, fixed from the audit's run table before the full evaluation: every run in which the floor bound, plus a
comparison sample of runs in which it did not, chosen to include the protocols most likely to raise a false alarm."""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "github" / "accounting-options-rl" / "results" / "campaign" / "results_long.csv")
name = lambda r: f"{r.variant}__{r.agent}__seed{int(r.seed)}__{r.region}"  # noqa: E731
bound = d[d.sleeve_cash_floor_hit == True]  # noqa: E712
assert len(bound) == 54
free = d[d.sleeve_cash_floor_hit == False]  # noqa: E712
SUB = {7, 42, 123}
pick = [
    free[(free.variant == "L0_strict") & (free.agent.isin(["marl", "anchor"])) & (free.seed == 7)],
    free[(free.variant == "L1_price_taker") & (free.agent == "marl") & (free.seed.isin(SUB))],
    free[(free.variant == "L1_price_taker") & (free.agent == "anchor")],
    free[(free.variant == "L3_price_taker_no_solvency") & (free.agent == "anchor")],
    free[(free.variant == "L3_price_taker_no_solvency") & (free.agent == "feasible") & (free.seed.isin(SUB))],
    free[(free.variant == "L7_thesis_like") & (free.agent.isin(["anchor", "feasible", "marl"]))],
    free[(free.variant.isin(["L7_minus_interp", "L7_minus_legacy_fee", "L7_minus_sweeper"])) & (free.agent == "marl")],
]
comparison = pd.concat(pick)
assert not set(comparison.index) & set(bound.index)
capon = d[(d.variant == "L7_thesis_like") & (d.agent.isin(["marl", "anchor"]))]
assert len(capon) == 22
lines = ["# L7 with the percent payoff's cap really on (22): the audit's runs had it off"]
lines += [f"L7_thesis_like_capon__{r.agent}__seed{int(r.seed)}__{r.region}" for r in capon.itertuples()]
lines += ["# floor bound in the audit (54)"] + [name(r) for r in bound.itertuples()]
lines += [f"# comparison: floor did not bind ({len(comparison)})"] + [name(r) for r in comparison.itertuples()]
(HERE / "JOBS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(len(bound), len(comparison))
print(comparison.groupby(["variant", "agent"]).size().to_string())
