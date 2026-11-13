r"""Post-hoc, after the results were read: is the cost-shift test's firing on runs with no forgiveness real or rounding?

twin_reward.single_floor_fit scales its tolerance to the pair's own values, 1e-9 x (1 + largest |value|). On the engine the
pairs are differences of cash near 1e9 DKK, whose rounding is about 1e-7 DKK, so on a step whose values are small that
tolerance can fall below the rounding. Here the same exact test is repeated with the tolerance scaled to the account as
well, 1e-9 x (1 + largest of |value| and cash before the step), and the misfit of each flagged step is recorded.

Writes results/shift_tolerance.csv and results/shift_tolerance.json; --print writes nothing and prints the summary.
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for root in (HERE.parent / "github" / "accounting-options-rl" / "ACCOUNTING_OPTIONS_2026-10-14", HERE.parent.parent / "ACCOUNTING_OPTIONS_2026-10-14"):
    if (root / "twin_reward" / "twin_reward.py").is_file():
        sys.path.insert(0, str(root / "twin_reward"))
from twin_reward import single_floor_fit  # noqa: E402


def one(path: str) -> dict:
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    c = {str(n): z["rows"][:, i] for i, n in enumerate(z["columns"])}
    mult = z["cost_multipliers"]
    positive = c["access"] + c["roll"] + c["transaction"] + c["funding"]
    inf = np.flatnonzero(~np.isnan(c["r_k0"]))
    plain = scaled = 0
    plain_small = 0
    for i in inf:
        pairs = [(mult[k] * positive[i] - c["battery"][i], c[f"r_k{k}"][i], c[f"rt_k{k}"][i]) for k in range(len(mult))]
        ok, _, _ = single_floor_fit(pairs)
        if ok:
            continue
        plain += 1
        vals = np.abs(np.array(pairs)).max()
        tol = 1e-9 * (1.0 + max(vals, abs(c["cash_pre"][i]))) / (1.0 + vals)
        if not single_floor_fit(pairs, tol=tol)[0]:
            scaled += 1
        else:
            plain_small += 1
    return dict(name=p.stem, informative=int(len(inf)), flagged_plain=plain, flagged_scaled=scaled, cleared_by_scaling=plain_small)


def main() -> int:
    paths = sorted(str(p) for p in (HERE / "traces").glob("*.npz"))
    with mp.Pool(7) as pool:
        res = pool.map(one, paths, chunksize=2)
    bound, section = set(), ""
    for line in (HERE / "JOBS.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            section = line
        elif line.strip() and "floor bound" in section:
            bound.add(line.strip())
    a = [r for r in res if "_capon" not in r["name"]]
    b = [r for r in a if r["name"] in bound]
    n = [r for r in a if r["name"] not in bound]
    summary = dict(
        bound_runs_flagged_plain=sum(r["flagged_plain"] > 0 for r in b), bound_runs_flagged_scaled=sum(r["flagged_scaled"] > 0 for r in b),
        comparison_runs_flagged_plain=sum(r["flagged_plain"] > 0 for r in n), comparison_runs_flagged_scaled=sum(r["flagged_scaled"] > 0 for r in n),
        bound_steps_flagged_plain=sum(r["flagged_plain"] for r in b), bound_steps_flagged_scaled=sum(r["flagged_scaled"] for r in b),
        comparison_steps_flagged_plain=sum(r["flagged_plain"] for r in n), comparison_steps_flagged_scaled=sum(r["flagged_scaled"] for r in n),
        comparison_runs_scaled=[r["name"] for r in n if r["flagged_scaled"] > 0],
    )
    if "--print" in sys.argv:
        print(json.dumps(summary, indent=1))
        return 0
    out = HERE / "results"
    with (out / "shift_tolerance.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    (out / "shift_tolerance.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
