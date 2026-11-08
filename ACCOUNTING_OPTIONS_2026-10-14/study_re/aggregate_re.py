"""Study RE aggregation (PREREGISTRATION_RE.md): writes runs_re.csv, cells_re.csv and VERDICTS_RE.md from results_re/jobs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOBS = HERE / "results_re" / "jobs"
SCALES = (1.0, 0.3, 0.1, 0.03, 0.01)
SEEDS = (0, 1, 2, 3, 4)
EGRID = ("0", "0.02", "0.05", "0.1", "0.25", "0.5", "1", "2", "3")


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(JOBS.glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({k: v for k, v in r.items() if k != "lev"} | {f"L({e})": r["lev"][e] for e in EGRID})
    return pd.DataFrame(rows)


def main() -> int:
    runs = load()
    runs.to_csv(HERE / "runs_re.csv", index=False)
    reach = json.loads((HERE / "reach_re.json").read_text())["scales"]
    lines = ["# Study RE verdicts", "", f"- runs: {len(runs)} (expected {2 * len(SCALES) * len(SEEDS)})", ""]
    cells = []
    pairs = {}
    for c in SCALES:
        zf = runs[(runs.rule == "ZF") & (runs.scale == c)].set_index("seed").sort_index()
        fl = runs[(runs.rule == "FL") & (runs.scale == c)].set_index("seed").sort_index()
        common = zf.index.intersection(fl.index)
        D = (zf.loc[common, "X"] - fl.loc[common, "X"]).to_numpy(float)
        same_probe = np.array([all(abs(zf.loc[s, f"L({e})"] - fl.loc[s, f"L({e})"]) < 1e-12 for e in EGRID) for s in common])
        same_params = np.array([zf.loc[s, "policy_sha256"] == fl.loc[s, "policy_sha256"] for s in common])
        pairs[c] = dict(D=D, same_probe=same_probe, same_params=same_params)
        cells.append(dict(scale=c, reach_R=reach[str(c)]["reach_R"], n_pairs=len(common),
                          mean_D=float(D.mean()) if len(D) else np.nan, pairs_D_positive=int((D > 0).sum()),
                          pairs_identical_probe=int(same_probe.sum()), pairs_identical_params=int(same_params.sum()),
                          ZF_mean_X=float(zf.loc[common, "X"].mean()), FL_mean_X=float(fl.loc[common, "X"].mean()),
                          ZF_mean_L002=float(zf.loc[common, "L(0.02)"].mean()), FL_mean_L002=float(fl.loc[common, "L(0.02)"].mean()),
                          ZF_floor_episodes_min=int(zf.loc[common, "train_floor_episodes"].min()),
                          ZF_floor_episodes_max=int(zf.loc[common, "train_floor_episodes"].max()),
                          FL_closed_episodes_min=int(fl.loc[common, "train_floor_episodes"].min()),
                          FL_closed_episodes_max=int(fl.loc[common, "train_floor_episodes"].max()),
                          ZF_reported_minus_ledger_pct=float((zf.loc[common, "drawmean_reported_return_pct"]
                                                              - zf.loc[common, "drawmean_ledger_return_pct"]).mean())))
    cells = pd.DataFrame(cells)
    cells.to_csv(HERE / "cells_re.csv", index=False)
    lines.append(cells.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    def row(c):
        return cells[cells.scale == c].iloc[0]

    # RE1: at c = 0.01 (predicted reach 0) the FL and ZF agents of every pair are identical on the probe grid, and no ZF run
    # reached the floor in training.
    p = pairs[0.01]
    re1 = bool(p["same_probe"].all()) and row(0.01).ZF_floor_episodes_max == 0 and len(p["D"]) == len(SEEDS)
    lines.append(f"- RE1 (reach is necessary): c=0.01 identical probes in {int(p['same_probe'].sum())}/{len(p['D'])} pairs, identical "
                 f"parameters in {int(p['same_params'].sum())}/{len(p['D'])}, ZF floor-reaching training episodes max "
                 f"{row(0.01).ZF_floor_episodes_max} (need all pairs identical on the probe grid and 0 floor episodes) -> "
                 f"{'SUPPORTED' if re1 else 'NOT SUPPORTED'}")
    # RE2: at c = 1 the exploit is present: mean D >= 4 and D > 0 in at least 4 of 5 pairs.
    p = pairs[1.0]
    re2 = len(p["D"]) == len(SEEDS) and p["D"].mean() >= 4.0 and (p["D"] > 0).sum() >= 4
    lines.append(f"- RE2 (the exploit where the reach is large): c=1 mean D = {p['D'].mean():.2f}, D > 0 in {(p['D'] > 0).sum()}/{len(p['D'])} "
                 f"pairs (need mean >= 4 and >= 4 of 5) -> {'SUPPORTED' if re2 else 'NOT SUPPORTED'}")
    # RE3: the exploit follows the reach: mean D at every scale with R >= 100 (c = 1, 0.3, 0.1) exceeds mean D at c = 0.03 and at
    # c = 0.01 by at least 2 units of leverage.
    hi = [pairs[c]["D"].mean() for c in (1.0, 0.3, 0.1)]
    lo = [pairs[c]["D"].mean() for c in (0.03, 0.01)]
    re3 = all(h >= l + 2.0 for h in hi for l in lo)
    lines.append(f"- RE3 (the exploit follows the reach): mean D at c=1, 0.3, 0.1 = {', '.join(f'{h:.2f}' for h in hi)}; at c=0.03, 0.01 = "
                 f"{', '.join(f'{l:.2f}' for l in lo)} (need every high >= every low + 2) -> {'SUPPORTED' if re3 else 'NOT SUPPORTED'}")
    # Descriptive: floor-reaching training episodes of the ZF runs against the predicted reach.
    lines.append("- Descriptive (floor-reaching ZF training episodes against the predicted reach R at maximal leverage): "
                 + "; ".join(f"c={c:g}: R={reach[str(c)]['reach_R']:.1f}, observed {row(c).ZF_floor_episodes_min}-{row(c).ZF_floor_episodes_max}"
                             for c in SCALES))
    (HERE / "VERDICTS_RE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
