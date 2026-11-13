r"""Exploratory, after the rules were read: why the bracket misses on real clones (S2), why the twin differs from the replay (S1), and why the reading misses steps of the uncapped percentage payoff.
Reads the campaign's scans and traces; changes nothing. Writes scan_campaign/mechanisms_scan.json."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
from reward_diagram import read_diagram  # noqa: E402

E0 = 662068965.5172415
PP = 100 / E0
out = {}
for p in sorted((HERE / "scan_campaign" / "scans").glob("*.npz")):
    name = p.stem
    z = dict(np.load(p, allow_pickle=False))
    grid = z["read_grid"]
    idx = {float(t): j for j, t in enumerate(grid)}
    r1, rm = z["phi"][:, idx[1.0]], z["phi"][:, idx[-1.0]]
    C1, Cm = z["own_1"], z["own_m"]
    cols = [str(c) for c in z["real_columns"]]
    real = {c: z["real"][:, i] for i, c in enumerate(cols)}
    x = real["gross"] - (real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"])
    lo = -rm - 2.0 * C1
    over_low, over_high = lo - x, x - r1
    miss = (over_low > 1.0) | (over_high > 1.0)
    # the premise: the same own cost on both branches, and the position unchanged on the twin
    cost_gap = np.abs(C1 - Cm)
    pos_gap = np.abs(z["pos_1"] - z["pos_m"])
    # with the twin's own cost in the bound: -r' - C(1) - C(-1)
    lo2 = -rm - C1 - Cm
    miss2 = (lo2 - x > 1.0) | (over_high > 1.0)
    out[name] = dict(reads=len(x), outside=int(miss.sum()), largest_dkk=float(max(over_low.max(), over_high.max())), largest_points=float(max(over_low.max(), over_high.max()) * PP),
                     cost_gap_steps=int((cost_gap > 1.0).sum()), of_outside_with_cost_gap=int((miss & (cost_gap > 1.0)).sum()), outside_with_each_branch_cost=int(miss2.sum()), position_differs_steps=int((pos_gap > 1.0).sum()),
                     twin_replay_gap_steps=int((np.abs(rm - real["r_twin"]) > 1.0).sum()), liquidated_steps=int((real["liquidated"] > 0).sum()), sum_cost_gap_points=float(cost_gap.sum() * PP))
    # the reading's misses, on the sampled full diagrams
    if "d_line" in z:
        fields = [str(f) for f in z["fields"]]
        g = z["grid"]
        order = np.argsort(g)
        ts = g[order]
        phi = z["d_line"][:, :, fields.index("r")][:, order]
        gross = z["d_line"][:, :, fields.index("gross")][:, order]
        pos = z["d_line"][:, :, fields.index("position")][:, order]
        rd = {c: z["d_real"][:, i] for i, c in enumerate(cols)}
        Cd = rd["access"] + rd["roll"] + rd["transaction"] + rd["funding"] - rd["battery"]
        xd = rd["gross"] - Cd
        near = np.abs(ts) <= 8.0
        i1 = int(np.argmin(np.abs(ts - 1.0)))
        tot = slope_short = moved = pos_moves = reaches = 0
        for k in range(len(phi)):
            d = read_diagram(ts[near], phi[k][near], rel_tol=1e-9, abs_tol=1e-3)
            err = abs(d.gross - Cd[k] - xd[k])
            if abs(rd["gross"][k]) <= 1.0:
                continue
            moved += 1
            if err >= 1.0:
                tot += 1
                slope_short += abs(d.gross) < abs(rd["gross"][k]) - 1.0
                pos_moves += float(np.abs(pos[k][near] - pos[k][i1]).max()) > 1.0
                s = np.diff(phi[k]) / np.diff(ts)
                reaches += float(np.abs(s).max()) >= abs(rd["gross"][k]) * (1.0 - 1e-6)
        out[name].update(moved_sampled=moved, reading_misses_sampled=tot, of_which_slope_below_gross=int(slope_short), of_which_position_moves_within_8=int(pos_moves), of_which_full_grid_reaches_the_gross=int(reaches))
(HERE / "scan_campaign" / "mechanisms_scan.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in out.items():
    print(k[:46].ljust(46), {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items() if a not in ("reads",)})
