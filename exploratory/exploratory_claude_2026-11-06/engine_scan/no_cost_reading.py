r"""Exploratory, written after the rules were hashed: how much of the reading's exactness rests on the declared own cost? Reads the campaign's scans; changes nothing.

Without a declared cost the only reading of the cost is phi(0), which is -C where the account can pay it and shows nothing of a cost forgiven at no move (manuscript Proposition "The reward diagram", part 5).
On an escalated step: gross from the tails + phi(0). On a step that did not escalate the reward itself. Off the read steps: the reward, whose only content there is the cost.
Writes scan_campaign/no_cost_reading.json."""
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
    phi = z["phi"]
    idx = {float(t): j for j, t in enumerate(grid)}
    r1, rm = phi[:, idx[1.0]], phi[:, idx[-1.0]]
    C = z["own_1"]
    tol = 1e-3 + 1e-9 * np.abs(z["cash_pre"] + z["mtm_pre"])
    est_declared = r1.copy()
    est_free = r1.copy()
    for k in np.flatnonzero(np.abs(r1 + rm + 2.0 * C) > tol):
        have = ~np.isnan(phi[k])
        order = np.argsort(grid[have])
        d = read_diagram(grid[have][order], phi[k][have][order], rel_tol=1e-9, abs_tol=1e-3)
        est_declared[k] = d.gross - C[k]
        est_free[k] = d.gross + phi[k, idx[0.0]]
    tr = np.load(HERE / "scan_campaign" / "traces" / f"{name}.npz", allow_pickle=False)
    cols = [str(c) for c in tr["columns"]]
    R = tr["rows"]
    c = lambda k: R[:, cols.index(k)]  # noqa: E731
    allC = c("access") + c("roll") + c("transaction") + c("funding") - c("battery")
    allx = c("gross") - allC
    allr = c("r")
    mask = np.isin(c("t").astype(int), z["t"])
    booked = float(allx.sum() * PP)
    declared = float((-allC[~mask]).sum() * PP + est_declared.sum() * PP)
    free = float(allr[~mask].sum() * PP + est_free.sum() * PP)
    cost_total = float(allC.sum() * PP)
    out[name] = dict(booked=booked, reported=float(allr.sum() * PP), cost_total=cost_total, gross_total=float(c("gross").sum() * PP), audit_declared=declared, audit_free=free, err_declared=abs(declared - booked), err_free=abs(free - booked), credit_total=float((allr - allx).sum() * PP))
    print(f"{name[:50]:50s} booked {booked:10.2f} reported {out[name]['reported']:8.2f} cost {cost_total:9.2f} gross {out[name]['gross_total']:9.2f} | error with declared cost {abs(declared - booked):8.4f} | with phi(0) only {abs(free - booked):9.2f} | total credit {out[name]['credit_total']:9.2f}")
(HERE / "scan_campaign" / "no_cost_reading.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
