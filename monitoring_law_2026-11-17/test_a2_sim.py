"""Tests A2, A3 and B3 (predictions_v2.json).
A2: ADDITIVE profit and loss (the paper's ledger model, Y_k = -L sum r_j), Gaussian fine steps, rules LL, TERM, ZF, against the
    additive forms. Never simulated before.
A3: compounded prices (the simulation of test_a_sim.py, new seed) against the compounded-price forms. Post hoc for these cells.
B3: empirical daily bars, compounded, against the compounded-price forms with the log price's overshoot constant. Post hoc."""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
P1 = json.loads((HERE / "predictions.json").read_text())
P = json.loads((HERE / "predictions_v2.json").read_text())
s_e, n_d = P1["inputs"]["s_eurusd"], P1["inputs"]["n_days"]
rng = np.random.default_rng(20261118)
M, BATCH, NB = 64, 10_000, 20
LEVS, MS = (20, 50, 200), (1, 4, 16, 64)


def rules(y):
    """y: booked loss in units of the initial equity on the monitored grid. Returns forgiven under LL and ZF, and the ruin flag."""
    hit = y >= 1.0
    any_hit = hit.any(axis=1)
    over = np.where(any_hit, y[np.arange(len(y)), hit.argmax(axis=1)] - 1.0, 0.0)
    return over, np.maximum(y.max(axis=1) - 1.0, 0.0), any_hit


acc = {(k, L, m, r): [] for k in ("add", "cmp") for L in LEVS for m in MS for r in ("LL", "ZF", "ruin")}
term = {(k, L): [] for k in ("add", "cmp") for L in LEVS}
for _ in range(NB):
    r = rng.normal(0.0, s_e / sqrt(M), size=(BATCH, n_d * M))
    xs = {"add": np.cumsum(r, axis=1), "cmp": np.cumprod(1.0 + r, axis=1) - 1.0}
    for k, x in xs.items():
        for m in MS:
            xm = x[:, (M // m) - 1::(M // m)]
            for L in LEVS:
                ll, zf, ru = rules(-L * xm)
                acc[(k, L, m, "LL")].append(ll); acc[(k, L, m, "ZF")].append(zf); acc[(k, L, m, "ruin")].append(ru)
        for L in LEVS:
            term[(k, L)].append(np.maximum(-L * x[:, -1] - 1.0, 0.0))

out = {}
for k, key, title in (("add", "A2", "A2. Additive profit and loss, additive forms (never simulated before)"),
                      ("cmp", "A3", "A3. Compounded prices, compounded-price forms (post hoc for these cells)")):
    ok, rows = True, []
    print(title)
    print(f"{'L':>4} {'m':>3} | {'ruin sim':>8} {'pred':>6} | {'LL sim':>8} {'pred':>8} {'ratio':>6} | {'TERM sim':>8} {'pred':>8} {'ratio':>6} | "
          f"{'ZF sim':>8} {'pred':>8} {'ratio':>6} | ZF/TERM sim  pred")
    for pr in P[key]:
        L, m = pr["L"], pr["m"]
        ll = np.concatenate(acc[(k, L, m, "LL")]).mean(); zf = np.concatenate(acc[(k, L, m, "ZF")]).mean()
        ru = np.concatenate(acc[(k, L, m, "ruin")]).mean(); te = np.concatenate(term[(k, L)]).mean()
        r_ll, r_te, r_zf = ll / pr["LL"], te / pr["TERM"], zf / pr["ZF"]
        cell = all(abs(x - 1) <= 0.05 for x in (r_ll, r_te, r_zf))
        ok &= cell
        rows.append(dict(L=L, m=m, ruin=float(ru), LL=float(ll), TERM=float(te), ZF=float(zf), r_LL=float(r_ll), r_TERM=float(r_te), r_ZF=float(r_zf)))
        print(f"{L:>4} {m:>3} | {ru:8.4f} {pr['p_ruin']:6.4f} | {100 * ll:8.3f} {100 * pr['LL']:8.3f} {r_ll:6.3f} | {100 * te:8.3f} "
              f"{100 * pr['TERM']:8.3f} {r_te:6.3f} | {100 * zf:8.3f} {100 * pr['ZF']:8.3f} {r_zf:6.3f} | {zf / te:6.3f} {pr['ZF'] / pr['TERM']:6.3f}"
              f"{'' if cell else '   <-- outside 5%'}")
    worst = max(max(abs(r["r_LL"] - 1), abs(r["r_TERM"] - 1), abs(r["r_ZF"] - 1)) for r in rows)
    print(f"{key}: {'ALL CELLS WITHIN 5%' if ok else 'NOT ALL CELLS WITHIN 5%'} (largest deviation {100 * worst:.1f}%)\n")
    out[key] = dict(rows=rows, ok=bool(ok), worst=float(worst))

print("B3. Empirical daily bars, compounded prices, compounded-price forms with the log price's overshoot constant (post hoc)")
okB, rowsB = True, []
for name, mags in (("eurusd", np.load(HERE / "eurusd_mags.npy")), ("crypto", np.load(HERE / "crypto_mags.npy"))):
    for pr in [b for b in P["B3"] if b["market"] == name]:
        L, res = pr["L"], dict(LL=[], ZF=[], ruin=[], TERM=[])
        for _ in range(20):
            r = rng.choice([-1.0, 1.0], size=(BATCH, n_d)) * mags[rng.integers(0, len(mags), size=(BATCH, n_d))]
            y = -L * (np.cumprod(1.0 + r, axis=1) - 1.0)
            ll, zf, ru = rules(y)
            res["LL"].append(ll); res["ZF"].append(zf); res["ruin"].append(ru); res["TERM"].append(np.maximum(y[:, -1] - 1.0, 0.0))
        v = {k: float(np.concatenate(x).mean()) for k, x in res.items()}
        tol = 0.08 if name == "eurusd" else 0.25
        rr = {k: v[k] / pr[k] for k in ("LL", "TERM", "ZF")}
        cell = all(abs(x - 1) <= tol for x in rr.values())
        okB &= cell
        rowsB.append(dict(market=name, L=L, **v, **{f"r_{k}": float(x) for k, x in rr.items()}))
        print(f"{name:<7} L={L:>3}: ruin {v['ruin']:.4f} (pred {pr['p_ruin']:.4f}) | LL {100 * v['LL']:7.3f} / {100 * pr['LL']:7.3f} = {rr['LL']:.3f} | "
              f"TERM {100 * v['TERM']:8.3f} / {100 * pr['TERM']:8.3f} = {rr['TERM']:.3f} | ZF {100 * v['ZF']:8.3f} / {100 * pr['ZF']:8.3f} = {rr['ZF']:.3f}"
              f"{'' if cell else '   <-- outside tolerance'}")
print("B3:", "ALL CELLS WITHIN TOLERANCE" if okB else "NOT ALL CELLS WITHIN TOLERANCE")
out["B3"] = dict(rows=rowsB, ok=bool(okB))
(HERE / "test_a2_sim.json").write_text(json.dumps(out, indent=1))
