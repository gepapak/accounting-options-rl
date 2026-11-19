"""Tests A and B: direct simulation of the three accounting rules against predictions.json (written before this ran).
A: Gaussian fine steps (64 a day, 86 days), the same paths monitored m = 1, 4, 16, 64 times a day; rules LL, TERM, ZF.
B: empirical daily bars (fair signs on EURUSD or 2021 crypto magnitudes), capped loss, m = 1.
Equity in units of the initial equity: 1 + L (P_k / P_0 - 1), prices compounded, no fees."""
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "predictions.json").read_text())
s_e, n_d = P["inputs"]["s_eurusd"], P["inputs"]["n_days"]
rng = np.random.default_rng(20261117)
M, BATCH, NB = 64, 10_000, 20            # 200,000 paths
LEVS, MS = (20, 50, 200), (1, 4, 16, 64)

acc = {(L, m, r): [] for L in LEVS for m in MS for r in ("LL", "ZF", "ruin")}
term = {L: [] for L in LEVS}
for _ in range(NB):
    r = rng.normal(0.0, s_e / sqrt(M), size=(BATCH, n_d * M)).astype(np.float64)
    x = np.cumprod(1.0 + r, axis=1) - 1.0                       # P_k / P_0 - 1 on the fine grid
    for m in MS:
        xm = x[:, (M // m) - 1::(M // m)]                       # the same paths, checked m times a day
        for L in LEVS:
            y = -L * xm                                          # booked loss in units of the initial equity
            hit = y >= 1.0
            any_hit = hit.any(axis=1)
            first = hit.argmax(axis=1)
            over = np.where(any_hit, y[np.arange(BATCH), first] - 1.0, 0.0)
            acc[(L, m, "LL")].append(over)
            acc[(L, m, "ruin")].append(any_hit)
            acc[(L, m, "ZF")].append(np.maximum(y.max(axis=1) - 1.0, 0.0))
    for L in LEVS:
        term[L].append(np.maximum(-L * x[:, -1] - 1.0, 0.0))

rows, ok = [], True
print("A. Gaussian fine steps: simulated (standard error) against predicted, in % of the initial equity")
print(f"{'L':>4} {'m':>3} | {'ruin sim':>8} {'pred':>6} | {'LL sim':>8} {'(se)':>6} {'pred':>8} {'ratio':>6} | {'TERM sim':>8} {'pred':>8} {'ratio':>6} | "
      f"{'ZF sim':>8} {'pred':>8} {'ratio':>6} | ZF/TERM sim  pred")
for pr in P["A"]:
    L, m = pr["L"], pr["m"]
    ll = np.concatenate(acc[(L, m, "LL")]); zf = np.concatenate(acc[(L, m, "ZF")]); ru = np.concatenate(acc[(L, m, "ruin")]).mean()
    te = np.concatenate(term[L])
    r_ll, r_te, r_zf = ll.mean() / pr["LL"], te.mean() / pr["TERM"], zf.mean() / pr["ZF"]
    se_rel = ll.std() / sqrt(len(ll)) / ll.mean()
    cell_ok = all(abs(x - 1) <= 0.05 for x in (r_ll, r_te, r_zf))
    ok &= cell_ok
    rows.append(dict(L=L, m=m, ruin_sim=float(ru), LL_sim=float(ll.mean()), LL_se=float(ll.std() / sqrt(len(ll))), TERM_sim=float(te.mean()),
                     ZF_sim=float(zf.mean()), r_LL=float(r_ll), r_TERM=float(r_te), r_ZF=float(r_zf)))
    print(f"{L:>4} {m:>3} | {ru:8.4f} {pr['p_ruin']:6.4f} | {100 * ll.mean():8.3f} {100 * ll.std() / sqrt(len(ll)):6.3f} {100 * pr['LL']:8.3f} {r_ll:6.3f} | "
          f"{100 * te.mean():8.3f} {100 * pr['TERM']:8.3f} {r_te:6.3f} | {100 * zf.mean():8.3f} {100 * pr['ZF']:8.3f} {r_zf:6.3f} | "
          f"{zf.mean() / te.mean():6.3f} {pr['ZF'] / pr['TERM']:6.3f}{'' if cell_ok else '   <-- outside 5%'}")
slopes = {}
for L in LEVS:
    v = [np.concatenate(acc[(L, m, "LL")]).mean() for m in MS]
    slopes[L] = float(np.polyfit(np.log(MS), np.log(v), 1)[0])
print("log-log slope of LL against m:", ", ".join(f"L={L}: {s:+.3f}" for L, s in slopes.items()), "(criterion -0.56 to -0.42)")
ok &= all(-0.56 <= s <= -0.42 for s in slopes.values())
print("A:", "ALL CRITERIA MET" if ok else "NOT ALL CRITERIA MET")

print("\nB. Empirical daily bars, capped loss, m = 1: simulated against predicted, in % of the initial equity")
okB = True
rowsB = []
for name, mags in (("eurusd", np.load(HERE / "eurusd_mags.npy")), ("crypto", np.load(HERE / "crypto_mags.npy"))):
    for pr in [b for b in P["B"] if b["market"] == name]:
        L, over, ruin = pr["L"], [], []
        for _ in range(20):
            r = rng.choice([-1.0, 1.0], size=(BATCH, n_d)) * mags[rng.integers(0, len(mags), size=(BATCH, n_d))]
            y = -L * (np.cumprod(1.0 + r, axis=1) - 1.0)
            hit = y >= 1.0
            any_hit = hit.any(axis=1)
            over.append(np.where(any_hit, y[np.arange(BATCH), hit.argmax(axis=1)] - 1.0, 0.0)); ruin.append(any_hit)
        over, ruin = np.concatenate(over), np.concatenate(ruin).mean()
        ratio, ratio_g = over.mean() / pr["LL"], over.mean() / pr["LL_gaussian_constant"]
        tol = 0.08 if name == "eurusd" else 0.25
        cell_ok = abs(ratio - 1) <= tol
        okB &= cell_ok
        rowsB.append(dict(market=name, L=L, ruin_sim=float(ruin), LL_sim=float(over.mean()), ratio=float(ratio), ratio_gaussian=float(ratio_g)))
        print(f"{name:<7} L={L:>3}: ruin {ruin:.4f} (pred {pr['p_ruin']:.4f}) | LL {100 * over.mean():7.3f} (se {100 * over.std() / sqrt(len(over)):.3f}) "
              f"pred {100 * pr['LL']:7.3f} ratio {ratio:.3f} | with the Gaussian constant: ratio {ratio_g:.3f}{'' if cell_ok else '   <-- outside tolerance'}")
print("B:", "ALL CRITERIA MET" if okB else "NOT ALL CRITERIA MET")
(HERE / "test_a_sim.json").write_text(json.dumps(dict(A=rows, slopes=slopes, A_ok=bool(ok), B=rowsB, B_ok=bool(okB)), indent=1))
