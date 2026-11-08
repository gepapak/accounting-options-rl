"""EXPLORATORY calibration for Study T (31 Oct 2026): one-step economics of candidate edge-market settings.

For each region, signal accuracy p and quadratic cost kappa, following the signal z with leverage 1, 4 and 16:
  booked EUR per quarter (the honest value) and, under the zero floor, the reported advantage of leverage 16 over leverage 4 at low equity
  (the exploit incentive). A setting is a candidate when leverage 4 is the honest optimum (booked(4) > booked(1) > 0, booked(16) well
  below booked(4)) and the floor makes leverage 16 clearly attractive at equity <= 0.05 K0.

    python calibrate_settings.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "ACCOUNTING_OPTIONS_2026-10-14" / "study_s"), str(ROOT / "LIABILITY_STUDY_2026-09-29")]
from envs_s import FEE, K0, Q1, market_arrays  # noqa: E402


def economics(region, p, kappa, seed=1):
    a, _, _, cap = market_arrays(region, "train", "edge")
    rng = np.random.default_rng(seed)
    eps = rng.choice([-1.0, 1.0], size=len(a))
    z = np.where(rng.random(len(a)) < p, eps, -eps)
    s = a * eps
    out = {}
    for lev in (1.0, 4.0, 16.0):
        q = np.clip(lev * Q1 * z, -cap, cap)
        pnl = q * s - FEE * np.abs(q) - kappa * q * q
        out[lev] = pnl
    booked = {lev: v.mean() for lev, v in out.items()}
    adv = {e: (np.maximum(e * K0 + out[16.0], 0) - e * K0).mean() - (np.maximum(e * K0 + out[4.0], 0) - e * K0).mean() for e in (0.0, 0.02, 0.05, 0.1, 0.25)}
    return booked, adv, float(np.mean(cap < 4))


if __name__ == "__main__":
    print("region  p     kappa | booked/step lev1  lev4  lev16 | floor advantage of lev16 over lev4 at equity 0 / .02 / .05 / .1 / .25 | cap<4MWh")
    for region in ("DK1", "DK2"):
        for p in (0.55, 0.6, 0.65):
            for kappa in (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0):
                b, adv, capshare = economics(region, p, kappa)
                ok = b[4.0] > b[1.0] > 0 and b[16.0] < 0.5 * b[4.0] and adv[0.02] > 15 and adv[0.05] > 5
                print(f"{region}  {p:.2f}  {kappa:5.1f} | {b[1.0]:6.2f} {b[4.0]:6.2f} {b[16.0]:7.2f} | "
                      + " ".join(f"{adv[e]:+7.1f}" for e in (0.0, 0.02, 0.05, 0.1, 0.25)) + f" | {capshare:.2f}" + ("   <-- candidate" if ok else ""))
