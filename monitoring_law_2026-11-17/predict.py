"""Writes the law's predictions from inputs only (bar-return magnitudes, leverage, horizon, time step, fee), before any simulation or
engine run: predictions.json and PREDICTIONS.md. No accounting rule is simulated here."""
import json
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

import law

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
eur = np.load(HERE / "eurusd_mags.npy")
cry = np.load(HERE / "crypto_mags.npy")
s_e, s_c = sqrt(float(np.mean(eur ** 2))), sqrt(float(np.mean(cry ** 2)))

# the overshoot constant rho / s: checks of the integral on two laws with known answers, then the two empirical bar laws
from statistics import NormalDist  # noqa: E402

qs = (np.arange(3000) + 0.5) / 3000                                         # quantile grids instead of random samples (no sampling noise)
g = np.abs(np.array([NormalDist().inv_cdf(0.5 + q / 2) for q in qs]))
g /= sqrt(float(np.mean(g ** 2)))
e = -np.log(1.0 - qs)
chk_gauss = law.rho_siegmund(g)                                             # expect 0.5826
chk_lapl = law.rho_siegmund(e) / sqrt(float(np.mean(e ** 2)))               # Laplace: expect 0.7071
rho_e, rho_c = law.rho_siegmund(eur), law.rho_siegmund(cry)
mc_e, n_e = law.rho_far_barrier_mc(eur)
mc_c, n_c = law.rho_far_barrier_mc(cry)

N_D = 86            # bars monitored after entry in the engines' 88-bar frames (entry at the close of bar 1, checks through bar 87)
P = dict(inputs=dict(s_eurusd=s_e, s_crypto=s_c, rho_over_s_eurusd=rho_e / s_e, rho_over_s_crypto=rho_c / s_c,
                     rho_over_s_eurusd_mc=mc_e / s_e, rho_over_s_crypto_mc=mc_c / s_c, check_gaussian=chk_gauss, check_laplace=chk_lapl,
                     n_days=N_D, beta=law.BETA))

# A. Gaussian fine steps, three rules, time step m
A = []
for L in (20, 50, 200):
    for m in (1, 4, 16, 64):
        s, n = s_e / sqrt(m), N_D * m
        p, ll = law.s_ll(L, s, n)
        A.append(dict(L=L, m=m, p_ruin=p, LL=ll, TERM=law.s_term(L, s, n), ZF=law.s_zf(L, s, n)))
P["A"] = A

# B. empirical bars at m = 1 (the overshoot from Siegmund's integral; the Gaussian constant for comparison)
B = []
for name, s, rho, levs in (("eurusd", s_e, rho_e, (20, 50, 100, 200)), ("crypto", s_c, rho_c, (2, 5, 10))):
    for L in levs:
        p, ll = law.s_ll(L, s, N_D, rho=rho)
        pg, llg = law.s_ll(L, s, N_D)
        B.append(dict(market=name, L=L, p_ruin=p, LL=ll, LL_gaussian_constant=llg))
P["B"] = B

# C. unmodified backtesting.py on fine Gaussian paths (16 steps a day), monitored m times a day; fee-free and with 2 bp a side
C = []
for fee in (0.0, 0.0002):
    for L in (50, 200):
        Leff = 0.9999 * L / (1.0 + fee)
        for m in (1, 4, 16):
            p, ll = law.s_ll(Leff, s_e / sqrt(m), N_D * m, fee=fee)
            C.append(dict(fee=fee, L=L, m=m, p_ruin=p, forgiven=ll))
P["C"] = C

# D1. the registered Study XE census (EURUSD magnitudes, daily bars, 2 bp a side)
D1 = []
for L in (20, 50, 100, 200):
    Leff = 0.9999 * L / 1.0002
    p, ll = law.s_ll(Leff, s_e, N_D, rho=rho_e, fee=0.0002)
    D1.append(dict(L=L, p_ruin=p, forgiven=ll))
P["D1"] = D1
(HERE / "predictions.json").write_text(json.dumps(P, indent=1))

L_ = []
L_.append("# Predictions of the monitoring law, written before any test ran\n")
L_.append(f"Generated {datetime.now().isoformat(timespec='seconds')} by `predict.py` from inputs only. No accounting rule was simulated and no "
          "engine was run to produce these numbers. Exploratory, not registered.\n")
L_.append("## Inputs\n")
L_.append(f"- Daily bar volatility: EURUSD magnitudes {100 * s_e:.4f}%, 2021 crypto magnitudes {100 * s_c:.3f}%.")
L_.append(f"- Overshoot constant rho / s from Siegmund's integral: EURUSD {rho_e / s_e:.4f}, crypto {rho_c / s_c:.4f}. Gaussian bars: "
          f"{law.BETA:.4f}. Fixed-size moves: 0.5.")
L_.append(f"- Checks of the integral on laws with known answers (3,000-point quantile grids, so the far tail is slightly cut): Gaussian "
          f"{chk_gauss:.4f} (exact 0.5826), Laplace {chk_lapl:.4f} (exact 0.7071).")
L_.append(f"- Far-barrier Monte Carlo of the same constant (a second method): EURUSD {mc_e / s_e:.4f} ({n_e} crossings), crypto "
          f"{mc_c / s_c:.4f} ({n_c}).\n")
L_.append("## A. Three rules on Gaussian fine steps (subsidy in % of the initial equity; m = checks per day; 86 days)\n")
L_.append("| L | m | P(ruin) | LL (capped loss, closes) | TERM | ZF (zero floor, continues) | ZF / TERM |")
L_.append("|---|---|---|---|---|---|---|")
for r in A:
    L_.append(f"| {r['L']} | {r['m']} | {r['p_ruin']:.4f} | {100 * r['LL']:.3f} | {100 * r['TERM']:.3f} | {100 * r['ZF']:.3f} | {r['ZF'] / r['TERM']:.3f} |")
L_.append("\nCriteria: every LL, TERM and ZF cell within 5% of the simulated value; the log-log slope of LL against m between -0.56 and "
          "-0.42; ZF / TERM rising with m towards 2.\n")
L_.append("## B. Capped loss on empirical daily bars (m = 1)\n")
L_.append("| Market | L | P(ruin) | LL with the market's own overshoot constant | LL with the Gaussian constant |")
L_.append("|---|---|---|---|---|")
for r in B:
    L_.append(f"| {r['market']} | {r['L']} | {r['p_ruin']:.4f} | {100 * r['LL']:.3f} | {100 * r['LL_gaussian_constant']:.3f} |")
L_.append("\nCriteria: EURUSD within 8% of the simulated value; crypto within 25% (the barrier is only two to seven bars away and "
          "compounding is not small there). Disclosure: fee-free simulated values for these cells were already seen on 16 November "
          "(`../novelty_search_2026-11-16/overshoot_bgk_out.txt`, where the Gaussian constant gave ratios of 1.02 to 1.05 for EURUSD and "
          "0.82 to 1.33 for crypto). What is new here is the market's own overshoot constant. Section D1's XE number was also known. "
          "Sections A (finer time steps, TERM and ZF), C and D4 had not been computed in any form.\n")
L_.append("## C. Unmodified backtesting.py 0.6.6, fine Gaussian paths, m checks per day\n")
L_.append("| Fee per side | L | m | P(ruin) | Forgiven (% of equity) |")
L_.append("|---|---|---|---|---|")
for r in C:
    L_.append(f"| {1e4 * r['fee']:.0f} bp | {r['L']} | {r['m']} | {r['p_ruin']:.4f} | {100 * r['forgiven']:.3f} |")
L_.append("\nCriteria: the engine's forgiven amount (reported equity minus its own trade records) within 10% of the prediction or "
          "the prediction inside the 95% bootstrap interval, in every cell; ruin share within 0.03.\n")
L_.append("## D1. The registered Study XE census (2,000 paths, EURUSD magnitudes, 2 bp a side)\n")
L_.append("| L | P(ruin) | Forgiven (% of equity) |")
L_.append("|---|---|---|")
for r in D1:
    L_.append(f"| {r['L']} | {r['p_ruin']:.4f} | {100 * r['forgiven']:.3f} |")
L_.append("\nCriterion: within 10% at L = 50, 100 and 200, and within 25% at L = 20, where the amount is small.\n")
L_.append("## D4. The paper's Nordic ladder (24 zone-years, placebo signs on real hourly spreads, from zero equity, fee-free)\n")
L_.append("For fixed magnitudes m_i with fair signs, the law gives ZF / TERM = 2 - 1.2533 kappa, with kappa = sum |m_i|^3 / (sum m_i^2)^1.5 "
          "(the constant is sqrt(2 pi) / 2, from the fixed-size-move overshoot 1/2). kappa is computed from the spreads; the measured "
          "ratios are recomputed with 20,000 sign draws.\n")
L_.append("Criterion: Spearman rank correlation of at least 0.8 between predicted and measured ratios, and a mean absolute error of at "
          "most 0.05.\n")
(HERE / "PREDICTIONS.md").write_text("\n".join(L_), encoding="utf-8")
print("\n".join(L_))
