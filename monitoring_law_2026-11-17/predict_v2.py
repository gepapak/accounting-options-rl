"""Second set of predictions (17 Nov, after Tests A and B of PREDICTIONS.md were scored): writes predictions_v2.json and
PREDICTIONS_v2.md from inputs only.
  A2  the additive forms on ADDITIVE profit and loss (the paper's ledger model; never simulated before)
  A3  the compounded-price forms on the cells of Test A (post hoc for those cells: their simulated values are known)
  B3  the compounded-price forms on empirical daily bars (post hoc; simulated values known)
  C   unmodified backtesting.py at finer time steps (never run in any form): BLIND
  D1  the registered XE census with the compounded-price forms (its measured values were known)"""
import json
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

import law

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
P1 = json.loads((HERE / "predictions.json").read_text())
eur, cry = np.load(HERE / "eurusd_mags.npy"), np.load(HERE / "crypto_mags.npy")
s_e, s_c, N_D = P1["inputs"]["s_eurusd"], P1["inputs"]["s_crypto"], P1["inputs"]["n_days"]
rho_e, rho_c = P1["inputs"]["rho_over_s_eurusd"] * s_e, P1["inputs"]["rho_over_s_crypto"] * s_c
rl_e, n1 = law.rho_log_mc(eur)
rl_c, n2 = law.rho_log_mc(cry)
P = dict(inputs=dict(rho_log_over_s_eurusd=rl_e / s_e, rho_log_over_s_crypto=rl_c / s_c))
P["A2"] = P1["A"]                                                        # additive walk: the additive forms, unchanged
A3 = []
for L in (20, 50, 200):
    for m in (1, 4, 16, 64):
        A3.append(dict(L=L, m=m, **law.ln_forms(L, s_e / sqrt(m), N_D * m)))
P["A3"] = A3
B3 = []
for name, s, rho, levs in (("eurusd", s_e, rl_e, (20, 50, 100, 200)), ("crypto", s_c, rl_c, (2, 5, 10))):
    for L in levs:
        B3.append(dict(market=name, L=L, **law.ln_forms(L, s, N_D, rho=rho)))
P["B3"] = B3
C = []
for fee in (0.0, 0.0002):
    for L in (20, 50, 200):
        Leff = 0.9999 * L / (1.0 + fee)
        for m in (1, 4, 16):
            r = law.ln_forms(Leff, s_e / sqrt(m), N_D * m, fee=fee)
            C.append(dict(fee=fee, L=L, m=m, p_ruin=r["p_ruin"], forgiven=r["LL"]))
P["C"] = C
D1 = []
for L in (20, 50, 100, 200):
    r = law.ln_forms(0.9999 * L / 1.0002, s_e, N_D, rho=rl_e, fee=0.0002)
    D1.append(dict(L=L, p_ruin=r["p_ruin"], forgiven=r["LL"]))
P["D1"] = D1
(HERE / "predictions_v2.json").write_text(json.dumps(P, indent=1))

T = ["# Second set of predictions (compounded prices), written after Tests A and B were scored and before Tests A2, A3, C and D ran\n",
     f"Generated {datetime.now().isoformat(timespec='seconds')} by `predict_v2.py` from inputs only. Exploratory, not registered.\n",
     "Why a second set: in Test A the additive forms were 12% high at L = 20 in all three rules, TERM included, because the simulated "
     "prices compound. The time-step content passed (slopes -0.47 to -0.49; ZF / TERM within 1%). The forms below use the log price.\n",
     f"Overshoot constant of the log price, by far-barrier Monte Carlo: EURUSD {rl_e / s_e:.4f}, crypto {rl_c / s_c:.4f} "
     f"(simple returns: {rho_e / s_e:.4f} and {rho_c / s_c:.4f}).\n",
     "## C. Unmodified backtesting.py 0.6.6 at finer time steps (BLIND: never run in any form)\n",
     "| Fee per side | L | m | P(ruin) | Forgiven (% of equity) |", "|---|---|---|---|---|"]
for r in C:
    T.append(f"| {1e4 * r['fee']:.0f} bp | {r['L']} | {r['m']} | {r['p_ruin']:.4f} | {100 * r['forgiven']:.3f} |")
T.append("\nCriteria: in every cell, the engine's forgiven amount within 10% of the prediction, or the prediction inside the 95% "
         "bootstrap interval; ruin share within 0.03; fee-free log-log slope against m between -0.56 and -0.42.\n")
T.append("## A2. Additive profit and loss (the paper's ledger model), additive forms: the table of PREDICTIONS.md section A, unchanged\n")
T.append("Criterion: every LL, TERM and ZF cell within 5% of the simulated value.\n")
T.append("## A3 and B3. Compounded prices, the cells of Tests A and B (post hoc: their simulated values were known)\n")
T.append("| Case | L | m | P(ruin) | LL | TERM | ZF |")
T.append("|---|---|---|---|---|---|---|")
for r in A3:
    T.append(f"| Gaussian | {r['L']} | {r['m']} | {r['p_ruin']:.4f} | {100 * r['LL']:.3f} | {100 * r['TERM']:.3f} | {100 * r['ZF']:.3f} |")
for r in B3:
    T.append(f"| {r['market']} | {r['L']} | 1 | {r['p_ruin']:.4f} | {100 * r['LL']:.3f} | {100 * r['TERM']:.3f} | {100 * r['ZF']:.3f} |")
T.append("\nCriteria as before: 5% (Gaussian), 8% (EURUSD), 25% (crypto).\n")
T.append("## D1. The registered Study XE census, compounded-price forms with the fee term (measured values were known)\n")
T.append("| L | P(ruin) | Forgiven (% of equity) |")
T.append("|---|---|---|")
for r in D1:
    T.append(f"| {r['L']} | {r['p_ruin']:.4f} | {100 * r['forgiven']:.3f} |")
T.append("")
(HERE / "PREDICTIONS_v2.md").write_text("\n".join(T), encoding="utf-8")
print("\n".join(T))
