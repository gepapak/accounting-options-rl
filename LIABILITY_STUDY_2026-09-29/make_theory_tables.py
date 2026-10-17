"""LaTeX tables of the theory checks (Supplementary Section S11): tab_zones.tex, tab_cap.tex and tab_actiongap.tex.

Reads results_k/best_moment_zones.csv (best_moment_zones.py), results_k/false_profit_bound.csv (false_profit_bound.py)
and results_k/action_gap.csv (action_gap.py); writes outputs/tables/.

    python make_theory_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = HERE / "results_k"
OUT = HERE / "outputs" / "tables"
NL = " \\\\"
# What the equity-only PPO agents learned in each condition (VERDICTS_F.md, VERDICTS_J.md, VERDICTS_K.md, VERDICTS_H.md)
LEARNED = {("DK1", "gamma 0.99"): "yes", ("DK2", "gamma 0.99"): "yes", ("FI", "gamma 0.99"): "yes",
           ("NO2", "gamma 0.99"): "no effect", ("DK1", "Study H costs"): "no", ("DK2", "Study H costs"): "no"}


def signed(x: float, d: int = 1) -> str:
    return f"${x:.{d}f}$".replace("$-0.0$", "$0.0$")


def zones() -> str:
    d = pd.read_csv(R / "best_moment_zones.csv")
    rows = [f"{r.zone} & {r.mean_abs_spread:.1f} & {r.excess_kurtosis:,.0f} & {r.ks_D:.3f} & {r.mean_reported_K:.2f} & "
            f"{r.predicted_K:.2f} & ${r.error_pct:.0f}$ & {r.cv:.2f} & {r.power_n5:.2f}{NL}" for r in d.itertuples()]
    return ("\\begin{table}[!htbp]\n\\centering\n"
            "\\caption{Theorem~\\ref{thm:best} in the twelve Nordic bidding zones: a fixed position of 4~MWh held from zero equity under the "
            "zero floor in the placebo market (eSett imbalance-minus-spot spread, January--August 2026, fee 0.10~EUR/MWh, "
            "20,000 sign draws, no liquidity cap). KS: Kolmogorov--Smirnov distance between reported equity and the running "
            "maximum of the booked P\\&L. Predicted: the law of the maximum of a Brownian motion with the fee drift, no fitted "
            "parameter. CV: coefficient of variation (diffusion limit 0.756). Power: share of five-draw placebo tests that "
            "flag the floor (diffusion limit 0.90).}\n\\label{tab:zones}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
            "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
            f"Zone & Mean $|$spread$|$ & Excess kurtosis & KS & Reported & Predicted & Error & CV & Power{NL}\n"
            f" & (EUR/MWh) & & & (alloc.) & (alloc.) & (\\%) & &{NL}\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def cap() -> str:
    d = pd.read_csv(R / "false_profit_bound.csv")
    f = lambda x: "--" if np.isnan(x) else f"{x:.2f}"  # noqa: E731
    rows = [f"{r.zone} & {r.cap_B_K:.2f} & {r.zero_fixed_share:.2f} & {r.zero_dp_share:.2f} & {r.one_fixed_K:.2f} & "
            f"{r.one_dp_K:.2f} & {f(r.one_closed_form_K)} & {r.one_learned_K:.2f}{NL}" for r in d.itertuples()]
    return ("\\begin{table}[!htbp]\n\\centering\n"
            "\\caption{The universal cap of Theorem~\\ref{thm:cap} in the four zones of the learning studies (placebo market, test period, "
            "liquidity cap applied, fee 0.10~EUR/MWh, 20,000 sign draws shared by all policies; allocations of 20,000~EUR). "
            "Cap: $B_T$. From zero equity: expected reported equity of the fixed maximal position and of the optimal threshold "
            "policy (dynamic program on the full training distribution, $\\gamma = 0.99$), as a share of the cap. From one "
            "allocation: expected reported profit of the fixed maximal position, of the threshold policies with the "
            "dynamic-program barrier and the closed-form barrier of Proposition~\\ref{prop:barrier}, and of the learned floor-trained agents "
            "with equity as their only input (Studies F, J and K at $\\gamma = 0.99$).}\n\\label{tab:cap}\n\\scriptsize\n"
            "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
            f" & & \\multicolumn{{2}}{{c}}{{From zero (share of cap)}} & \\multicolumn{{4}}{{c}}{{From one allocation (allocations)}}{NL}\n"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-8}\n"
            f"Zone & Cap & Fixed & Threshold & Fixed & Threshold & Threshold & Learned{NL}\n"
            f" & $B_T$ & maximal & (DP) & maximal & (DP) & (closed form) & agents{NL}\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def actiongap() -> str:
    d = pd.read_csv(R / "action_gap.csv")
    probes = ("0", "0.02", "0.1", "0.25", "0.5", "1", "1.5", "2")
    rows = []
    for r in d.to_dict("records"):
        cond = "Study H costs" if r["costs"] == "H" else f"$\\gamma = {r['gamma']:g}$"
        learned = LEARNED.get((r["zone"], r["condition"]), "size only")
        rows.append(f"{r['zone']} & {cond} & {r['sigma16']:.0f} & " + " & ".join(signed(r[f'A_{p}']) for p in probes)
                    + f" & {learned}{NL}")
    return ("\\begin{table}[!htbp]\n\\centering\n"
            "\\caption{Action gap of the dynamic program under the zero floor, $A(e) = Q(e, 16) - Q(e, 0)$, in euros of "
            "discounted reward, at probed equity levels (allocations), for every condition in which floor-trained agents were "
            "trained, on the full training distribution. $\\sigma_{16}$: standard deviation of the one-quarter-hour P\\&L at "
            "leverage~16 (EUR), the reward noise. Learned: whether equity-only PPO agents learned the gambling region's "
            "threshold shape (Studies F, J, K; at $\\gamma \\ne 0.99$ they matched the region's size but not its trend) or any "
            "gambling at all under the costs of Study H.}\n\\label{tab:actiongap}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n"
            "\\begin{tabular}{@{}llrrrrrrrrrl@{}}\n\\toprule\n"
            f"Zone & Condition & $\\sigma_{{16}}$ & 0 & 0.02 & 0.1 & 0.25 & 0.5 & 1 & 1.5 & 2 & Learned{NL}\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("tab_zones", zones), ("tab_cap", cap), ("tab_actiongap", actiongap)):
        (OUT / f"{name}.tex").write_text(make(), encoding="utf-8")
        print("wrote", OUT / f"{name}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
