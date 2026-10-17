"""LaTeX tables of the settlement analysis (main Table 12 and Supplementary Section S15), from the CSV outputs here.
Writes tables/tab_ladder12.tex, tab_tails12.tex, tab_frontier12.tex and tab_dkregimes.tex."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
ZONES = ["DK1", "DK2", "FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5"]
NL = " \\\\"


def table(caption, label, spec, head, rows):
    return ("\\begin{table}[!htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\scriptsize\n\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            + head + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def mc(n, text, col="c"):
    return f"\\multicolumn{{{n}}}{{{col}}}{{{text}}}"


# Main Table 12: the ladder in twelve zones (ladder12.py)
d = pd.read_csv(HERE / "ladder12.csv").set_index(["zone", "period"])
rows = []
for z in ZONES:
    a, b = d.loc[(z, "2024")], d.loc[(z, "post-reform year")]
    rows.append(f"{z} & {a.ZF / 1e3:.1f} & {b.ZF / 1e3:.1f} & {b.ZF / a.ZF:.1f} & {a.TERM / 1e3:.1f} & {b.TERM / 1e3:.1f}"
                f" & {a.ZF_over_TERM_feefree:.2f} & {b.ZF_over_TERM_feefree:.2f}{NL}")
head = (f"& {mc(3, 'Zero floor')} & {mc(2, 'Terminal limited liability')} & {mc(2, 'Ratio, no fee')}{NL}\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n"
        f"Zone & 2024 & Post-reform & Factor & 2024 & Post-reform & 2024 & Post-reform{NL}\n")
(OUT / "tab_ladder12.tex").write_text(table(
    "The ladder in the twelve Nordic zones: forgiven loss per MW-year (thousand EUR) of a constant 1~MW exposure from "
    "zero equity, placebo signs on hourly eSett spreads, fee 0.10~EUR/MWh, in 2024 and in the first year after the mFRR "
    "go-live (4 March 2025 to 3 March 2026), 2,000 sign draws. Ratio: zero floor over terminal limited liability without "
    "fees (Theorem~\\ref{thm:factortwo}: between 1 and 2).",
    "tab:ladder12", "@{}lrrrrrrr@{}", head, rows), encoding="utf-8")

# S15: spread distribution before and after the go-live (tails12.py)
t = pd.read_csv(HERE / "tails12.csv")
rows = []
for z in ZONES:
    a = t[(t.zone == z) & (t.period != "post-reform")].iloc[0]
    b = t[(t.zone == z) & (t.period == "post-reform")].iloc[0]
    rows.append(f"{z} & {a.median_abs:.1f} & {b.median_abs:.1f} & {a.sd:.0f} & {b.sd:.0f} & {a.per10k_over_1000:.1f}"
                f" & {b.per10k_over_1000:.1f} & {a.max_abs:,.0f} & {b.max_abs:,.0f}{NL}")
head = (f"& {mc(2, 'Median $|s|$')} & {mc(2, 'Std.~dev.')} & {mc(2, 'Spikes')} & {mc(2, 'Max $|s|$')}{NL}\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
        f"Zone & before & after & before & after & before & after & before & after{NL}\n")
(OUT / "tab_tails12.tex").write_text(table(
    "Imbalance-minus-spot spreads $s$ per quarter-hour (EUR/MWh) before (1 January 2023 to 3 March 2025) and after "
    "(4 March 2025 to 27 September 2026) the mFRR EAM go-live, eSett Open Data. Spikes: quarter-hours with "
    "$|s|>1{,}000$~EUR/MWh, per 10,000.",
    "tab:tails12", "@{}lrrrrrrrr@{}", head, rows), encoding="utf-8")

# S15: level- versus volatility-indexed collateral (coverage12.py)
rows = []
for fee in (0.1, 0.5):
    f = pd.read_csv(HERE / f"frontier12_fee{fee}.csv")
    f = f[f.zones == "FI+SE+NO"]
    for rule in ["LEVEL (eSett volume term)", "VOL x1", "VOL x1.5", "VOL x2", "VOL x3"]:
        a = f[(f.rule == rule) & (~f.post)].iloc[0]
        b = f[(f.rule == rule) & (f.post)].iloc[0]
        name = "eSett volume term" if rule.startswith("LEVEL") else f"${rule.split('x')[1]}\\,\\hat\\kappa$"
        rows.append(f"{fee:.2f} & {name} & {100 * a.fail:.1f} & {100 * b.fail:.1f} & {a.mean_collateral:,.0f}"
                    f" & {b.mean_collateral:,.0f}{NL}")
nd = mc(2, "Not deterred (\\%)")
head = (f"Cost of risk & Rule & {nd} & {mc(2, 'Mean collateral (EUR/MW)')}{NL}\n"
        "\\cmidrule(lr){3-4}\\cmidrule(l){5-6}\n"
        f"(EUR/MWh) & & before & after & before & after{NL}\n")
(OUT / "tab_frontier12.tex").write_text(table(
    "Level-indexed versus volatility-indexed collateral in Finland, Sweden and Norway (FI, SE1--SE4, NO1--NO5), "
    "zone-months before and after the mFRR EAM go-live. Not deterred: collateral per MW below the buffer $\\kappa$ that "
    "deters a three-week limited-liability gamble. $\\hat\\kappa$: the buffer estimated from the three preceding months.",
    "tab:frontier12", "@{}llrrrr@{}", head, rows), encoding="utf-8")

# S15: Danish regimes, Energinet data (market_ladder.py, coverage.py)
REG = ("R1", "R2", "R3", "R4")  # regime labels of build_panel.py; dates in the caption
ml = pd.read_csv(HERE / "market_ladder.csv")
ml = ml[ml.start == "e=kappa3w"]
cov = pd.read_csv(HERE / "coverage_monthly.csv").replace(np.inf, np.nan)
rows = []
for z in ("DK1", "DK2"):
    for r in REG:
        m = ml[(ml.zone == z) & ml.regime.str.startswith(r)].iloc[0]
        c = cov[(cov.zone == z) & cov.regime.str.startswith(r)]
        rows.append(f"{z} & {r} & {m.sigma_year_eur_per_MW / 1e3:.1f} & {m.kappa3w_fee0p1:,.0f}"
                    f" & {m.DK_cap_MW_DKK2M:.0f} & {c['cov_DK_C1_0.1'].median():.2f}"
                    f" & {100 * (c['cov_DK_C1_0.1'] < 1).mean():.0f}{NL}")
    if z == "DK1":
        rows.append("\\addlinespace")
head = (f"& & $\\sigma$ & $\\kappa$ & DKK 2M cap & {mc(2, 'C1 consumption rule')}{NL}\n"
        "\\cmidrule(l){6-7}\n"
        f"Zone & Regime & (kEUR/MW) & (EUR/MW) & (MW) & coverage & not deterred (\\%){NL}\n")
(OUT / "tab_dkregimes.tex").write_text(table(
    "Danish rules across the four settlement regimes, Energinet data, hourly spreads, cost of risk 0.10~EUR/MWh. "
    "R1: two-price settlement, 2019-11 to 2021-10; R2: energy crisis, 2021-11 to 2022-12; R3: single price, hourly, "
    "2023-01 to 2025-03; R4: quarter-hourly mFRR EAM, 2025-03 to 2026-08. $\\sigma$: "
    "standard deviation of the yearly placebo P\\&L of 1~MW (thousand EUR). $\\kappa$: buffer that deters a three-week "
    "gamble, pooled over the regime. DKK 2M cap: open exposure up to which the fixed trading guarantee deters it. C1: consumption rule "
    "(5\\% of a month's consumption at the forward price) per MW, median monthly coverage and share of months below one.",
    "tab:dkregimes", "@{}llrrrrr@{}", head, rows), encoding="utf-8")
for p in sorted(OUT.glob("*.tex")):
    print(p.name, len(p.read_text(encoding="utf-8").splitlines()), "lines")
