"""Study W's ruin-penalty cells under the common removal criterion of Studies S to Y (exploratory; written after the fourth AI-assisted review).

    python w_common_criterion.py      # writes w_common_criterion.md and ../../accounting-options-rl-latex-latex/sections/tab_w_common.tex

Study W's registered W2 judges a strength by arm means (mean X at most ORACLE + 0.10, mean E at least 0.80 of ORACLE). The common definition of
Studies S to Y also requires X(arm) < X(PLAIN) by a one-sided Mann-Whitney test (p < 0.05) and, for keeping the edge, a one-sided 95% bootstrap
lower bound of the edge ratio of at least 0.60 (10,000 unpaired resamples, as in aggregate_x.py). The references are Study T's reported-reward
(PLAIN) and ledger-trained (ORACLE) runs in the same settings, as in Study W.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ACC = HERE.parent / "ACCOUNTING_OPTIONS_2026-10-14"
TEX = HERE.parents[1] / "accounting-options-rl-latex-latex" / "sections" / "tab_w_common.tex"
rng = np.random.default_rng(20261103)
LABEL = {"T_DK2": "DK2", "T_DK1M": "DK1, weaker edge"}


def main():
    w = pd.read_csv(ACC / "study_w" / "runs_w.csv")
    t = pd.read_csv(ACC / "study_t" / "runs_t.csv")
    rows = []
    for setting in ("T_DK2", "T_DK1M"):
        P = t[(t.setting == setting) & (t.arm == "PLAIN")]
        O = t[(t.setting == setting) & (t.arm == "ORACLE")]
        for arm in ("RUIN1", "RUIN5", "RUIN25"):
            A = w[(w.setting == setting) & (w.arm == arm)]
            p = float(stats.mannwhitneyu(A.X, P.X, alternative="less").pvalue)
            ratio = A.E.mean() / O.E.mean()
            boot = np.array([rng.choice(A.E.to_numpy(), len(A)).mean() / rng.choice(O.E.to_numpy(), len(O)).mean() for _ in range(10_000)])
            lb = float(np.quantile(boot, 0.05))
            mean_remove = A.X.mean() <= O.X.mean() + 0.10
            mean_keep = A.E.mean() >= 0.80 * O.E.mean()
            rows.append(dict(setting=setting, strength=arm, runs=len(A), X=A.X.mean(), p=p, ratio=ratio, lb=lb,
                             w2_both=bool(mean_remove and mean_keep),
                             common_removes=bool(p < 0.05 and mean_remove), common_keeps=bool(ratio >= 0.80 and lb >= 0.60)))
    df = pd.DataFrame(rows)
    df["common_both"] = df.common_removes & df.common_keeps
    md = ["# Study W cells under the common removal criterion (exploratory)", "",
          "| setting | strength | runs | mean X | Mann-Whitney p vs Study T PLAIN | edge ratio | bootstrap LB | W2 (means) both | common: removes | common: keeps | common: both |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        md.append(f"| {r.setting} | {r.strength} | {r.runs} | {r.X:.3f} | {r.p:.4f} | {r.ratio:.3f} | {r.lb:.3f} | {r.w2_both} | {r.common_removes} | {r.common_keeps} | {r.common_both} |")
    (HERE / "w_common_criterion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    yes = lambda b: "yes" if b else "no"  # noqa: E731
    tex = ["\\begin{tabular}{@{}llrrrrcc@{}}", "\\toprule",
           " & & & & & & \\multicolumn{2}{c}{Does both} \\\\", "\\cmidrule(lr){7-8}",
           "Setting & Strength & $\\bar X$ & $p$ & $E$ ratio & LB & W2 & Common \\\\", "\\midrule"]
    for r in df.itertuples():
        tex.append(f"{LABEL[r.setting]} & {r.strength[4:]} & {r.X:.3f} & {r.p:.3f} & {r.ratio:.2f} & {r.lb:.2f} & {yes(r.w2_both)} & {yes(r.common_both)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("wrote", TEX)


if __name__ == "__main__":
    main()
