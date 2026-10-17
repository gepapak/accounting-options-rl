"""Figure for Study O (exploratory output script, not registered): fig_studyO.pdf/png.

(a) Share of runs choosing leverage 16 at 2% equity against the learnability index: the calibration conditions of
    Studies F, H, J, K (open markers) and Study O's prospective arms (filled), with Wilson 95% intervals, the registered
    threshold c* and the logistic fit of Study O.
(b) Gambling region by arm: the DP optimum (leverage 16 optimal below it) against the learned region of the agents
    (Study K's measure), mean of ten runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).resolve().parent
LIAB = HERE.parents[1] / "LIABILITY_STUDY_2026-09-29"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def calibration_points() -> pd.DataFrame:
    cal = json.loads((HERE / "calibration_o.json").read_text())
    F = pd.read_csv(LIAB / "results_f/runs_f.csv"); K = pd.read_csv(LIAB / "results_k/runs_k.csv")
    J = pd.read_csv(LIAB / "results_j/runs_j.csv"); H = pd.read_csv(LIAB / "results_h/runs_h.csv")
    rows = []
    for r in cal["rows"]:
        z, c = r["zone"], r["condition"]
        if c == "Study H costs":
            g = H[(H.region == z) & (H.obsset == "EQ") & (H.rule == "ZF")]
        elif z in ("FI", "NO2"):
            g = J[(J.region == z) & (J.obsset == "EQ") & (J.rule == "ZF")]
        elif c == "gamma 0.99":
            g = pd.concat([F[(F.region == z) & (F.obsset == "EQ") & (F.rule == "ZF")], K[(K.region == z) & (K.gamma == 0.99) & (K.rule == "ZF")]])
        else:
            g = K[(K.region == z) & (K.gamma == float(c.split()[1])) & (K.rule == "ZF")]
        rows.append(dict(label=f"{z} {c}", index=r["index_pct"], k=int((g["L0.02"] == 16).sum()), n=len(g)))
    return pd.DataFrame(rows)


def main() -> int:
    R = pd.read_csv(HERE / "runs_o.csv")
    cal = json.loads((HERE / "calibration_o.json").read_text())
    T = R[R.alpha != 1.0]
    x, y = np.log10(T["index"].to_numpy()), (T.L002 == 16).to_numpy(float)
    nll = lambda p: -np.sum(y * p[0] * (x - p[1]) - np.logaddexp(0, p[0] * (x - p[1])))  # noqa: E731
    s, lc = optimize.minimize(nll, [5.0, np.log10(0.4)], method="Nelder-Mead").x
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1]))
    xs = np.logspace(np.log10(0.04), np.log10(2.5), 200)
    a.plot(xs, 1 / (1 + np.exp(-s * (np.log10(xs) - lc))), color=BLUE, lw=1.2, label=f"Fit, O (c$_{{50}}$ = {10 ** lc:.2f}%)")
    a.axvline(cal["c_star_pct"], color=MUTED, lw=0.8, ls="--")
    a.text(0.045, 0.02, f"c* = {cal['c_star_pct']:.2f}%", color=MUTED, fontsize=9.5)
    C = calibration_points()
    for r in C.itertuples():
        lo, hi = wilson(r.k, r.n)
        a.errorbar(r.index, r.k / r.n, yerr=[[r.k / r.n - lo], [hi - r.k / r.n]], fmt="o", mfc="white", mec=ORANGE, ecolor=ORANGE,
                   ms=4, lw=0.7, capsize=0, zorder=3)
    for al, g in T.groupby("alpha"):
        k, n = int((g.L002 == 16).sum()), len(g)
        lo, hi = wilson(k, n)
        a.errorbar(g["index"].mean(), k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", color=BLUE, ms=4.5, lw=0.8, capsize=0, zorder=4)
        label_pos = {0.0: (0.82, 0.74), 0.25: (0.65, 0.91), 0.5: (0.56, 0.70),
                     0.7: (0.48, 0.32), 0.8: (0.24, 0.47), 0.9: (0.31, 0.10), 0.95: (0.03, 0.20)}[float(al)]
        a.annotate(f"$\\alpha$={al:g}", (g["index"].mean(), k / n), textcoords="axes fraction", xytext=label_pos,
                   fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    a.plot([], [], "o", color=BLUE, ms=4.5, label="Prospective arms")
    a.plot([], [], "o", mfc="white", mec=ORANGE, ms=4, label="Calibration (F, H, J, K)")
    a.set_xscale("log"); a.set_xlim(0.04, 2.5); a.set_ylim(-0.04, 1.06)
    a.set_xlabel("Pre-training index (%)")
    a.set_ylabel("Runs at leverage 16 (share)")
    a.grid(True, color=GRID, lw=0.5); a.legend(loc="upper left", fontsize=9.5)
    a.set_title("(a) Prospective arms and calibration", loc="left", fontsize=10, color=INK)
    arms = R.groupby("alpha").agg(dp=("dp_region", "mean"), T=("T", "mean")).reset_index()
    xi = np.arange(len(arms))
    b.bar(xi - 0.2, arms.dp, 0.38, color=GRID, edgecolor=MUTED, lw=0.6, label="DP optimum")
    b.bar(xi + 0.2, arms["T"], 0.38, color=BLUE, label="Learned mean")
    b.set_xticks(xi); b.set_xticklabels([f"{v:g}" for v in arms.alpha], rotation=45, ha="right")
    b.set_xlabel("Loss share booked, $\\alpha$")
    b.set_ylabel("Gambling region (allocations)")
    b.grid(True, axis="y", color=GRID, lw=0.5); b.legend(loc="upper right", fontsize=9.5)
    b.set_title("(b) Gambling regions", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_studyO.pdf"); fig.savefig(HERE / "fig_studyO.png", dpi=180)
    print(f"logistic: slope {s:.2f}, c50 {10 ** lc:.3f}%; calibration points:\n{C.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
