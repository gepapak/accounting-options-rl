"""Figure for Study P (exploratory output script, not registered): fig_studyP.pdf/png.

Share of runs choosing leverage 16 at 2% equity against the learnability index, by training budget (250k, 10^6 from
Study O, 4M), with Wilson 95% intervals and the joint logistic model of aggregate_p.py: c50(N) = c0 (N / 10^6)^beta.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aggregate_p import CONTROL, fit, load

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {250_000: "#eb6834", 1_000_000: "#2a78d6", 4_000_000: "#1baf7a"}
LAB = {250_000: "250k steps (Study P)", 1_000_000: "1M steps (Study O)", 4_000_000: "4M steps (Study P)"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main() -> int:
    R = load()
    T = R[R.alpha != CONTROL]
    s, lc0, beta = fit(T)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    xs = np.logspace(np.log10(0.04), np.log10(2.5), 200)
    offs = {250_000: 0.97, 1_000_000: 1.0, 4_000_000: 1.03}
    for n in (250_000, 1_000_000, 4_000_000):
        lc = lc0 + beta * np.log10(n / 1e6)
        ax.plot(xs, 1 / (1 + np.exp(-s * (np.log10(xs) - lc))), color=COL[n], lw=1.2)
        ax.axvline(10 ** lc, color=COL[n], lw=0.6, ls=":")
        for a, g in T[T.budget == n].groupby("alpha"):
            k, m = int((g.L002 == 16).sum()), len(g)
            lo, hi = wilson(k, m)
            x = g["index"].mean() * offs[n]
            ax.errorbar(x, k / m, yerr=[[k / m - lo], [hi - k / m]], fmt="o", color=COL[n], ms=4, lw=0.8, capsize=0, zorder=3)
        ax.plot([], [], "o-", color=COL[n], ms=4, lw=1.2, label=f"{LAB[n]}: c$_{{50}}$ = {10 ** lc:.2f}%")
    ax.set_xscale("log"); ax.set_xlim(0.04, 2.5); ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Learnability index (%), computed before training")
    ax.set_ylabel("Share of runs at leverage 16, 2% equity")
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title(f"Threshold falls with training: c$_{{50}}$ $\\propto$ N$^{{{beta:.2f}}}$ (law: N$^{{-0.5}}$)", loc="left", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(HERE / "fig_studyP.pdf"); fig.savefig(HERE / "fig_studyP.png", dpi=180)
    print(f"s {s:.2f}, c50(1M) {10 ** lc0:.3f}%, beta {beta:.3f}; c50(250k) {10 ** (lc0 + beta * np.log10(0.25)):.3f}%, c50(4M) {10 ** (lc0 + beta * np.log10(4)):.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
