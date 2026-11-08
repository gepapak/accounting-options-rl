"""Figure: the reward diagram, on real clones of the engine and on the public simulator.

    python make_fig_diagram.py --extract   # from the campaign's scans (outside the repository) and the public simulator: writes fig_diagram_data.json
    python make_fig_diagram.py             # from fig_diagram_data.json: writes fig_diagram.pdf (and .png) here and copies the PDF to the manuscript if it is next to this repository

(a) One funded step of the audit's volume-payoff protocol. (b) The percentage payoff with its cap on and off, read from the reward alone: median over the sampled moved steps of one run, normalized by the
realized gross. (c) A step of Gym-Trading-Env at position 10: the reward follows the wealth until the wealth reaches zero, then it is replaced by 0.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCANS = REPO.parents[1] / "engine_clone_check_2026-11-06" / "scan_campaign" / "scans"
DATA = HERE / "fig_diagram_data.json"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load_scan(name):
    return dict(np.load(SCANS / f"{name}.npz", allow_pickle=False))


def extract():
    out = {}
    # (a) a funded step whose floor lies within view
    name_a = "L7_minus_percent_mtm__marl__seed7__original"
    z = load_scan(name_a)
    fields = [str(f) for f in z["fields"]]
    grid = z["grid"]
    order = np.argsort(grid)
    ts = grid[order]
    phi = z["d_line"][:, :, fields.index("r")][:, order]
    cols = [str(c) for c in z["real_columns"]]
    real = {c: z["d_real"][:, i] for i, c in enumerate(cols)}
    C = real["access"] + real["roll"] + real["transaction"] + real["funding"] - real["battery"]
    cands = []
    for k in range(len(phi)):
        cash, g = z["d_cash_pre"][k], real["gross"][k]
        if cash < 1e7 or abs(z["d_mtm_pre"][k]) > 1.0 or g <= 0.0:
            continue
        tk = (C[k] - cash) / g                                       # where the unfavourable branch reaches the floor
        if -3.5 < tk < -1.3 and abs(phi[k][np.argmin(np.abs(ts + 8.0))] + cash) < 2000.0:
            cands.append((cash, k))
    cands.sort()
    k = cands[len(cands) // 2][1]
    sel = np.abs(ts) <= 8.0
    out["a"] = dict(run=name_a, step=int(z["d_t"][k]), t=ts[sel].tolist(), phi=phi[k][sel].tolist(), cash=float(z["d_cash_pre"][k]), gross=float(real["gross"][k]), cost=float(C[k]), candidates=len(cands))

    # (b) cap on and off: median normalized diagram over the sampled moved steps
    def normalized(name):
        z = load_scan(name)
        fields = [str(f) for f in z["fields"]]
        grid = z["grid"]
        order = np.argsort(grid)
        ts = grid[order]
        phi = z["d_line"][:, :, fields.index("r")][:, order]
        i0, i1 = int(np.argmin(np.abs(ts))), int(np.argmin(np.abs(ts - 1.0)))
        funded = z["d_cash_pre"] > 1e7
        chord = phi[:, i1] - phi[:, i0]
        ok = funded & (np.abs(chord) > 1.0)
        norm = (phi[ok] - phi[ok, i0][:, None]) / chord[ok][:, None]
        sel = np.abs(ts) <= 8.0
        return dict(run=name, steps=int(ok.sum()), t=ts[sel].tolist(), median=np.median(norm[:, sel], axis=0).tolist(), q25=np.quantile(norm[:, sel], 0.25, axis=0).tolist(), q75=np.quantile(norm[:, sel], 0.75, axis=0).tolist())

    out["b_on"] = normalized("L7_thesis_like_capon__marl__seed7__original")
    out["b_off"] = normalized("L7_thesis_like__marl__seed7__original")

    # (c) the public simulator at position 10
    sys.path.insert(0, str(REPO / "ACCOUNTING_OPTIONS_2026-10-14" / "study_ext"))
    from envs_ext import POSITIONS, ExtEnv
    env = ExtEnv("test", "placebo", "REP", seed=11)
    a = POSITIONS.index(10)
    env.reset()
    # hold the position for a few steps, so that the holdings are the engine's own, then read one step
    for _ in range(3):
        env.step(a)
    p = env.core._price_array
    move = float(p[env.core._idx + 1] / p[env.core._idx] - 1.0)
    ts_c = np.linspace(-1.0, 1.0, 801) * (1.6 / abs(move) / 10.0)
    ph = [env.query(a, float(t)) for t in ts_c]
    out["c"] = dict(t=ts_c.tolist(), phi=ph, wealth=env.wealth, move=move, position=10)
    DATA.write_text(json.dumps(out), encoding="utf-8")
    print({k: {kk: vv for kk, vv in v.items() if kk not in ("t", "phi", "median", "q25", "q75")} for k, v in out.items()})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.75)
    ax.tick_params(colors=INK2, length=2.5, width=0.75, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42, "axes.titlesize": 9, "axes.titleweight": "normal", "axes.labelcolor": INK2, "axes.labelsize": 8})
    d = json.loads(DATA.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45), constrained_layout=True)

    # (a)
    ax = axes[0]
    a = d["a"]
    t, phi = np.array(a["t"]), np.array(a["phi"]) / 1e6
    cash, g, c = a["cash"] / 1e6, a["gross"] / 1e6, a["cost"] / 1e6
    tk = (a["cost"] - a["cash"]) / a["gross"]
    tt = np.sort(np.r_[t, tk])
    yy = np.interp(tt, np.r_[t[t < tk], tk, t[t > tk]], np.r_[phi[t < tk], -cash, phi[t > tk]])
    ax.plot(tt, yy, color=BLUE, lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    for tx in (1.0, -1.0, 0.0):
        y = float(phi[np.argmin(np.abs(t - tx))])
        ax.plot([tx], [y], "o", ms=5, color=BLUE, mec="white", mew=1.0, zorder=4)
    ax.set_xlim(-8, 8)
    ax.set_xlabel("$t$, multiple of the realized move")
    ax.set_ylabel("Reported reward (million DKK)")
    ax.set_title("(a) A funded step in the engine", color=INK, loc="left")
    style(ax)
    ax.text(-7.6, -cash + 0.055 * (phi.max() + cash), "floor at minus\nthe cash", fontsize=7.5, color=INK2, va="bottom")
    ax.annotate("slope = gross exposure", xy=(4.0, float(phi[np.argmin(np.abs(t - 4.0))])), xytext=(-6.6, phi.max() * 0.62), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=2))
    ax.annotate("$t=1$: the real step", xy=(1.0, float(phi[np.argmin(np.abs(t - 1.0))])), xytext=(1.5, phi.min() * 0.40), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=2))

    # (b)
    ax = axes[1]
    on, off = d["b_on"], d["b_off"]
    for s, color, label in ((off, BLUE, "cap off, as run"), (on, ORANGE, "cap on")):
        tb, m = np.array(s["t"]), np.array(s["median"])
        ax.plot(tb, m, color=color, lw=1.5, solid_joinstyle="round", solid_capstyle="round", label=label, zorder=3)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8.5, 8.5)
    ax.set_xlabel("$t$, multiple of the realized move")
    ax.set_ylabel("Reward change, per realized gross")
    ax.set_title("(b) A cap, read from the reward", color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2, handlelength=1.6)

    # (c)
    ax = axes[2]
    c_ = d["c"]
    tc, pc = np.array(c_["t"]), np.array(c_["phi"])
    # break the line at the jump, so that the jump is not drawn as a mark
    jumps = np.flatnonzero(np.abs(np.diff(pc)) > 5.0 * np.median(np.abs(np.diff(pc))))
    pc_plot = pc.copy()
    tc_plot = tc.copy()
    for j in jumps[::-1]:
        tc_plot = np.insert(tc_plot, j + 1, np.nan)
        pc_plot = np.insert(pc_plot, j + 1, np.nan)
    ax.plot(tc_plot, pc_plot, color=BLUE, lw=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    if len(jumps):
        j = jumps[0]
        ax.annotate("bust: the reward is\nreplaced by 0;\nthe jump is the wealth", xy=(tc[j], 0.5 * (pc[j] + pc[j + 1])), xytext=(tc[0] * 0.98, pc.max() * 0.45), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=2))
    y1 = float(np.interp(1.0, tc, pc))
    ax.plot([1.0], [y1], "o", ms=5, color=BLUE, mec="white", mew=1.0, zorder=4)
    ax.set_xlabel("$t$, multiple of the realized move")
    ax.set_ylabel("Reward (% of initial wealth)")
    ax.set_title("(c) A public simulator", color=INK, loc="left")
    style(ax)

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_diagram.{ext}", dpi=200)
    target = REPO.parent / "accounting-options-rl-latex-latex" / "figures"
    if target.is_dir():
        shutil.copy(HERE / "fig_diagram.pdf", target / "fig_diagram.pdf")
    print("written", HERE / "fig_diagram.pdf")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        extract()
    main()
