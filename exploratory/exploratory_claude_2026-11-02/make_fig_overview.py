"""Overview figure of the paper (fig_overview.pdf): simulator accounting -> censored reward -> learned exploit -> recovery by access.

    python make_fig_overview.py [output_dir]      # default: the figures/ folder of the manuscript repository next to this repository
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "accounting-options-rl-latex-latex" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "mathtext.default": "regular"})

GREY, RED, BLUE, GREEN, AMBER, INK, SOFT = "#eef0f2", "#f6d9d6", "#dbe8f5", "#dff0df", "#fbefcf", "#1d2530", "#44505c"
EDGE = "#56606b"
BODY = 5.7


def box(ax, x, y, w, h, face, edge=EDGE, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.012", fc=face, ec=edge, lw=lw))


def text(ax, x, y, s, size=BODY, weight="normal", ha="left", va="top", color=INK, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, fontstyle=style, linespacing=1.3)


def arrow(ax, p, q, lw=1.2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=EDGE))


fig, ax = plt.subplots(figsize=(6.4, 4.1))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# column 1: the simulator
box(ax, 0.004, 0.37, 0.30, 0.60, GREY)
text(ax, 0.018, 0.955, "1  Simulator accounting", weight="bold", size=6.8)
text(ax, 0.018, 0.865, r"booked profit   $x_t=\sigma_t g_t-C_t$")
text(ax, 0.018, 0.775, "rule ladder (Theorem 2):\nFL $\\leq$ LL $\\leq$ LAG $\\leq$ TERM $\\leq$ ZF")
text(ax, 0.018, 0.64, "reported reward", size=BODY)
text(ax, 0.018, 0.585, r"$r_t=\max(x_t,\tau_t)$", size=8, weight="bold")
text(ax, 0.018, 0.495, "$\\tau_t$: equity (floor) or carried\ndeficit (lag, terminal); the credit\n$\\tau_t-x_t\\geq0$ is rewarded, never booked", size=5.4, color=SOFT)

# column 2: the learner and the evaluation
box(ax, 0.345, 0.37, 0.30, 0.60, RED)
text(ax, 0.359, 0.955, "2  Learned exploitation", weight="bold", size=6.8)
text(ax, 0.359, 0.865, "PPO, A2C, DQN trained on $r_t$\ngamble at low equity (10 of 10\nfloor agents at maximal leverage)")
text(ax, 0.359, 0.70, "placebo market (no edge):")
text(ax, 0.359, 0.645, "reported  +38 to +148%", weight="bold", color="#9b2c20")
text(ax, 0.359, 0.59, "booked    $-$15 to $-$57%", weight="bold", color="#9b2c20")
text(ax, 0.359, 0.54, "engine shortcuts, six runs:\nreported +162 to +795%,\nbooked $-$1,822 to $-$3,655%", size=5.0, color=SOFT)
text(ax, 0.359, 0.425, "a diagnostic orders acquisition\n(Studies O to Q)", size=5.4, color=SOFT)
arrow(ax, (0.307, 0.72), (0.343, 0.72))
text(ax, 0.325, 0.76, "$r_t$", size=7, ha="center", style="italic")

# column 3: recovery by access
text(ax, 0.675, 0.955, "3  What each access recovers", weight="bold", size=6.8)
rows = [
    (0.806, GREEN, "ledger, or the payoff formula", "train on the booked reward (exact), or on\nposition $\\times$ move $-$ own cost"),
    (0.684, BLUE, "clone + input along a line:  $\\varphi(t)$", "reward diagram: floors, caps, exposure; a range law;\nlogs inside an agreed interval cannot separate floors"),
    (0.562, BLUE, "clone + sign twin  $(r_t,r_t')$", "forgiving rule: $x_t\\in[2\\,\\mathrm{SPR}_t-r_t,\\,r_t]$\nsingle floor: TIR exact where $r_t\\neq r_t'$"),
    (0.440, BLUE, "action, sign, own cost (no clone)", "PMR: pooled magnitude (Assumption M);\ncertificate: credit $\\geq (r_t+C_t)^+$ if gross $\\leq 0$"),
    (0.340, AMBER, "reported reward only", "booked profit not identified (Proposition 3)"),
]
heights = (0.112, 0.112, 0.112, 0.112, 0.090)
for (y, face, head, body), h in zip(rows, heights):
    box(ax, 0.675, y, 0.32, h, face)
    text(ax, 0.687, y + h - 0.010, head, size=5.8, weight="bold")
    text(ax, 0.687, y + h - 0.045, body, size=5.2, color="#2c3742")
arrow(ax, (0.648, 0.72), (0.673, 0.72))

# bottom strip: evidence
box(ax, 0.004, 0.012, 0.992, 0.312, "#ffffff", edge="#9aa5b1")
text(ax, 0.018, 0.314, "Tests (2,085 registered agents; gym-mtsim and engine scan exploratory)", weight="bold", size=6.4)
LINES = (
    "gym-mtsim, unmodified: reward = change in equity matches all 1,100 probes yet hides 11,587–11,846 USD; 5,000 logged moves inside 0.4% are blind to the floor.",
    "Sleeves: reported-reward PPO gambled in 24 of 30 agents; ledger 1, SPR 0, TIR 2; SPR, TIR kept 95–116% of the ledger edge.",
    "No ORPO, KL or quadratic weight did both; a 5% ruin penalty did in one account, not with split accounts or isolated margin.",
    "Carried deficit: SPR, TIR kept 1% (no funded training) to 58% (half). Public simulator, linear reward: bust in 61% of windows (2% with the loss charged).",
    "Engine, real clones, ten runs: reading within 0.0001 points (largest twin-audit errors: TIR 9–390, SPR 31–1,410); 8 of 12 rules met.",
)
y = 0.272
for line in LINES:
    text(ax, 0.018, y, line, size=4.7)
    y -= 0.041
text(ax, 0.018, y, "Limits: sleeves are ours; own cost must be declared; range is what the simulator admits; a trusted formula needs none of this.", size=4.9, color="#9b2c20")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_overview.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(HERE / "fig_overview.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT / "fig_overview.pdf")
