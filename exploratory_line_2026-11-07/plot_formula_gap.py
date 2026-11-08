"""One state from the completed audit, for the manuscript figure. Does not refit any rule."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_formula_gap as g

HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "results.json").read_text())["summary"]
entry = summary["example_public"]["entry"]
control = json.loads((HERE / "results.json").read_text())["control"]
print("control worst shortfall min/max",
      min(r["worst_shortfall"] for r in control),
      max(r["worst_shortfall"] for r in control))

env, _ = g.prepared(entry, floor=True)
entry_price = float(env.simulator.orders[0].entry_price)
ts, reward, economic = [], [], []
for t in g.GRID:
    price = entry_price * (1.0 + g.DP_REL * float(t))
    row = g.query(env, price)
    ts.append(float(t))
    reward.append(row["reward"])
    economic.append(row["economic"])
env_c, _ = g.prepared(entry, floor=False)
reward_c = []
for t in g.GRID:
    price = entry_price * (1.0 + g.DP_REL * float(t))
    reward_c.append(g.query(env_c, price)["reward"])

ts = np.array(ts)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.axhline(0, color="0.6", lw=0.6)
ax.plot(ts, economic, "o-", color="#1b4f72", ms=4, label="Economic profit of the step")
ax.plot(ts, reward, "s-", color="#b03a2e", ms=4, label="Public reward (change in equity)")
ax.plot(ts, reward_c, "--", color="#196f3d", lw=1.2, label="Same step, balance floor removed")
ax.set_xlabel("Price move, in units of one percent of the entry price")
ax.set_ylabel("USD over the probed step")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
out = HERE.parents[1] / "accounting-options-rl-latex-latex" / "figures" / "fig_mtsim_formula.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print("wrote", out)
