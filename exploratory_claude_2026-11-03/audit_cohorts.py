"""Ledger-free audit by cohort, from the stored result files of Studies V, W, X and Y (descriptive; written after the fourth AI-assisted review).

    python audit_cohorts.py     # writes audit_cohorts.md, audit_cohorts.json and the manuscript table sections/tab_audit_cohorts.tex

For every agent: X (share of the 16 reachable low-equity probes at maximal leverage, as in aggregate_x.py) and the absolute error of the TIR
estimate, and of the reported return, of the agent's mean booked return over the 100 test weeks, in the edge and the placebo market. A gambler
has X >= 0.25. Study V's registered audit cohort (VD1) is Parts A and C (settings S_DK1 and S_DK1+Z); its Part B is listed separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ACC = HERE.parent / "ACCOUNTING_OPTIONS_2026-10-14"
TEX = HERE.parents[1] / "accounting-options-rl-latex-latex" / "sections" / "tab_audit_cohorts.tex"
LOW_E = ("0", "0.02", "0.05", "0.1")
GAMBLER_X = 0.25
COHORTS = [  # (key, label for the table, floor, learner, study folder, settings)
    ("V_AC", "V, Parts A, C", "single", "PPO", "study_v", ("S_DK1", "S_DK1+Z")),
    ("V_B", "V, Part B", "single", "PPO", "study_v", ("T_DK2", "T_DK1M")),
    ("W", "W", "single", "PPO", "study_w", None),
    ("X", "X", "two writes", "PPO", "study_x", None),
    ("Y_S", "Y, zero starts", "single", "A2C", "study_y", ("S_DK1+Z",)),
    ("Y_N", "Y, no zero starts", "single", "A2C", "study_y", ("S_DK1",)),
    ("Y_X", "Y, zero starts", "two writes", "A2C", "study_y", ("X_DK1+Z",)),
]


def agent(d):
    low = [abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]
    a, b = d["audit_edge"], d["audit_placebo"]
    return dict(X=float(np.mean(low)), tir_edge=abs(a["tir"] - a["booked"]), tir_plac=abs(b["tir"] - b["booked"]),
                rep_edge=abs(a["reported"] - a["booked"]), rep_plac=abs(b["reported"] - b["booked"]))


def main():
    out = {}
    for key, label, floor, learner, folder, settings in COHORTS:
        res_dir = ACC / folder / f"results_{folder[-1]}"
        rows = []
        for p in sorted(res_dir.glob("*.json")):
            d = json.loads(p.read_text())
            if settings is None or d["setting"] in settings:
                rows.append(agent(d))
        if not rows:
            continue
        g = [r for r in rows if r["X"] >= GAMBLER_X]
        h = [r for r in rows if r["X"] < GAMBLER_X]
        mx = lambda rs, k: float(max(r[k] for r in rs)) if rs else float("nan")  # noqa: E731
        out[key] = dict(label=label, floor=floor, learner=learner, agents=len(rows), gamblers=len(g),
                        max_tir_edge=mx(rows, "tir_edge"), max_tir_plac=mx(rows, "tir_plac"),
                        max_tir_edge_nongamblers=mx(h, "tir_edge"), max_tir_plac_nongamblers=mx(h, "tir_plac"),
                        median_rep_plac_gamblers=float(np.median([r["rep_plac"] for r in g])) if g else float("nan"),
                        median_tir_edge_gamblers=float(np.median([r["tir_edge"] for r in g])) if g else float("nan"))
    single = [k for k in out if out[k]["floor"] == "single"]
    pooled = dict(agents=sum(out[k]["agents"] for k in single), max_tir_edge=max(out[k]["max_tir_edge"] for k in single),
                  max_tir_plac=max(out[k]["max_tir_plac"] for k in single), cohorts=single)
    out["single_floor_registered_cohorts"] = pooled
    (HERE / "audit_cohorts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["| cohort | floor | learner | agents | gamblers | max TIR error edge / placebo | non-gamblers: max edge / placebo | gamblers: median reported error (placebo) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, v in out.items():
        if k == "single_floor_registered_cohorts":
            continue
        md.append(f"| {v['label']} | {v['floor']} | {v['learner']} | {v['agents']} | {v['gamblers']} | {v['max_tir_edge']:.4f} / {v['max_tir_plac']:.4f} | "
                  f"{v['max_tir_edge_nongamblers']:.4f} / {v['max_tir_plac_nongamblers']:.4f} | {v['median_rep_plac_gamblers']:.2f} |")
    md.append(f"\nSingle-floor cohorts {pooled['cohorts']}: {pooled['agents']} agents, max TIR error {pooled['max_tir_edge']:.4f} (edge) / {pooled['max_tir_plac']:.4f} (placebo).")
    (HERE / "audit_cohorts.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    def f(v, nd=2):
        return "--" if v != v else f"{v:.{nd}f}"
    tex = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llrrccr@{}}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{Max TIR error, edge / placebo} & Gamblers' \\\\",
           "\\cmidrule(lr){5-6}",
           "Study & Floor, learner & Agents & Gamblers & All agents & Non-gamblers & reported error \\\\", "\\midrule"]
    for k, v in out.items():
        if k == "single_floor_registered_cohorts":
            continue
        tex.append(f"{v['label']} & {v['floor']}, {v['learner']} & {v['agents']} & {v['gamblers']} & {f(v['max_tir_edge'])} / {f(v['max_tir_plac'])} & "
                   f"{f(v['max_tir_edge_nongamblers'], 3)} / {f(v['max_tir_plac_nongamblers'], 3)} & {f(v['median_rep_plac_gamblers'], 1)} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("wrote", TEX)


if __name__ == "__main__":
    main()
