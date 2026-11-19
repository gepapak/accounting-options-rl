"""The twin bracket: an interval for the booked P&L that needs only an accounting that forgives (exploratory, 3 November 2026).

    python twin_bracket.py      # writes twin_bracket_results.json, twin_bracket.md and the manuscript table sections/tab_twin_bracket.tex

Claim. Let a step's booked P&L be x = sigma*g - C (gross antisymmetric in the exogenous sign, own cost C the same on both branches and known),
and let the simulator report r = x + l(sigma) on the real branch and r' = -sigma*g - C + l(-sigma) on its sign twin, with ANY credits
l >= 0 (an accounting that can only forgive). Then
    x = r - l(sigma) <= r      and      x = -r' - 2C + l(-sigma) >= -r' - 2C = 2*SPR - r,
so x lies in [2*SPR - r, r], an interval of width r + r' + 2C = l(sigma) + l(-sigma), the total credit paid on the pair. Over all accountings
with non-negative credits the interval is the sharp identified set (both ends attainable), SPR is its midpoint, and a negative width
(r + r' + 2C < 0) is impossible, so it exposes an undeclared charge. No single-floor assumption is needed: the bracket stays valid for the
two-write floor of Study X and for the public engine's own cash writes, where TIR's point estimate is wrong.

Because the bracket holds step by step, it holds for every episode sum. The stored audits of Studies V, W, X and Y keep, per agent and test
episode, the booked return, the reported return and the SPR sum, so the episode bracket [2*SPR_sum - reported_sum, reported_sum] can be
checked against the booked return of 60,800 episodes without re-running anything. The engine's hand cases are those of
../exploratory_codex_2026-11-01/native_reward_identification_001/runs/hand_v1/RESULTS.json (own cost = fee + funding, equal on both branches).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
ACC = CODE / "ACCOUNTING_OPTIONS_2026-10-14"
TEX = CODE.parent / "accounting-options-rl-latex-latex" / "sections" / "tab_twin_bracket.tex"
LOW_E = ("0", "0.02", "0.05", "0.1")
TOL = 1e-6
STUDIES = [("V", "study_v", "single", "PPO"), ("W", "study_w", "single", "PPO"), ("X", "study_x", "two writes", "PPO"),
           ("Y", "study_y", None, "A2C")]


def x_share(d):
    return float(np.mean([abs(d["lev"][f"{e}|{z}|{r}"]) == 16.0 for e in LOW_E for z in ("-1", "+1") for r in ("0.05", "0.5")]))


def agent_rows():
    rows = []
    for study, folder, floor, learner in STUDIES:
        for p in sorted((ACC / folder / f"results_{folder[-1]}").glob("*.json")):
            d = json.loads(p.read_text())
            fl = floor or ("two writes" if d.get("sleeve") == "two" else "single")
            for mk in ("edge", "placebo"):
                ep = d[f"audit_{mk}"]["episodes"]
                b, rep, spr, tir = (np.asarray(ep[k], float) for k in ("booked", "reported", "spr", "tir"))
                lo, hi = 2 * spr - rep, rep
                n = len(b)
                lb95 = lo.mean() - 1.645 * lo.std(ddof=1) / np.sqrt(n)
                rows.append(dict(study=study, floor=fl, learner=learner, arm=d["arm"], seed=d["seed"], setting=d["setting"], market=mk,
                                 X=x_share(d), episodes=n,
                                 below_lower=int((b < lo - TOL).sum()), above_upper=int((b > hi + TOL).sum()),
                                 negative_width=int((hi - lo < -TOL).sum()),
                                 width_mean=float((hi - lo).mean()), width_max=float((hi - lo).max()),
                                 booked=float(b.mean()), lower_mean=float(lo.mean()), lb95=float(lb95), upper_mean=float(hi.mean()),
                                 tir_err=float(abs(tir.mean() - b.mean())), tir_outside=int(((tir < lo - TOL) | (tir > hi + TOL)).sum())))
    return rows


def engine_cases():
    p = CODE / "exploratory_codex_2026-11-01" / "native_reward_identification_001" / "runs" / "hand_v1" / "RESULTS.json"
    res = json.loads(p.read_text())
    out = []
    for row in res["rows"]:
        e0 = row["known_receipts"]["initial_cash"]
        r = row["reported_ending_cash"]["real"] - e0
        rp = row["reported_ending_cash"]["shadow"] - e0
        gold = row["evaluator_only_gold"]["real"]
        c = gold["cost"] + gold["funding"]
        assert c == row["evaluator_only_gold"]["shadow"]["cost"] + row["evaluator_only_gold"]["shadow"]["funding"]
        booked = float(eval(gold["booked"]))  # noqa: S307  (a fraction string such as "-9")
        lo, hi = -rp - 2 * c, r
        out.append(dict(case=row["case"], r=r, r_twin=rp, own_cost=c, lower=lo, upper=hi, booked=booked,
                        covered=bool(lo - 1e-9 <= booked <= hi + 1e-9), tir=row["naive_TIR_on_cash_increment"],
                        tir_exact=abs(row["naive_TIR_on_cash_increment"] - booked) < 1e-9, width=hi - lo))
    return out


def main():
    rows = agent_rows()
    eng = engine_cases()
    groups = {}
    for r in rows:
        key = (r["study"], r["floor"], r["learner"], "gamblers" if r["X"] >= 0.25 else "non-gamblers")
        groups.setdefault(key, []).append(r)
    summary = []
    for (study, floor, learner, who), rs in sorted(groups.items()):
        summary.append(dict(study=study, floor=floor, learner=learner, who=who, agent_markets=len(rs), episodes=sum(r["episodes"] for r in rs),
                            violations=sum(r["below_lower"] + r["above_upper"] for r in rs), negative_width=sum(r["negative_width"] for r in rs),
                            lb95_covers=sum(r["lb95"] <= r["booked"] + TOL for r in rs),
                            median_width=float(np.median([r["width_mean"] for r in rs])), max_width=float(max(r["width_max"] for r in rs)),
                            median_gap=float(np.median([r["booked"] - r["lb95"] for r in rs])),
                            max_tir_err=float(max(r["tir_err"] for r in rs)), tir_outside=sum(r["tir_outside"] for r in rs)))
    total = dict(agents=len({(r["study"], r["setting"], r["arm"], r["seed"]) for r in rows}), episodes=sum(r["episodes"] for r in rows),
                 violations=sum(r["below_lower"] + r["above_upper"] for r in rows), negative_width=sum(r["negative_width"] for r in rows),
                 lb95_covers=sum(r["lb95"] <= r["booked"] + TOL for r in rows), agent_markets=len(rows),
                 exact_episodes=int(sum(1 for r in rows for _ in [0] if False)))
    res = dict(total=total, summary=summary, engine_cases=eng)
    (HERE / "twin_bracket_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    md = [f"# Twin bracket on stored audits: {total['agents']} agents, {total['episodes']:,} episodes; violations {total['violations']}; "
          f"negative widths {total['negative_width']}; one-sided 95% lower bound below the booked mean in {total['lb95_covers']} of {total['agent_markets']}", "",
          "| study | floor | learner | agents | agent-markets | episodes | violations | median width | max width | median booked - LB95 | max TIR error | TIR outside bracket (episodes) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['study']} | {s['floor']} | {s['learner']} | {s['who']} | {s['agent_markets']} | {s['episodes']} | {s['violations']} | "
                  f"{s['median_width']:.4f} | {s['max_width']:.2f} | {s['median_gap']:.4f} | {s['max_tir_err']:.4f} | {s['tir_outside']} |")
    md += ["", "Engine hand cases (own cost = fee + funding):", "", "| case | r | r' | C | bracket | booked | covered | TIR | TIR exact |", "|---|---|---|---|---|---|---|---|---|"]
    for e in eng:
        md.append(f"| {e['case']} | {e['r']:g} | {e['r_twin']:g} | {e['own_cost']:g} | [{e['lower']:g}, {e['upper']:g}] | {e['booked']:g} | {e['covered']} | {e['tir']:g} | {e['tir_exact']} |")
    (HERE / "twin_bracket.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    def n(v):
        return f"{v:,}".replace(",", "{,}")
    tex = ["\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
           "Study & Cohort & Agents & Episodes & Outside & Median width & Max TIR error \\\\", "\\midrule"]
    for s in summary:
        who = "gamblers" if s["who"] == "gamblers" else "others"
        tex.append(f"{s['study']} & {s['floor']}, {s['learner']}, {who} & {s['agent_markets'] // 2} & {n(s['episodes'])} & {s['violations']} & "
                   f"{s['median_width']:.2f} & {s['max_tir_err']:.2f} \\\\")
    tex += ["\\midrule", f"All & & {total['agents']} & {n(total['episodes'])} & {total['violations']} & & \\\\", "\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    names = {"NO_COST": "No cost", "FEE_FLOOR_NATIVE_TIR_COUNTEREXAMPLE": "Fee before floor", "LOSING_REAL_BRANCH": "Losing real branch",
             "POST_FUNDING_BOTH_CENSORED": "Funding after settlement", "PRE_FEE_POST_FUNDING": "Fee and funding", "TWO_DELIVERIES": "Two deliveries"}
    et = ["\\begin{tabular}{@{}lrrrcrr@{}}", "\\toprule", "Hand case & $r$ & $r'$ & $C$ & Bracket & Booked & TIR \\\\", "\\midrule"]
    star = "$^\\ast$"
    for e in eng:
        mark = "" if e["tir_exact"] else star
        et.append(f"{names[e['case']]} & ${e['r']:g}$ & ${e['r_twin']:g}$ & ${e['own_cost']:g}$ & $[{e['lower']:g},\\,{e['upper']:g}]$ & ${e['booked']:g}$ & "
                  f"${e['tir']:g}${mark} \\\\")
    et += ["\\bottomrule", "\\end{tabular}"]
    (TEX.parent / "tab_twin_bracket_engine.tex").write_text("\n".join(et) + "\n", encoding="utf-8")
    print("wrote", TEX, "and tab_twin_bracket_engine.tex")


if __name__ == "__main__":
    main()
