"""Holm sensitivity analysis of the registered decisions of Studies S to Z and RS (exploratory; the registered verdicts are unadjusted and unchanged).

    python holm_sensitivity.py     # writes holm_sensitivity.md and ../../accounting-options-rl-latex-latex/sections/tab_holm.tex

Every registered decision whose rule contains a significance test ("p = ... (need < 0.05)") is re-decided with Holm-adjusted p-values, (a) within
its study and (b) across all such decisions together; the other parts of a conjunctive rule (mean margins, bootstrap bounds)
are kept as registered. Decisions without a significance test are unaffected. The p-values are those printed in the VERDICTS files (four
decimals); a printed 0.0000 is treated as 0.00005. Extended on 5 November 2026 from Studies S to Y to Studies S to Z and RS, and on 7 November 2026 to
Studies EXT and FT; the two parts of Study Z form one study. The last lines of holm_sensitivity.md give the result for Studies S to Y and for S to Z and RS alone.
"""
from __future__ import annotations

import re
from pathlib import Path

ACC = Path(__file__).resolve().parent.parent / "ACCOUNTING_OPTIONS_2026-10-14"
HERE = Path(__file__).resolve().parent
TEX = HERE.parents[1] / "accounting-options-rl-latex-latex" / "sections" / "tab_holm.tex"
STUDIES = {s: [f"study_{s.lower()}/VERDICTS_{s}.md"] for s in "STUVWXY"}
STUDIES["Z"] = ["study_z/VERDICTS_Z.md", "study_zb/VERDICTS_ZB.md"]
STUDIES["RS"] = ["study_rs/VERDICTS_RS.md"]
STUDIES["EXT"] = ["study_ext/VERDICTS_EXT.md"]
STUDIES["FT"] = ["study_ft/VERDICTS_FT.md"]
ROW = re.compile(r"^\|\s*([A-Z][^|]*?)\s*\|\s*(.*)\|\s*(SUPPORTED|NOT SUPPORTED|NOT TESTABLE)\s*\|\s*$")
PVAL = re.compile(r"p = (\d+\.\d+) \(need < 0\.05\)")


def holm(ps):
    """Holm step-down adjusted p-values, in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    rows = []
    for s, files in STUDIES.items():
        for name in files:
            for line in (ACC / name).read_text(encoding="utf-8").splitlines():
                m = ROW.match(line)
                if not m:
                    continue
                pm = PVAL.search(m.group(2))
                p = None if pm is None else max(float(pm.group(1)), 0.00005)
                rows.append(dict(study=s, id=m.group(1).strip(), verdict=m.group(3), p=p))
    tested = [r for r in rows if r["p"] is not None and r["verdict"] != "NOT TESTABLE"]
    for s in STUDIES:
        sub = [r for r in tested if r["study"] == s]
        for r, a in zip(sub, holm([r["p"] for r in sub])):
            r["p_holm_study"] = a
    for r, a in zip(tested, holm([r["p"] for r in tested])):
        r["p_holm_all"] = a
    summary = []
    for s in STUDIES:
        sr = [r for r in rows if r["study"] == s]
        if not sr:
            continue
        pt = [r for r in sr if r["p"] is not None and r["verdict"] != "NOT TESTABLE"]
        sup = sum(r["verdict"] == "SUPPORTED" for r in sr)
        lost_s = sum(1 for r in pt if r["verdict"] == "SUPPORTED" and r["p_holm_study"] >= 0.05)
        lost_a = sum(1 for r in pt if r["verdict"] == "SUPPORTED" and r["p_holm_all"] >= 0.05)
        worst = max((r["p_holm_all"] for r in pt if r["verdict"] == "SUPPORTED"), default=float("nan"))
        summary.append(dict(study=s, decisions=len(sr), with_test=len(pt), supported=sup, supported_holm_study=sup - lost_s,
                            supported_holm_all=sup - lost_a, worst_holm_all=worst))
    md = ["| study | decisions | with a significance test | supported (registered) | supported, Holm within study | supported, Holm across all | largest Holm-adjusted p (across all) among supported |",
          "|---|---|---|---|---|---|---|"]
    for v in summary:
        md.append(f"| {v['study']} | {v['decisions']} | {v['with_test']} | {v['supported']} | {v['supported_holm_study']} | {v['supported_holm_all']} | {v['worst_holm_all']:.4f} |")
    md.append("")
    md.append("Changed decisions: " + (", ".join(f"{r['study']}:{r['id']} (p {r['p']:.4f}, Holm {r['p_holm_all']:.4f})" for r in tested
                                               if r["verdict"] == "SUPPORTED" and (r["p_holm_all"] >= 0.05 or r["p_holm_study"] >= 0.05)) or "none"))
    sup = [r for r in tested if r["verdict"] == "SUPPORTED"]
    md.append(f"Decisions with a significance test: {len(tested)}; supported: {len(sup)}; largest adjusted p among the supported: {max(r['p_holm_all'] for r in sup):.4f}")
    for label, keep in (("Studies S to Y alone", lambda r: r["study"] not in ("Z", "RS", "EXT", "FT")), ("Studies S to Z and RS alone", lambda r: r["study"] not in ("EXT", "FT"))):
        old = [r for r in tested if keep(r)]
        adj = holm([r["p"] for r in old])
        md.append(f"{label}: {len(old)} decisions; largest adjusted p among the supported {max(a for r, a in zip(old, adj) if r['verdict'] == 'SUPPORTED'):.4f}; "
                  f"changed decisions {sum(1 for r, a in zip(old, adj) if r['verdict'] == 'SUPPORTED' and a >= 0.05)}")
    (HERE / "holm_sensitivity.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    nan = lambda x: "--" if x != x else f"{x:.4f}"  # noqa: E731
    tex = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
           " & & & & \\multicolumn{2}{c}{Supported after Holm} & Largest \\\\", "\\cmidrule(lr){5-6}",
           "Study & Decisions & Tested & Supported & within study & across all & adjusted $p$ \\\\", "\\midrule"]
    for v in summary:
        tex.append(f"{v['study']} & {v['decisions']} & {v['with_test']} & {v['supported']} & {v['supported_holm_study']} & {v['supported_holm_all']} & {nan(v['worst_holm_all'])} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("wrote", TEX)


if __name__ == "__main__":
    main()
