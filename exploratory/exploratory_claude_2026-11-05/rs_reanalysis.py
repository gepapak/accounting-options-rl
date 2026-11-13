"""Exploratory reanalysis of Study RS from its per-run results and the market panels (not registered; the registered verdicts are unchanged).

    python rs_reanalysis.py            # writes rs_reanalysis.json and the manuscript's sections/tab_studyRS_months.tex
    python rs_reanalysis.py --print    # prints the results, writes nothing

It reports what the registered threshold decisions do not: the head-to-head contrast of TIR and SPR, the uncertainty that comes from the reused
test panel, and what the 8-row lag and the 672-row episodes are in calendar time. Reads stored results only and trains nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RES = REPO / "ACCOUNTING_OPTIONS_2026-10-14" / "study_rs" / "results_rs"
PANEL = REPO / "GENERALIZATION_STUDY_2026-09-29" / "data_panel"
TEX = REPO.parent / "accounting-options-rl-latex-latex" / "sections" / "tab_studyRS_months.tex"
ARMS = ("PLAIN", "ORACLE", "SPR", "TIR", "RUIN5")
SEEDS = range(300, 316)
EP, LAG = 672, 8
N_BOOT = 20_000


def load():
    return {a: [json.loads((RES / f"RS_DK2__{a}__s{s}.json").read_text()) for s in SEEDS] for a in ARMS}


def time_semantics(panel: pd.DataFrame) -> dict:
    out = {}
    for split in ("train", "test"):
        idx = panel[panel.split == split].index
        step = pd.Series(idx[1:] - idx[:-1])
        lag = pd.Series(idx[LAG:] - idx[:-LAG])
        span = pd.Series(idx[EP - 1:] - idx[:-(EP - 1)])
        week = pd.Timedelta(days=7) - pd.Timedelta(minutes=15)
        out[split] = dict(rows=len(idx), first=str(idx[0]), last=str(idx[-1]),
                          missing_quarters=int(((step - pd.Timedelta(minutes=15)).dt.total_seconds().clip(lower=0) / 900).sum()),
                          largest_gap=str(step.max()), lag_exactly_2h=float((lag == pd.Timedelta(hours=2)).mean()), lag_max=str(lag.max()),
                          windows_longer_than_week=float((span > week).mean()), window_span_max_days=float(span.max().total_seconds() / 86400))
    return out


def analysis() -> dict:
    runs = load()
    E = {a: np.array([d["test_edge"]["booked_episodes"] for d in runs[a]]) for a in ARMS}   # seeds x episodes
    e = {a: E[a].mean(axis=1) for a in ARMS}
    tir, spr, ora = e["TIR"], e["SPR"], e["ORACLE"]
    rng = np.random.default_rng(20261105)
    res: dict = {}
    # 1. head to head over the 16 training seeds
    boot = np.array([rng.choice(tir, 16).mean() - rng.choice(spr, 16).mean() for _ in range(N_BOOT)])
    r_t = np.array([rng.choice(tir, 16).mean() / rng.choice(ora, 16).mean() for _ in range(N_BOOT)])
    r_s = np.array([rng.choice(spr, 16).mean() / rng.choice(ora, 16).mean() for _ in range(N_BOOT)])
    res["head_to_head"] = dict(difference=float(tir.mean() - spr.mean()), ci95=[float(x) for x in np.quantile(boot, [.025, .975])],
                               welch_p=float(stats.ttest_ind(tir, spr, equal_var=False).pvalue),
                               mann_whitney_p=float(stats.mannwhitneyu(tir, spr, alternative="two-sided").pvalue),
                               paired_t_p=float(stats.ttest_1samp(tir - spr, 0).pvalue), wilcoxon_p=float(stats.wilcoxon(tir - spr).pvalue),
                               seeds_tir_above_spr=int((tir > spr).sum()))
    res["ratio_ci95"] = dict(TIR=[float(x) for x in np.quantile(r_t, [.025, .975])], SPR=[float(x) for x in np.quantile(r_s, [.025, .975])])
    # 2. the test windows, reconstructed from the evaluation seed (EdgeEnvT.reset draws one integer per episode)
    panel = pd.read_pickle(PANEL / "panel_DK2.pkl")
    p = panel[panel.split == "test"]
    n = len(p)
    g = np.random.default_rng(9000)
    t0s = np.array([int(g.integers(200, n - EP)) for _ in range(100)])
    cover = np.zeros(n, int)
    for t in t0s:
        cover[t:t + EP] += 1
    res["windows"] = dict(share_of_test_quarters_covered=float((cover > 0).mean()), mean_multiplicity=float(cover[cover > 0].mean()),
                          overlapping_pairs=int(sum(abs(t0s[i] - t0s[j]) < EP for i in range(100) for j in range(i + 1, 100))),
                          non_overlapping_weeks_in_panel=n // EP,
                          windows_with_gap=int(sum((p.index[t + EP - 1] - p.index[t]) > pd.Timedelta(days=7) - pd.Timedelta(minutes=15) for t in t0s)))
    # 3. uncertainty from the panel: resample seeds and windows (optimistic, windows overlap), and seeds and calendar months
    months = np.array([p.index[t].strftime("%Y-%m") for t in t0s])
    two_way = {"TIR": [], "SPR": []}
    by_month = {"TIR": [], "SPR": []}
    mm = sorted(set(months))
    for _ in range(N_BOOT):
        w = rng.integers(0, 100, 100)
        idx = np.concatenate([np.where(months == m)[0] for m in rng.choice(mm, len(mm))])
        for a in two_way:
            sa, so = rng.integers(0, 16, 16), rng.integers(0, 16, 16)
            two_way[a].append(E[a][sa][:, w].mean() / E["ORACLE"][so][:, w].mean())
            by_month[a].append(E[a][sa][:, idx].mean() / E["ORACLE"][so][:, idx].mean())
    res["lower_bound_5pct"] = dict(two_way={a: float(np.quantile(v, .05)) for a, v in two_way.items()},
                                   seeds_and_months={a: float(np.quantile(v, .05)) for a, v in by_month.items()})
    res["months"] = {m: {**{a: float(E[a][:, months == m].mean()) for a in ("ORACLE", "TIR", "SPR", "PLAIN")}, "episodes": int((months == m).sum())} for m in mm}
    # 4. markets and calendar time
    res["time_semantics_DK2"] = time_semantics(panel)
    s_tr, s_te = ((panel[panel.split == k].imb - panel[panel.split == k].spot).to_numpy() for k in ("train", "test"))
    res["sign_shares"] = {k: dict(positive=float(np.mean(x > 0)), zero=float(np.mean(x == 0)), negative=float(np.mean(x < 0)), mean_abs=float(np.abs(x).mean()))
                          for k, x in (("train", s_tr), ("test", s_te))}
    res["lag8_agreement"] = {}
    for k, x in (("train", s_tr), ("test", s_te)):
        a, b = np.sign(x[LAG:]), np.sign(x[:-LAG])
        nz = (a != 0) & (b != 0)
        res["lag8_agreement"][k] = float(np.mean(a[nz] == b[nz]))
    # 5. reported and booked returns
    res["returns"] = {a: dict(reported_edge=float(np.mean([d["test_edge"]["reported_pct_K0"] for d in runs[a]])), booked_edge=float(e[a].mean()),
                              forgiven_edge=float(np.mean([d["test_edge"]["forgiven_pct_K0"] for d in runs[a]])),
                              reported_placebo=float(np.mean([d["test_placebo"]["reported_pct_K0"] for d in runs[a]])),
                              booked_placebo=float(np.mean([d["test_placebo"]["booked_pct_K0"] for d in runs[a]]))) for a in ARMS}
    return res


def write_tex(res: dict) -> None:
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule", "Start month & Episodes & Ledger & TIR & SPR & Reported \\\\", "\\midrule"]
    for m, v in res["months"].items():
        lines.append(f"{m} & {v['episodes']} & ${v['ORACLE']:+.1f}$ & ${v['TIR']:+.1f}$ & ${v['SPR']:+.1f}$ & ${v['PLAIN']:+.1f}$ \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="print_only", action="store_true")
    a = ap.parse_args()
    res = analysis()
    print(json.dumps(res, indent=1))
    if a.print_only:
        return
    (HERE / "rs_reanalysis.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    write_tex(res)
    print("wrote rs_reanalysis.json and", TEX.name)


if __name__ == "__main__":
    main()
