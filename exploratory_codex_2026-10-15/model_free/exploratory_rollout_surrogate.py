"""EXPLORATORY, not pre-registered: DP-free fixed-policy rollout diagnostics.

No learner, critic, DP solver, or optimized continuation policy is used. The primary
reference is leverage 16 below one allocation and flat otherwise. Flat and constant
16 are fixed diagnostic references; all are reported without outcome-based selection.
Run only after Study Q has finished. See README.md for estimands and limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LIAB = ROOT / "LIABILITY_STUDY_2026-09-29"
ACCOUNTING = ROOT / "ACCOUNTING_OPTIONS_2026-10-14"
PROBE = (0.02, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
ALPHAS = (0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)
REFERENCES = ("flat", "threshold1", "max16")
K0, Q1 = 20000.0, 0.25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def cid(zone, costs, gamma, alpha):
    return f"{zone}|{costs}|g{gamma:g}|a{alpha:g}"


def index_pct(gaps, sigma):
    """Same seven-probe/log-interpolation functional as O, different Q/noise."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(2.0), 400))
    interp = np.interp(np.log(grid), np.log(PROBE), gaps)
    return float(100 * np.maximum(interp, 0.0).mean() / sigma)


def conditions():
    """Read stored DP outputs only; never solve a DP or modify its artifacts."""
    out = {}
    with (LIAB / "results_k/action_gap.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            z, c, g = r["zone"], r["costs"], float(r["gamma"])
            key = cid(z, c, g, 0.0)
            sigma = float(r["sigma16"])
            gap = [float(r[f"A_{e:g}"]) for e in PROBE]
            out[key] = dict(condition_id=key, zone=z, costs=c, gamma=g,
                            alpha=0.0, dp_index_pct=index_pct(gap, sigma),
                            dp_sigma16=sigma, legacy_condition=r["condition"],
                            dp_sources="legacy action_gap.csv")
    for z in ("DK1", "DK2"):
        d = json.loads((ACCOUNTING / f"study_o/dp_o_{z}.json").read_text())
        for a, arm in d["arms"].items():
            key = cid(z, "base", 0.99, float(a))
            if key in out:
                out[key]["dp_secondary_index_pct"] = float(arm["index_pct"])
                out[key]["dp_sources"] += "; Study O stored design"
            else:
                out[key] = dict(condition_id=key, zone=z, costs="base", gamma=0.99,
                                alpha=float(a), dp_index_pct=float(arm["index_pct"]),
                                dp_sigma16=float(d["sigma16"]), legacy_condition="",
                                dp_sources="Study O stored design")
    return out


def empirical_payoffs(zone, costs):
    """Exact TRAIN rows/cap formula of envs_b/envs_j/dp_frictions, no env imports."""
    panel_path = ((ROOT / "GENERALIZATION_STUDY_2026-09-29/data_panel")
                  if zone in ("DK1", "DK2") else LIAB / "data_panel_j") / f"panel_{zone}.pkl"
    p = pd.read_pickle(panel_path)
    p = p.loc[p["split"] == "train"].iloc[200:]
    spread = np.abs(p["imb"].to_numpy(float) - p["spot"].to_numpy(float))
    cap = 0.25 * 0.25 * p["bal_mw"].to_numpy(float)
    if costs == "H":
        cap *= 0.10 / 0.25
    quantity = np.minimum(16 * Q1, cap)
    cost = 0.10 * quantity
    if costs == "H":
        cost = quantity + 50.0 * quantity ** 2 / np.maximum(cap, 0.05)
    mag = quantity * spread
    if not len(mag) or not (np.isfinite(mag).all() and np.isfinite(cost).all()) or (cap < 0).any():
        raise ValueError(f"Invalid empirical payoffs: {zone}/{costs}")
    # Uniform row, independent fair sign: mean=-E[c], E[X^2]=E[z^2+c^2].
    sigma = float(np.sqrt(np.mean(mag ** 2 + cost ** 2) - np.mean(cost) ** 2))
    digest = hashlib.sha256(mag.tobytes() + cost.tobytes()).hexdigest()
    panel_digest = sha(panel_path)
    registered = []
    panel_status = "not_a_Study_J_panel"
    revised = False
    if zone in ("FI", "NO2"):
        rel = f"data_panel_j/panel_{zone}.pkl"
        for line in (LIAB / "PREREGISTRATION_HASHES.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1] == "*" + rel:
                registered.append(parts[-2].lower())
        if not registered:
            raise ValueError(f"No registered Study J panel hash for {zone}")
        revised = panel_digest.lower() not in registered
        panel_status = "REVISED_J_PANEL" if revised else "MATCHES_REGISTERED_J_PANEL"
    return mag, cost, sigma, dict(panel=str(panel_path.relative_to(ROOT)),
                                 panel_sha256=panel_digest, payoff_sha256=digest,
                                 registered_J_panel_sha256=";".join(registered),
                                 empirical_panel_registration_status=panel_status,
                                 dp_source_revision_confound=revised,
                                 n_empirical_train_rows=len(mag))


def horizon_for(reference, gamma, max_pnl, sigma, target_pct, maximum):
    if reference == "flat":
        return 1, 0.0
    # max16: same future actions; reflected equity difference contracts and keeps
    # its sign, so omitted discounted reward difference <= gamma^H * max|X|.
    # threshold1: |reward| <= max|X| on every step for every alpha in [0,1].
    scale = max_pnl if reference == "max16" else 2 * max_pnl / (1 - gamma)
    ratio = target_pct * sigma / (100 * scale)
    h = max(1, int(math.ceil(math.log(min(ratio, 0.999999)) / math.log(gamma))))
    h = min(h, maximum)
    return h, 100 * scale * gamma ** h / sigma


def batch_rollout(mag, cost, gamma, reference, horizon, paths, rng):
    """Independent antithetic path pairs; CRN between first-action branches.

    Each independent pair shares magnitudes and uses opposite fair signs at every
    time. All equity probes share this randomness. Bootstrap unit is a complete
    independent batch, retaining probe/alpha dependence. Reported equity is never
    clipped to the DP grid. Returns are in EUR, without learner normalization.
    """
    equity = K0 * np.asarray(PROBE)[None, :, None]
    high = np.broadcast_to(equity, (2, len(PROBE), paths)).copy()
    low = high.copy()
    reported = np.zeros_like(high)
    booked = np.zeros_like(high)
    active = np.ones_like(high, dtype=bool)
    transitions = 0
    for t in range(horizon):
        pick = rng.integers(0, len(mag), size=paths)
        signs = 2 * rng.integers(0, 2, size=paths) - 1
        x = (np.asarray([1.0, -1.0])[:, None] * signs[None, :] * mag[pick][None, :]
             - cost[pick][None, :])[:, None, :]
        if t == 0:
            ah, al = np.ones_like(high, dtype=bool), np.zeros_like(low, dtype=bool)
        elif reference == "max16":
            ah, al = active, active
        elif reference == "threshold1":
            ah, al = active & (high < K0), active & (low < K0)
        else:
            break
        ph, pl = ah * x, al * x
        high2, low2 = np.maximum(high + ph, 0), np.maximum(low + pl, 0)
        reported += gamma ** t * ((high2 - high) - (low2 - low))
        booked += gamma ** t * (ph - pl)
        transitions += 2 * int(active.sum())
        high, low = high2, low2
        if reference == "threshold1":
            active &= (high != low) & ~((high >= K0) & (low >= K0))
        elif reference == "max16":
            active &= high != low
        else:
            active[:] = False
        if not active.any():
            break
    return (np.stack([reported.mean(axis=(0, 2)), booked.mean(axis=(0, 2))]),
            transitions, t + 1, float(active.mean()))


def checkpoint(path, spec, means, seconds, counts, steps, active):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, spec=json.dumps(spec, sort_keys=True), means=np.asarray(means),
                            seconds=np.asarray(seconds), counts=np.asarray(counts),
                            steps=np.asarray(steps), active=np.asarray(active))
    tmp.replace(path)


def summarize_group(spec, means, conds, sigma, tail_pct, rng, boot):
    b = np.asarray(means)
    # Block bootstrap: recompute positive part only after averaging complete batches.
    draw = rng.integers(0, len(b), size=(boot, len(b)))
    boot_means = b[draw].mean(axis=1)
    rows = []
    for c in conds:
        a = c["alpha"]
        gaps = (1 - a) * b[:, 0] + a * b[:, 1]
        point = index_pct(gaps.mean(axis=0), sigma)
        simulated = [index_pct((1 - a) * v[0] + a * v[1], sigma) for v in boot_means]
        lo, hi = np.quantile(simulated, [0.025, 0.975])
        tail = tail_pct * (1 - a) if spec["reference"] == "max16" else tail_pct
        rows.append(dict(**c, reference=spec["reference"], primary_reference=spec["reference"] == "threshold1",
                         surrogate_index_pct=point, mc_low_pct=float(lo), mc_high_pct=float(hi),
                         truncation_bound_pct=tail, combined_low_pct=max(0.0, float(lo) - tail),
                         combined_high_pct=float(hi) + tail, sigma16_empirical=sigma,
                         sigma_empirical_over_binned=sigma / c["dp_sigma16"],
                         empirical_panel_registration_status=spec["empirical_panel_registration_status"],
                         dp_source_revision_confound=spec["dp_source_revision_confound"],
                         shared_rollout_group=spec["group"], n_batches=len(b),
                         antithetic_pairs_per_batch=spec["paths"], horizon=spec["horizon"],
                         n_paired_trajectories_per_probe=2 * len(b) * spec["paths"]))
    return rows


def finite(value):
    v = float(value)
    return v if math.isfinite(v) else None


def compare_and_transfer(rows, outdir):
    result = {"status": "partial", "rank_comparisons": [], "thresholds": [],
              "interpretation": "Exploratory. O/P/Q share condition indices; they are not independent index replications."}
    if not rows:
        return result
    from scipy import stats
    d = pd.DataFrame(rows)
    cal = json.loads((ACCOUNTING / "study_o/calibration_o.json").read_text())["rows"]
    outcomes = []
    op = pd.read_csv(ACCOUNTING / "study_o/runs_o.csv")
    outcomes.append(("O", "PPO", 1000000, op))
    pp = pd.read_csv(ACCOUNTING / "study_p/runs_p.csv")
    for budget, subset in pp[pp["study"] != "O"].groupby("budget"):
        outcomes.append(("P", "PPO", int(budget), subset))
    qp = ACCOUNTING / "study_q/runs_q.csv"
    if qp.exists():
        for learner, subset in pd.read_csv(qp).groupby("learner"):
            outcomes.append(("Q", learner, 1000000, subset))
    transfer = []
    coverage = []
    for study, learner, budget, data in outcomes:
        for (z, a), s in data.groupby(["region", "alpha"]):
            coverage.append(dict(study=study, learner=learner, budget=budget, zone=z,
                                 alpha=float(a), n_runs=len(s), condition_id=cid(z, "base", .99, float(a))))
    write_csv(outdir / "exploratory_condition_coverage.csv", coverage)
    for ref, r in d.groupby("reference"):
        for name, sub in (("all_unique_conditions", r), ("excluding_revised_J_panel", r[~r.dp_source_revision_confound]),
                          ("legacy_FHJK_records", r[r.legacy_condition != ""]),
                          ("O_shared_OPQ_conditions", r[(r.costs == "base") & (r.gamma == .99) & r.zone.isin(["DK1", "DK2"]) ]),
                          ("O_shared_OPQ_treatments", r[(r.costs == "base") & (r.gamma == .99) & (r.alpha < 1) & r.zone.isin(["DK1", "DK2"]) ])):
            corr = stats.spearmanr(sub.dp_index_pct, sub.surrogate_index_pct)
            result["rank_comparisons"].append(dict(reference=ref, subset=name, n_conditions=len(sub),
                                                      spearman=finite(corr.statistic),
                                                      rmse_index_pct=float(np.sqrt(np.mean((sub.dp_index_pct - sub.surrogate_index_pct) ** 2)))))
        labeled = []
        for c in cal:
            costs = "H" if c["condition"] == "Study H costs" else "base"
            gamma = .99 if costs == "H" else float(c["condition"].split()[-1])
            hit = r[r.condition_id == cid(c["zone"], costs, gamma, 0)]
            if len(hit) == 1:
                labeled.append(dict(hit.iloc[0], learned=c["learned"]))
        c = pd.DataFrame(labeled)
        fit = dict(reference=ref, n_calibration_conditions=len(c), separated=False, threshold_pct=None,
                   calibration_includes_revised_J_panel=bool(c.dp_source_revision_confound.any()) if len(c) else False)
        if len(c) == len(cal):
            unlearned, learned = c[~c.learned], c[c.learned]
            h, l = float(unlearned.surrogate_index_pct.max()), float(learned.surrogate_index_pct.min())
            fit.update(largest_unlearned_pct=h, smallest_learned_pct=l, separated=h < l and h > 0)
            if fit["separated"]:
                threshold = math.sqrt(h * l)
                fit["threshold_pct"] = threshold
                fit["uncertainty_separation"] = bool(unlearned.combined_high_pct.max() < learned.combined_low_pct.min())
                for study, learner, budget, data in outcomes:
                    zmap = {v.condition_id: v for v in r.itertuples()}
                    treatment = data[data.alpha != 1].copy()
                    predictions = []
                    for row in treatment.itertuples():
                        key = cid(row.region, "base", .99, row.alpha)
                        if key in zmap:
                            predictions.append((zmap[key].surrogate_index_pct >= threshold) == (row.L002 == 16))
                    ctrl = data[data.alpha == 1].L002.mean()
                    for a, group in treatment.groupby("alpha"):
                        vals = [zmap[cid(z, "base", .99, float(a))] for z in sorted(group.region.unique())
                                if cid(z, "base", .99, float(a)) in zmap]
                        if len(vals) != len(group.region.unique()):
                            continue
                        value = float(np.mean([v.surrogate_index_pct for v in vals]))
                        gap = float(group.L002.mean() - ctrl)
                        low = float(np.mean([v.combined_low_pct for v in vals]))
                        high = float(np.mean([v.combined_high_pct for v in vals]))
                        transfer.append(dict(reference=ref, study=study, learner=learner, budget=budget,
                                             alpha=float(a), mean_zone_surrogate_pct=value,
                                             calibration_threshold_pct=threshold, predicted_learned=value >= threshold,
                                             observed_gap=gap, observed_learned=gap >= 4,
                                             arm_prediction_correct=(value >= threshold) == (gap >= 4),
                                             mc_and_truncation_straddles_point_threshold=low <= threshold <= high,
                                             n_treatment_runs=len(group),
                                             cohort_raw_max16_accuracy=float(np.mean(predictions)) if predictions else None,
                                             cohort_n_scored_raw_max16=len(predictions)))
        result["thresholds"].append(fit)
    write_csv(outdir / "exploratory_transfer.csv", transfer)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", type=int, default=24)
    ap.add_argument("--paths", type=int, default=256, help="independent antithetic pairs per batch per probe")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026101560)
    ap.add_argument("--tail-bound-pct", type=float, default=0.01)
    ap.add_argument("--max-horizon", type=int, default=6000)
    ap.add_argument("--max-wall-seconds", type=float, default=900)
    ap.add_argument("--references", nargs="+", choices=REFERENCES, default=list(REFERENCES))
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    a = ap.parse_args()
    if (a.batches < 2 or a.paths < 1 or a.bootstrap < 10 or not 0 < a.tail_bound_pct < 1
            or a.max_horizon < 1 or a.max_wall_seconds <= 0 or a.seed < 0):
        ap.error("Need >=2 batches, positive paths/horizon/wall cap, nonnegative seed, >=10 bootstrap draws and 0<tail-bound-pct<1.")
    if not (ACCOUNTING / "study_q/VERDICTS_Q.md").exists():
        raise SystemExit("Study Q has no verdict file. No analysis started; wait until Q is complete.")
    if not (ACCOUNTING / "study_q/runs_q.csv").exists():
        raise SystemExit("Study Q run table missing; complete unchanged Q aggregation first.")
    with (ACCOUNTING / "study_q/runs_q.csv").open(newline="") as f:
        q_rows = list(csv.DictReader(f))
    if len(q_rows) != 100:
        raise SystemExit(f"Study Q has {len(q_rows)} aggregated runs, expected 100; no analysis started.")
    if len({(r["learner"], r["region"], r["alpha"], r["seed"]) for r in q_rows}) != 100:
        raise SystemExit("Study Q aggregate has duplicate run identities; no analysis started.")
    global np, pd
    import numpy as np
    import pandas as pd
    outdir = a.out.resolve()
    if not outdir.is_relative_to(HERE):
        raise SystemExit("Outputs must remain inside this model_free subfolder.")
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    conds = conditions()
    rows, groups = [], []
    complete = True
    cache = {}
    # Finish the cheap all-condition baseline first, irrespective of CLI order.
    # A cost cap in a long continuation must not leave most conditions unscored.
    settings = sorted({(c["zone"], c["costs"], c["gamma"]) for c in conds.values()})
    for reference in (ref for ref in REFERENCES if ref in a.references):
        for zone, costs, gamma in settings:
            if (zone, costs) not in cache:
                cache[zone, costs] = empirical_payoffs(zone, costs)
            mag, cost, sigma, source = cache[zone, costs]
            group = f"{zone}_{costs}_g{gamma:g}_{reference}".replace(".", "p")
            horizon, tail = horizon_for(reference, gamma, float((mag + cost).max()), sigma,
                                        a.tail_bound_pct, a.max_horizon)
            spec = dict(group=group, zone=zone, costs=costs, gamma=gamma, reference=reference,
                        paths=a.paths, seed=a.seed, horizon=horizon,
                        script_sha256=sha(__file__), **source)
            path = outdir / f"exploratory_checkpoint_{group}.npz"
            means, seconds, counts, steps, active = [], [], [], [], []
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if json.loads(str(saved["spec"])) != spec:
                        raise SystemExit(f"Incompatible checkpoint: {path}; use a fresh --out folder.")
                    means, seconds, counts, steps, active = [saved[k].tolist() for k in
                                                            ("means", "seconds", "counts", "steps", "active")]
            group_seed = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16)
            for batch in range(len(means), a.batches):
                if time.perf_counter() - started >= a.max_wall_seconds:
                    complete = False
                    break
                rng = np.random.default_rng([a.seed, group_seed, batch])
                before = time.perf_counter()
                estimate, count, step, survival = batch_rollout(mag, cost, gamma, reference,
                                                               horizon, a.paths, rng)
                means.append(estimate.tolist())
                seconds.append(time.perf_counter() - before)
                counts.append(count)
                steps.append(step)
                active.append(survival)
                checkpoint(path, spec, means, seconds, counts, steps, active)
                print(f"{group} batch {batch + 1}/{a.batches}: {seconds[-1]:.2f}s, {count} transitions", flush=True)
            if len(means) >= 2:
                selected = [c for c in conds.values() if (c["zone"], c["costs"], c["gamma"]) == (zone, costs, gamma)]
                rows.extend(summarize_group(spec, means, selected, sigma, tail,
                                            np.random.default_rng([a.seed, group_seed, 991]), a.bootstrap))
            groups.append(dict(**spec, completed_batches=len(means), requested_batches=a.batches,
                               simulation_seconds=float(sum(seconds)), simulated_branch_transitions=int(sum(counts)),
                               max_executed_steps=max(steps, default=0),
                               mean_unresolved_path_fraction=float(np.mean(active)) if active else None,
                               deterministic_tail_bound_pct=tail,
                               tail_target_met=tail <= a.tail_bound_pct * 1.000001))
            write_csv(outdir / "exploratory_indices.csv", rows)
            if not complete:
                break
        if not complete:
            break
    summary = compare_and_transfer(rows, outdir)
    summary.update(status="complete" if complete else "wall_time_cap_partial_resumable",
                   elapsed_wall_seconds=time.perf_counter() - started, groups=groups,
                   total_simulated_branch_transitions=sum(g["simulated_branch_transitions"] for g in groups),
                   noise="Exact full empirical TRAIN fair-sign mixture standard deviation; not binned DP noise.",
                   execution_order="All flat groups, then all primary threshold1 groups, then all max16 groups; no outcome-based reference selection.",
                   source_revision_caveat="Rows marked REVISED_J_PANEL use the current local panel whereas registered J agents used the original panel. A whole-panel hash mismatch can confound comparisons with stored DP/calibration outcomes; it does not quantify TRAIN-payoff changes. See THIRD_PARTY_LICENSES.md, eSett data.",
                   uncertainty="Batch bootstrap Monte Carlo only; fixed market sample; percentile intervals plus deterministic truncation bound.",
                   scope_exclusions={"N": "Ledger/time or zombie-age state and different terminal reward timing: no comparable equity-only index or transferred threshold computed.",
                                     "E/G": "Equity-scaled sizing or gym-mtsim actions/states/payoff units differ; no numeric forced mapping."},
                   reference_policy_caveat="Q_pi is not Q*: even a fully booked control may have a positive contrast when taking risk once avoids later fees under a poor continuation policy. Report rather than censor this mismatch.",
                   dp_input_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                                    [LIAB / "results_k/action_gap.csv", ACCOUNTING / "study_o/dp_o_DK1.json",
                                     ACCOUNTING / "study_o/dp_o_DK2.json"]})
    atomic_json(outdir / "exploratory_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "elapsed_wall_seconds", "total_simulated_branch_transitions")}), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
