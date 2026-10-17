"""Fixed exploratory unbounded-law validation; no learner or old study inputs."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "unbounded_outputs"
SHAPES, COST_LAWS, ACCOUNTS, ALPHAS = (1.5, 2.5), ("fixed", "proportional"), ("floor", "terminal"), (.8, .95)
CASES = tuple(itertools.product(SHAPES, COST_LAWS, ACCOUNTS, ALPHAS))
INITIAL, HORIZON, GAMMA = 2., 32, .99
NSEARCH, NCONFIRM, BATCH = 4096, 65536, 4096
CAPS = (1., 4., 16., 64., 256.)
DELTA = .05 / (2 * len(CASES))
WEIGHTS = GAMMA ** np.arange(HORIZON)
TERMINAL_WEIGHT, WEIGHT_SUM = float(WEIGHTS[-1]), float(WEIGHTS.sum())


class CpuBudgetExceeded(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ParetoOracle:
    """Exact conditional first moments for a unit trade under the declared law."""
    def __init__(self, shape, cost_law):
        self.shape, self.cost_law = float(shape), cost_law
        self.mean_z = shape / (shape - 1.)
        self.fixed_fee = .1 if cost_law == "fixed" else 0.
        self.rate = .05 if cost_law == "proportional" else 0.
        self.plus_scale, self.loss_scale = 1.-self.rate, 1.+self.rate
        self.mean_cost = self.fixed_fee + self.rate*self.mean_z
        self.h0 = float(.5*self.loss_scale*self.call(np.array([-self.fixed_fee/self.loss_scale]))[0])
        self.rho = self.h0 / self.mean_cost
        self.alpha_star = 1.-1./self.rho

    def call(self, threshold):
        """E[(Z-k)+], including k<=1; no finite-support approximation."""
        k = np.asarray(threshold, dtype=float)
        answer = np.empty_like(k)
        linear = k <= 1.
        answer[linear] = self.mean_z-k[linear]
        answer[~linear] = np.exp((1.-self.shape)*np.log(k[~linear]))/(self.shape-1.)
        return answer

    def put(self, threshold):
        """E[(k-Z)+]; the identity is used only in deterministic oracle checks."""
        k = np.asarray(threshold, dtype=float)
        answer = np.zeros_like(k)
        positive = k > 1.
        answer[positive] = np.maximum(self.call(k[positive])+k[positive]-self.mean_z, 0.)
        return answer

    def affine_positive_part(self, scale, intercept):
        """Analytic E[(scale*Z+intercept)+] for either sign of scale."""
        b = np.asarray(intercept, dtype=float)
        if scale > 0:
            return scale*self.call(-b/scale)
        if scale < 0:
            return (-scale)*self.put(b/(-scale))
        return np.maximum(b, 0.)

    def generic_h(self, ledger):
        n = np.asarray(ledger, dtype=float)
        return .5*(self.affine_positive_part(-self.plus_scale, self.fixed_fee-n)
                    + self.affine_positive_part(self.loss_scale, self.fixed_fee-n))

    def floor_h(self, equity):
        e = np.asarray(equity, dtype=float)
        assert np.all(e >= 0.)
        # The positive-gross branch cannot cause a loss from nonnegative equity
        # for the fixed parameters (.1<1 and 0<rate<1).
        return .5*self.loss_scale*self.call((e-self.fixed_fee)/self.loss_scale)

    def debt_delta(self, ledger):
        """Stable E[d(n+X)]-d(n), avoiding cancellation at large negative n."""
        n = np.asarray(ledger, dtype=float)
        answer = np.empty_like(n)
        solvent = n >= 0.
        answer[solvent] = self.floor_h(n[solvent])
        answer[~solvent] = (self.mean_cost+.5*self.plus_scale
                           *self.call((self.fixed_fee-n[~solvent])/self.plus_scale))
        return answer

    def predictable_range(self, accounting, alpha):
        initial_curve = float(self.floor_h(np.array([INITIAL]))[0])
        first_weight = 1. if accounting == "floor" else TERMINAL_WEIGHT
        first = (1-alpha)*first_weight*initial_curve-self.mean_cost
        low = first-self.mean_cost*float(WEIGHTS[1:].sum())
        if accounting == "floor":
            high = first+float(WEIGHTS[1:].sum())*max(0., (1-alpha)*self.h0-self.mean_cost)
        else:
            high = first+float(np.maximum((1-alpha)*TERMINAL_WEIGHT*self.h0
                                          - WEIGHTS[1:]*self.mean_cost, 0.).sum())
        return float(low), float(high), float(first)


def deterministic_oracle_checks():
    records = []
    for shape, law in itertools.product(SHAPES, COST_LAWS):
        oracle = ParetoOracle(shape, law)
        p = Fraction(str(shape))
        m = p/(p-1)
        c = Fraction(1, 10) if law == "fixed" else Fraction(1, 20)*m
        h0 = ((m+c)/2 if law == "fixed" else Fraction(21, 40)*m)
        threshold = 1-c/h0
        assert abs(oracle.mean_cost-float(c)) < 1e-13
        assert abs(oracle.h0-float(h0)) < 1e-13
        assert abs(oracle.alpha_star-float(threshold)) < 1e-13
        assert .95 > float(threshold)
        grid = np.array([-10., -2., -.1, 0., .1, 2., 10.])
        stable = oracle.debt_delta(grid)
        direct = oracle.generic_h(grid)-np.maximum(-grid, 0.)
        error = float(np.max(np.abs(stable-direct)))
        assert error < 2e-13
        assert np.all(stable >= -1e-13) and np.all(stable <= oracle.h0+1e-13)
        assert np.all(stable[grid < 0] >= oracle.mean_cost-1e-13)
        extreme = oracle.debt_delta(np.array([-1e300, -1e100, 0., 1e100, 1e300]))
        assert np.all(np.isfinite(extreme))
        assert np.all(extreme >= 0.) and np.all(extreme <= oracle.h0+1e-13)
        independent_h2 = (.5*oracle.loss_scale**shape*(INITIAL-oracle.fixed_fee)**(1-shape)/(shape-1))
        h2 = float(oracle.floor_h(np.array([INITIAL]))[0])
        assert abs(h2-independent_h2) < 1e-13
        if law == "fixed" and shape == 1.5:
            assert abs(.2*h2-.1-.04509525002200229) < 1e-13
        if law == "fixed" and shape == 2.5:
            assert abs(.2*h2-.1-(-.07454469297859608)) < 1e-13
        records.append(dict(shape=shape, cost_law=law, mean_magnitude_exact=str(m),
                            mean_cost_exact=str(c), h0_exact=str(h0),
                            uniform_repair_alpha_exact=str(threshold),
                            mean_cost=float(c), h0=float(h0), uniform_repair_alpha=float(threshold),
                            one_step_h_initial=h2,
                            one_step_expected_charged_alpha0p8=.2*h2-float(c),
                            stable_vs_generic_grid_error=error,
                            extreme_ledger_values=extreme.tolist(), all_checks_passed=True))
    return records


def rollout(oracle, accounting, alpha, n, seed_or_rng, cpu_deadline):
    rng = np.random.default_rng(seed_or_rng) if isinstance(seed_or_rng, int) else seed_or_rng
    ledger = np.full(n, INITIAL)
    predictable_a = np.zeros(n)
    predictable_b = np.zeros(n)
    actual_a = np.zeros(n)
    actual_c = np.zeros(n)
    gross_sum = np.zeros(n)
    active_steps = oracle_points = oracle_vector_calls = 0
    oracle_seconds = 0.
    for t in range(HORIZON):
        if time.process_time() > cpu_deadline:
            raise CpuBudgetExceeded()
        active = np.ones(n, dtype=bool) if t == 0 else ledger <= INITIAL
        points = int(active.sum())
        active_steps += points
        if points:
            before = time.perf_counter()
            curve = (oracle.floor_h(ledger[active]) if accounting == "floor"
                     else oracle.debt_delta(ledger[active]))
            oracle_seconds += time.perf_counter()-before
            assert np.all(curve >= -1e-12) and np.all(curve <= oracle.h0+1e-12)
            if accounting == "terminal":
                assert np.all(curve[ledger[active] < 0] >= oracle.mean_cost-1e-12)
            predictable_a[active] += ((WEIGHTS[t] if accounting == "floor" else TERMINAL_WEIGHT)*curve)
            predictable_b[active] += WEIGHTS[t]*oracle.mean_cost
            oracle_points += points
            oracle_vector_calls += 1
        # Draw all global slots, including flat ones; this makes generation and
        # logical/active accounting explicit and avoids state-dependent RNG use.
        z = 1.+rng.pareto(oracle.shape, n)
        sign = 2*rng.integers(2, size=n, dtype=np.int8)-1
        assert np.all(np.isfinite(z))
        fee = (oracle.fixed_fee+oracle.rate*z)*active
        gross = sign*z*active
        actual_c += WEIGHTS[t]*fee
        gross_sum += WEIGHTS[t]*gross
        next_ledger = ledger+gross-fee
        if accounting == "floor":
            injection = np.maximum(-next_ledger, 0.)
            actual_a += WEIGHTS[t]*injection
            ledger = np.maximum(next_ledger, 0.)
        else:
            ledger = next_ledger
    if accounting == "terminal":
        actual_a = TERMINAL_WEIGHT*np.maximum(-ledger, 0.)
    compensator = (1-alpha)*predictable_a-predictable_b
    mean_cost_residual = (1-alpha)*actual_a-predictable_b
    actual_residual = (1-alpha)*actual_a-actual_c
    raw = gross_sum+actual_residual
    counts = dict(logical_transitions=n*HORIZON, global_magnitude_draws=n*HORIZON,
                  active_transitions=active_steps, conditional_curve_scalar_evaluations=oracle_points,
                  conditional_curve_vector_calls=oracle_vector_calls,
                  expected_cost_scalar_lookups=oracle_points, oracle_evaluation_seconds=oracle_seconds)
    values = dict(compensator=compensator, mean_cost_residual=mean_cost_residual,
                  raw=raw, actual_cost=actual_c, predictable_cost=predictable_b,
                  actual_forgiveness=actual_a, predictable_forgiveness=predictable_a,
                  booked=gross_sum-actual_c, gross=gross_sum)
    assert all(np.all(np.isfinite(v)) for v in values.values())
    return values, counts


def lower_bound(values, lower, upper, delta=DELTA, projected_n=None):
    x = np.asarray(values, dtype=float)
    assert x.min() >= lower-1e-10 and x.max() <= upper+1e-10
    n = len(x) if projected_n is None else projected_n
    variance, mean = float(x.var(ddof=1)), float(x.mean())
    log = float(np.log(2/delta))
    radius = float(np.sqrt(2*variance*log/n)+7*(upper-lower)*log/(3*(n-1)))
    return dict(n_observed=len(x), n_in_bound=n, mean=mean, sample_variance=variance,
                support_lower=float(lower), support_upper=float(upper), delta=delta,
                radius=radius, lower=mean-radius,
                observed_min=float(x.min()), observed_max=float(x.max()),
                projected_search_score_only=projected_n is not None)


def add_counts(total, extra):
    for name, value in extra.items():
        total[name] = total.get(name, 0)+value


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "results.json").exists():
        raise RuntimeError("Preserve existing complete/partial outputs; no automatic rerun")
    cpu0, wall0 = time.process_time(), time.perf_counter()
    cpu_deadline = cpu0+115.
    checks = deterministic_oracle_checks()  # BEFORE every random draw.
    rows, totals, status = [], {}, "COMPLETE"
    for case_id, (shape, law, accounting, alpha) in enumerate(CASES):
        oracle = ParetoOracle(shape, law)
        low, high, first = oracle.predictable_range(accounting, alpha)
        baseline_low = -oracle.mean_cost*WEIGHT_SUM
        search_seed, confirm_seed = 202610161100+case_id, 202610161200+case_id
        case_cpu0, case_wall0 = time.process_time(), time.perf_counter()
        try:
            search, search_counts = rollout(oracle, accounting, alpha, NSEARCH, search_seed, cpu_deadline)
            add_counts(totals, search_counts)
            cap_scores = []
            for cap in CAPS:
                scoring = lower_bound(np.minimum(search["mean_cost_residual"], cap), baseline_low, cap,
                                      projected_n=NCONFIRM)
                cap_scores.append(dict(cap=cap, score=scoring))
            chosen = max(cap_scores, key=lambda r: r["score"]["lower"])
            cap = chosen["cap"]
            buffers = {name: np.empty(NCONFIRM) for name in ("compensator", "capped_mean_cost_residual")}
            descriptive = {name: 0. for name in search}
            rng, completed, confirm_counts = np.random.default_rng(confirm_seed), 0, {}
            for batch in range(NCONFIRM//BATCH):
                values, counts = rollout(oracle, accounting, alpha, BATCH, rng, cpu_deadline)
                buffers["compensator"][completed:completed+BATCH] = values["compensator"]
                buffers["capped_mean_cost_residual"][completed:completed+BATCH] = np.minimum(values["mean_cost_residual"], cap)
                for name, vector in values.items():
                    descriptive[name] += float(vector.sum())
                add_counts(confirm_counts, counts)
                add_counts(totals, counts)
                completed += BATCH
            comp = lower_bound(buffers["compensator"], low, high)
            capped = lower_bound(buffers["capped_mean_cost_residual"], baseline_low, cap)
            if alpha >= oracle.alpha_star:
                assert high <= 1e-12
                assert comp["observed_max"] <= 1e-12
            rows.append(dict(case_id=case_id, pareto_shape=shape, cost_law=law,
                accounting=accounting, alpha=alpha, initial_ledger=INITIAL, horizon=HORIZON, gamma=GAMMA,
                unit_trade_gross_support_unbounded=True, actual_cost_support_unbounded=law == "proportional",
                gross_second_moment_finite=shape > 2,
                actual_cost_second_moment_finite=(law == "fixed" or shape > 2),
                mean_cost=oracle.mean_cost, h0=oracle.h0, uniform_repair_alpha=oracle.alpha_star,
                analytically_all_policy_nonpositive=alpha >= oracle.alpha_star,
                fixed_first_compensator_term=first, expected_booked_upper=-oracle.mean_cost,
                n_search=NSEARCH, planned_confirmation_paths=NCONFIRM,
                completed_confirmation_paths=completed, confirmation_batches=NCONFIRM//BATCH,
                search_seed=search_seed, confirmation_seed=confirm_seed,
                selected_residual_cap=cap, all_cap_search_scores=cap_scores,
                search_compensator_statistics=lower_bound(search["compensator"], low, high, projected_n=NCONFIRM),
                uncapped_compensator_bound=comp, capped_mean_cost_residual_bound=capped,
                uncapped_compensator_positive=comp["lower"] > 0,
                capped_mean_cost_residual_positive=capped["lower"] > 0,
                paired_sample_variance_ratio_comp_over_capped=(comp["sample_variance"]/capped["sample_variance"]
                                                              if capped["sample_variance"] > 0 else None),
                descriptive_confirmation_means={name: value/completed for name, value in descriptive.items()},
                search_counts=search_counts, confirmation_counts=confirm_counts,
                cpu_seconds=time.process_time()-case_cpu0, wall_seconds=time.perf_counter()-case_wall0,
                raw_bounded_empirical_bernstein="unsupported: alpha>0 gives an unbounded negative charged-reward tail",
                inference_scope="Two fixed-alpha one-sided bounds only; no new confidence interval or pairwise mean test."))
        except CpuBudgetExceeded:
            status = "INCOMPLETE_cpu_guard"
            break
    complete = len(rows) == len(CASES)
    if not complete:
        status = "INCOMPLETE_cpu_guard"
    result = dict(status="EXPLORATORY_"+status, constructed_cases=True,
        planned_cases=len(CASES), completed_cases=len(rows), full_fixed_budgets_complete=complete,
        family_error_budget=.05, allocated_one_sided_events=2*len(CASES), delta_per_bound=DELTA,
        bounds_are_not_paper_wide_coverage=True, one_numerical_thread=True,
        policy="forced unit trade at t0, then unit trade iff observed ledger<=initial; fixed across alpha",
        deterministic_oracle_checks=checks, cases=rows, total_computation_counts=totals,
        family_positive_verdicts_valid=complete,
        uncapped_compensator_positive_count=(sum(r["uncapped_compensator_positive"] for r in rows) if complete else None),
        capped_mean_cost_residual_positive_count=(sum(r["capped_mean_cost_residual_positive"] for r in rows) if complete else None),
        cpu_seconds=time.process_time()-cpu0, wall_seconds=time.perf_counter()-wall0,
        script_sha256=sha(__file__), plan_sha256=sha(HERE/"UNBOUNDED_PLAN.md"),
        old_artifacts_read=False, old_artifacts_modified=False, R_inputs_read=False,
        learners_trained=0, dp_solved=False,
        interpretation="Constructed-law validation with an exact conditional-moment oracle. No universal variance, optimizer, acquisition, or market-generalization claim; failed bounds are abstentions.")
    (OUTPUT/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = [dict(case_id=r["case_id"], shape=r["pareto_shape"], cost_law=r["cost_law"],
                    accounting=r["accounting"], alpha=r["alpha"], mean_cost=r["mean_cost"], h0=r["h0"],
                    repair_alpha=r["uniform_repair_alpha"], analytic_nonpositive=r["analytically_all_policy_nonpositive"],
                    cap=r["selected_residual_cap"], completed_paths=r["completed_confirmation_paths"],
                    comp_mean=r["uncapped_compensator_bound"]["mean"],
                    comp_variance=r["uncapped_compensator_bound"]["sample_variance"],
                    comp_lower=r["uncapped_compensator_bound"]["lower"],
                    comp_support_lower=r["uncapped_compensator_bound"]["support_lower"],
                    comp_support_upper=r["uncapped_compensator_bound"]["support_upper"],
                    capped_mean=r["capped_mean_cost_residual_bound"]["mean"],
                    capped_variance=r["capped_mean_cost_residual_bound"]["sample_variance"],
                    capped_lower=r["capped_mean_cost_residual_bound"]["lower"],
                    comp_positive=(r["uncapped_compensator_positive"] if complete else None),
                    capped_positive=(r["capped_mean_cost_residual_positive"] if complete else None)) for r in rows]
    if summary:
        with (OUTPUT/"case_summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
    printed = dict(status=result["status"], complete=complete, cases=len(rows),
                   comp_positive=result["uncapped_compensator_positive_count"],
                   capped_positive=result["capped_mean_cost_residual_positive_count"],
                   cpu_seconds=result["cpu_seconds"], wall_seconds=result["wall_seconds"],
                   rows=summary)
    (OUTPUT/"run_summary.log").write_text(json.dumps(printed, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(printed, indent=2))


if __name__ == "__main__":
    main()
