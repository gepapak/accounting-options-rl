"""Source-only synthetic checks; no v2/native endpoint, import, fit or freeze.

Constant toy records have independently calculated zero-variance EB formulas.
They test arithmetic/schema and fail-closed behavior, not scientific outcomes.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_certificate(value, n, lo, hi, pad=1e-6):
    # Closed-form constant sample: its unbiased sample variance is exactly zero.
    delta = .05 / 340
    lower, upper = lo - pad, hi + pad
    radius = 7 * (upper - lower) * math.log(2 / delta) / (3 * (n - 1))
    return dict(n=n, mean=value, variance=0., radius=radius, lower=value - radius - pad,
                support_lower=lower, support_upper=upper, delta=delta, numerical_allowance=pad)


def synthetic_payload():
    seeds = (7, 42, 123, 2025, 3007, 5001, 8102, 9005, 10001, 11202)
    base = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
    extra = ("cash_capped_raw", "cash_ridge_control", "cash_mlp_control", "cash_local_control")
    cases, paths, selections, charged, costs = [], [], [], [], []
    counts = {"0.0": {m: 0 for m in base + extra}, "1.0": {m: 0 for m in base}}
    for cid in range(20):
        cost = 1 + cid / 100
        cases.append(dict(case_id=cid, policy_index=cid, policy_family="plain_mappo" if cid < 10 else "feasible_mappo",
            checkpoint_seed=seeds[cid % 10], profile="joint_fault", source_run001_case_id=2 * cid + 1,
            cash_fraction=1e-4, horizon=96, burn_in=384,
            bounds=dict(discounted_cost_bound=1.5, discounted_gross_bound=2.,
                        unavoidable_discounted_access_cost=.5, cash_inflow_bound=.75)))
        for phase, n in (("fit", 32), ("tune", 32), ("confirm", 256)):
            for episode in range(n):
                row = dict(case_id=cid, phase=phase, episode_id=episode, context_id=episode % 8,
                    gross=.5, cost=cost, forgiveness=2., paired_forgiveness=1.5, maturity_coins=[1, -1],
                    excluded_old_gross=.25, excluded_old_forgiveness=0., excluded_old_paired_forgiveness=0.,
                    excluded_old_maturity_events=1)
                if phase == "confirm":
                    row.update(delta_ridge=[1., 1.], delta_mlp=[2., 0.], delta_cash_local=[.5, .5],
                               delta_cash_ridge=[1.5, .5], delta_cash_mlp=[2.5, .5])
                paths.append(row)
        for alpha in (0., 1.):
            for method in base + (extra if alpha == 0 else ()):
                learned = method.endswith("_control")
                choice = dict(cap=None if method == "raw" else 1000., beta=.25 if learned else 0., budget=62.5 if learned else 0.)
                if method in ("raw", "capped_raw") or method.startswith("cash_"):
                    mean = .5 - cost + (1 - alpha) * 2
                elif method == "capped_residual":
                    mean = (1 - alpha) * 2 - cost
                else:
                    mean = (1 - alpha) * 1.5 - cost
                mean -= .5 if method in ("mlp_control", "cash_mlp_control") else .25 if method == "cash_ridge_control" else 0.
                if method in ("raw", "capped_raw"):
                    lo, hi = -3.5, 2.
                elif method.startswith("cash_"):
                    lo, hi = -.75 - choice["budget"], 2. + choice["budget"]
                else:
                    lo = -1.5 - choice["budget"]
                    hi = (1 - alpha) * (2. if method == "capped_residual" else 1.) + choice["budget"]
                cert = reference_certificate(mean, 320 if method == "raw" else 256, lo, hi)
                selections.append(dict(case_id=cid, alpha=alpha, method=method, choice=choice))
                charged.append(dict(case_id=cid, profile="joint_fault", alpha=alpha, method=method, choice=choice,
                    certificate=cert, positive=cert["lower"] > 0, confirmation_mean=mean, confirmation_variance=0.))
                counts[str(alpha)][method] += int(cert["lower"] > 0)
        cert = reference_certificate(cost, 320, 0., 1.5)
        costs.append(dict(case_id=cid, profile="joint_fault", certificate=cert, positive=cert["lower"] > 0,
            analytic_unavoidable_cost_lower=.5, booked_expectation_upper=-max(cert["lower"], .5)))
    result = dict(schema="sample_only_joint_fault_public_v2", multiplicity=dict(family_limits=340,
        total_delta=.05, delta_per_limit=.05 / 340), method_counts=counts,
        profile_counts={"joint_fault": deepcopy(counts)}, positive_cost_limits=20,
        charged_results=charged, cost_results=costs)
    return paths, cases, selections, result


def run():
    replay = load("synthetic_joint_fault_replay", "reproduce_public_joint_fault.py")
    exporter = load("synthetic_joint_fault_export", "public_export_joint_fault.py")
    payload = synthetic_payload()
    report = replay.verify_payload(*payload)
    assert report["bounds_recomputed"] == 340 and report["whole_paths"] == 6400
    assert report["positive_cost_limits"] == report["strict_booked_harm_cases"] == 20
    assert report["method_counts"]["0.0"]["cash_capped_raw"] == 20
    assert report["method_counts"]["0.0"]["cash_ridge_control"] == 0
    assert len(replay.methods_for_alpha(0.)) == 10 and len(replay.methods_for_alpha(1.)) == 6
    assert replay.sequential_control([10., -10., 10.], [1, 1, -1], 1., 3.) == 3.
    assert replay.sequential_control([10., -10.], [1, -1], 0., 3.) == 0.
    assert replay.sequential_control([1., 1.], [1, -1], 1., 3.) == 0.
    rejected = []

    def must_reject(name, function):
        try:
            function()
        except (AssertionError, ValueError, KeyError):
            rejected.append(name)
        else:
            raise AssertionError("Tamper unexpectedly passed: " + name)

    altered = deepcopy(payload)
    altered[0][0]["cost"] = -1.
    must_reject("negative_cost", lambda: replay.verify_payload(*altered))
    altered = deepcopy(payload)
    altered[0][0]["actor_observation"] = [1., 2.]
    must_reject("private_field_export", lambda: replay.verify_payload(*altered))
    altered = deepcopy(payload)
    next(r for r in altered[2] if r["method"] == "cash_local_control")["alpha"] = 1.
    must_reject("cash_control_at_alpha1", lambda: replay.verify_payload(*altered))
    altered = deepcopy(payload)
    altered[3]["charged_results"][0]["certificate"]["radius"] += 1.
    must_reject("wrong_saved_radius", lambda: replay.verify_payload(*altered))
    altered = deepcopy(payload)
    altered[3]["method_counts"]["0.0"]["raw"] -= 1
    must_reject("wrong_aggregate_count", lambda: replay.verify_payload(*altered))
    for path in ("C:/private/input", "../input", "/home/private/input"):
        must_reject("private_or_escaping_path:" + path, lambda p=path: exporter.simple_identifier(p))
    assert exporter.numeric_work({"seconds": 1.25, "scope": "not copied", "fit": {"rows": 10}}) == {"seconds": 1.25, "fit": {"rows": 10}}
    # Isolated empty source-test directory: the exporter must fail on existence
    # gates before attempting any outcome read. Never point this test at v2 data.
    validation = HERE / "validation"
    validation.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthetic_missing_gate_", dir=validation) as temporary:
        assert Path(temporary).resolve().is_relative_to(validation.resolve())
        original_here, original_read = exporter.HERE, exporter.read
        exporter.HERE = Path(temporary)

        def forbidden_read(_path):
            raise RuntimeError("Early endpoint read")

        exporter.read = forbidden_read
        try:
            must_reject("missing_gates_before_endpoint_reads", lambda: exporter._private_gate("run_synthetic"))
        finally:
            exporter.HERE, exporter.read = original_here, original_read
    forbidden = [name for name in sys.modules if name == "numpy" or name == "torch" or name.startswith("sklearn") or "native_engine_adapter" in name]
    assert not forbidden
    return dict(status="PASS SYNTHETIC SOURCE-ONLY CHECK", scope="constant toy records, not native/statistical outcomes",
        synthetic_whole_paths=6400, synthetic_arithmetic_limits_checked=340, arithmetic_checks=report["checks"],
        maximum_synthetic_arithmetic_difference=report["maximum_absolute_arithmetic_difference"], rejected_cases=rejected,
        native_simulator_calls=0, native_endpoint_reads=0, actor_inference=0, model_fitting=0, source_freeze_created=False,
        source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("reproduce_public_joint_fault.py", "public_export_joint_fault.py", Path(__file__).name)})


if __name__ == "__main__":
    result = run()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = HERE / "validation" / ("PUBLIC_ARITHMETIC_SYNTHETIC_CHECK_" + stamp + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], synthetic_arithmetic_limits_checked=340,
                         rejected_cases=len(result["rejected_cases"]), output=str(path))))
