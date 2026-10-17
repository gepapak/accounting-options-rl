"""Descriptive tables after BOTH private audits and public replays pass.

No new inference test, native query, actor inference or fitting. Outputs are local
LaTeX fragments/CSV/JSON; manuscript files and frozen evidence are never edited.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
METHODS = ("raw", "capped_raw", "capped_residual", "paired", "ridge_control", "mlp_control")
LABELS = dict(raw="RAW", capped_raw="Capped RAW", capped_residual="Realized residual",
              paired="Sign-paired", ridge_control="Ridge control", mlp_control="MLP control")
TARGETS = ("primary_alpha0", "primary_alpha1", "auxiliary_alpha0")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def public_replay(directory, filename, name):
    manifest = read(directory/"MANIFEST.json")
    require(manifest.get("complete") is True, "Completed public bundle required")
    require(sha(directory/filename) == manifest["file_sha256"][filename], "Public reproducer source hash")
    spec = importlib.util.spec_from_file_location(name, directory/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_directory(directory)
    require(report["status"] == "PASS", "Independent public arithmetic replay")
    return report


def inputs(run_id):
    primary_candidates = [HERE/"validation"/("NATIVE_VERIFICATION_"+run_id+s+".json") for s in ("_critics", "")]
    primary_path = next((p for p in primary_candidates if p.is_file()), None)
    auxiliary_path = HERE/"validation"/("NEW_CONTRACT_VERIFICATION_"+run_id+".json")
    primary_public, auxiliary_public = HERE/"results"/run_id, HERE/"results"/(run_id+"_new_contract")
    require(primary_path is not None and auxiliary_path.is_file() and
            (primary_public/"MANIFEST.json").is_file() and (auxiliary_public/"MANIFEST.json").is_file(),
            "NOT_READY: both private PASS reports and completed public bundles precede endpoint reads")
    primary_verification, auxiliary_verification = read(primary_path), read(auxiliary_path)
    require(primary_verification.get("status") == auxiliary_verification.get("status") == "PASS" and
            primary_verification.get("bounds_checked") == 520 and auxiliary_verification.get("bounds_checked") == 240,
            "Both complete private verification families must pass")
    primary_provenance, auxiliary_provenance = read(primary_public/"provenance.json"), read(auxiliary_public/"provenance.json")
    require(primary_provenance["private_result_sha256"] == primary_verification["result_sha256"] and
            auxiliary_provenance["private_auxiliary_result_sha256"] == auxiliary_verification["result_sha256"],
            "Public/private result bindings")
    primary_replay = public_replay(primary_public, "reproduce_public_native_audit.py", "report_primary_replay")
    auxiliary_replay = public_replay(auxiliary_public, "reproduce_public_new_contract.py", "report_auxiliary_replay")
    primary, auxiliary = read(primary_public/"results.json"), read(auxiliary_public/"results.json")
    cases = read(primary_public/"cases_and_bounds.json")
    require(len(cases) == 40 and {c["case_id"] for c in cases} == set(range(40)), "All original cases")
    root = HERE/"execution"/run_id
    rich_path = HERE/("native_results_"+run_id+".json")
    require(sha(rich_path) == primary_verification["result_sha256"], "Original rich result remains unchanged")
    private_auxiliary_paths = HERE/"auxiliary_new_contract"/run_id/"path_summaries.jsonl.gz"
    require(sha(private_auxiliary_paths) == auxiliary_verification["paths_sha256"], "Verified auxiliary component summaries")
    provenance = dict(primary_private_verification_sha256=sha(primary_path), auxiliary_private_verification_sha256=sha(auxiliary_path),
        primary_public_manifest_sha256=sha(primary_public/"MANIFEST.json"), auxiliary_public_manifest_sha256=sha(auxiliary_public/"MANIFEST.json"),
        rich_primary_result_sha256=sha(rich_path), private_auxiliary_paths_sha256=sha(private_auxiliary_paths),
        report_tool_sha256=sha(__file__))
    return primary, auxiliary, cases, root, read(rich_path), load_rows(private_auxiliary_paths), provenance, (primary_replay, auxiliary_replay)


def descriptive_comparison(rows, cases):
    matched, ratios, choices = [], [], []
    for target in TARGETS:
        for profile in ("strict", "vulnerable"):
            ids = [case["case_id"] for case in cases if case["profile"] == profile]
            require(len(ids) == 20, "Twenty fixed policies per guard profile")
            for method in ("ridge_control", "mlp_control"):
                record = dict(target=target, profile=profile, learned_method=method, n_cases=20,
                    certificate_wins=0, certificate_losses=0, certificate_ties=0,
                    both_certified=0, both_abstain=0, lower_bound_higher=0, lower_bound_lower=0, lower_bound_ties=0)
                differences, finite_ratios, statuses, budget_values, beta_values = [], [], [], [], []
                for cid in ids:
                    learned, pairing = rows[target, cid, method], rows[target, cid, "paired"]
                    lp, pp = bool(learned["positive"]), bool(pairing["positive"])
                    record["certificate_wins" if lp and not pp else "certificate_losses" if pp and not lp else "certificate_ties"] += 1
                    record["both_certified"] += lp and pp
                    record["both_abstain"] += not lp and not pp
                    ll, pl = learned["certificate"]["lower"], pairing["certificate"]["lower"]
                    difference, tolerance = ll-pl, max(1e-6, 1e-12*max(abs(ll), abs(pl)))
                    record["lower_bound_higher" if difference > tolerance else "lower_bound_lower" if difference < -tolerance else "lower_bound_ties"] += 1
                    differences.append(difference)
                    lv = learned["confirmation_variance"] if target.startswith("primary") else learned["certificate"]["variance"]
                    pv = pairing["confirmation_variance"] if target.startswith("primary") else pairing["certificate"]["variance"]
                    require(lv >= 0 and pv >= 0, "Nonnegative common-CONFIRM variances")
                    status = "finite" if pv > 0 else "both_zero" if lv == 0 else "zero_pair_positive_learned"
                    ratio = lv/pv if pv > 0 else None
                    if ratio is not None:
                        finite_ratios.append(ratio)
                    statuses.append(status)
                    ratios.append(dict(target=target, profile=profile, case_id=cid, learned_method=method,
                        learned_confirmation_variance=lv, pairing_confirmation_variance=pv, ratio=ratio, status=status))
                    budget_values.append(float(learned["choice"]["budget"]))
                    beta_values.append(float(learned["choice"]["beta"]))
                record.update(lower_bound_difference_median=statistics.median(differences),
                    lower_bound_difference_min=min(differences), lower_bound_difference_max=max(differences),
                    finite_variance_ratios=len(finite_ratios), variance_ratio_median=statistics.median(finite_ratios) if finite_ratios else None,
                    variance_ratio_min=min(finite_ratios) if finite_ratios else None,
                    variance_ratio_max=max(finite_ratios) if finite_ratios else None,
                    variance_both_zero=statuses.count("both_zero"), variance_zero_pair_positive_learned=statuses.count("zero_pair_positive_learned"))
                matched.append(record)
                choices.append(dict(target=target, profile=profile, method=method, n_cases=20,
                    nonzero_budget=sum(x > 0 for x in budget_values), nonzero_beta=sum(x > 0 for x in beta_values),
                    nonzero_both=sum(b > 0 and beta > 0 for b, beta in zip(budget_values, beta_values)),
                    budget_median=statistics.median(budget_values), budget_max=max(budget_values),
                    beta_counts={str(beta): beta_values.count(beta) for beta in sorted(set(beta_values))}))
    return matched, ratios, choices


def activity(cases, root, auxiliary_paths):
    auxiliary = {(r["case_id"], r["episode_id"]): r for r in auxiliary_paths}
    result = []
    for case in cases:
        cid = case["case_id"]
        rows = load_rows(root/f"case_{cid:02d}"/"confirm"/"rows.jsonl")
        require(len(rows) == 256, "Complete native activity cohort")
        mean = lambda values: math.fsum(values)/256
        new_events = [r["maturity_events"]-auxiliary[cid, r["episode_id"]]["excluded_old_maturity_events"] for r in rows]
        require(all(n >= 0 for n in new_events), "Direct old events form a maturity subset")
        result.append(dict(case_id=cid, profile=case["profile"], policy_family=case["policy_family"],
            episodes=256, episodes_with_any_maturity=sum(r["maturity_events"] > 0 for r in rows),
            episodes_with_new_maturity=sum(n > 0 for n in new_events), mean_new_maturities=mean(new_events),
            episodes_with_initial_prepaid_claims=sum(r["initial_pending_prepaid_contracts"] > 0 for r in rows),
            mean_initial_prepaid_claims=mean(r["initial_pending_prepaid_contracts"] for r in rows),
            mean_nonzero_actor_request_steps=mean(r["native_activity"]["nonzero_investor_action_steps"] for r in rows),
            terminal_native_trading_enabled=sum(r["native_activity"]["native_trading_enabled"] for r in rows),
            terminal_native_margin_active=sum(r["native_activity"]["native_margin_active"] for r in rows),
            max_terminal_open_contracts=max(r["terminal_open_contracts"] for r in rows),
            mean_primary_forgiveness=mean(r["forgiveness"] for r in rows),
            mean_excluded_direct_old_forgiveness=mean(auxiliary[cid, r["episode_id"]]["excluded_old_forgiveness"] for r in rows)))
    return result


def number(value):
    if value is None:
        return "--"
    text = format(float(value), ".4g")
    if "e" in text:
        mantissa, exponent = text.split("e")
        return r"\("+mantissa+r"\times10^{"+str(int(exponent))+r"}\)"
    return text


def tabular(headers, body, alignment=None):
    alignment = alignment or "l"+"r"*(len(headers)-1)
    row_end = " "+chr(92)*2
    return "\\begin{tabular}{"+alignment+"}\n\\toprule\n"+" & ".join(headers)+row_end+"\n\\midrule\n"+"\n".join(" & ".join(map(str, row))+row_end for row in body)+"\n\\bottomrule\n\\end{tabular}\n"


def write_text(path, content):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def csv_file(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def generate(run_id):
    primary, auxiliary, cases, root, rich, aux_paths, provenance, replays = inputs(run_id)
    lookup, lower_rows = {}, []
    for target in TARGETS:
        source = auxiliary["charged_results"] if target.startswith("auxiliary") else [r for r in primary["charged_results"] if r["alpha"] == (0. if target.endswith("0") else 1.)]
        require(len(source) == 240, "All case/method rows per target")
        for row in source:
            lookup[target, row["case_id"], row["method"]] = row
            lower_rows.append(dict(target=target, case_id=row["case_id"], profile=row["profile"], method=row["method"],
                n=row["certificate"]["n"], mean=row["certificate"]["mean"], variance=row["certificate"]["variance"],
                radius=row["certificate"]["radius"], lower=row["certificate"]["lower"], positive=row["positive"],
                cap=row["choice"]["cap"], beta=row["choice"]["beta"], budget=row["choice"]["budget"]))
    for cid in range(40):
        for method in METHODS:
            require(lookup["primary_alpha0", cid, method]["choice"] == lookup["auxiliary_alpha0", cid, method]["choice"],
                    "Auxiliary choices reused without retuning")
    matched, ratios, choice_summaries = descriptive_comparison(lookup, cases)
    native_activity = activity(cases, root, aux_paths)
    counts = [{"target": target, "profile": profile, "method": method,
               "certified": sum(lookup[target, c["case_id"], method]["positive"] for c in cases if c["profile"] == profile), "cases":20}
              for target in TARGETS for profile in ("strict", "vulnerable") for method in METHODS]
    model = rich["model_and_selection"]["model_metadata"]
    compute = dict(actual_work=primary["actual_work"], model_fit=dict(n_rows=model["n_rows"], n_features=model["n_features"],
        total_cpu_seconds=model["total_cpu_seconds"], total_wall_seconds=model["total_wall_seconds"],
        preprocessing_cpu_seconds=model["preprocessing_cpu_seconds"],
        ridge_cpu_seconds=model["ridge"]["cpu_seconds"], ridge_wall_seconds=model["ridge"]["wall_seconds"],
        mlp_cpu_seconds=model["mlp"]["cpu_seconds"], mlp_wall_seconds=model["mlp"]["wall_seconds"],
        mlp_iterations=model["mlp"]["n_iter"], mlp_training_samples_seen=model["mlp"]["training_samples_seen"]),
        selection_cpu_seconds=rich["model_and_selection"]["selection_cpu_seconds"],
        selection_wall_seconds=rich["model_and_selection"]["selection_wall_seconds"],
        development_wall_seconds=rich["development_stage"]["wall_seconds"],
        confirmation_wall_seconds=rich["confirmation_stage"]["wall_seconds"],
        additional_parent_tune_critic_predict_calls=160,
        scope="Shared native queries; extra fitting/branch arithmetic measured separately. Worker critic calls are CONFIRM only. Phase/preparationCPU excludes process imports/model loading; no equalCPU or generic query-complexity claim.")
    output = HERE/"report_tables"/run_id
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="COMPLETE DESCRIPTIVE TABLES", source_hashes=provenance,
        family_scopes={"primary":"520 limits, separate95%", "auxiliary":"240 limits, separate95%; no joint95%"},
        counts=counts, matched_learned_pairing=matched, confirmation_variance_ratios=ratios,
        choices=choice_summaries, native_activity=native_activity, compute=compute,
        public_replay_checks={"primary":replays[0]["checks"], "auxiliary":replays[1]["checks"]},
        lower_bound_tie_rule="abs(difference)<=max(1e-6,1e-12*max(abs(bounds))); descriptive only",
        variance_zero_rule="0/0 undefined both_zero; positive/0 undefined zero_pair_positive_learned; zero/positive=0 included; finite summaries exclude undefined cases",
        interpretation="No formal estimator superiority test. Counts and matched labels are descriptive. Actor requests do not prove executed exposure; maturity and new-maturity activity reported separately. Auxiliary retains inherited state effects.")
    write_text(output/"DESCRIPTIVE_RESULTS.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    csv_file(output/"all_lower_bounds.csv", lower_rows)
    csv_file(output/"confirmation_variance_ratios.csv", ratios)
    csv_file(output/"native_activity.csv", native_activity)
    count_map = {(r["target"],r["profile"],r["method"]):r["certified"] for r in counts}
    header = ["Method",r"P0 strict",r"P0 vulnerable",r"P1 strict",r"P1 vulnerable",r"Aux strict",r"Aux vulnerable"]
    body = [[LABELS[method]]+[str(count_map[t,p,method])+"/20" for t in TARGETS for p in ("strict","vulnerable")] for method in METHODS]
    write_text(output/"counts.tex", "% P0/P1: primary alpha0/1,520-family; Aux: alpha0,240-family. Separate95%, no joint95%.\n"+tabular(header,body))
    for target in TARGETS:
        for profile in ("strict", "vulnerable"):
            body = []
            for case in cases:
                if case["profile"] == profile:
                    body.append([f"{case['case_id']:02d}"]+[number(lookup[target,case["case_id"],m]["certificate"]["lower"]) for m in METHODS])
            write_text(output/(target+"_"+profile+"_bounds.tex"), "% Lower confidence limits inDKK, four significant digits; exact values inall_lower_bounds.csv.\n"+
                tabular(["Case","RAW","Cap RAW","Residual","Paired","Ridge","MLP"],body))
    write_text(output/"matched_certification.tex", "% Descriptive paired certification discordance; not a formal superiority test.\n"+
        tabular(["Target/profile","Control","Win","Loss","Tie"],[[r["target"].replace("_",r"\_")+"/"+r["profile"], LABELS[r["learned_method"]],r["certificate_wins"],r["certificate_losses"],r["certificate_ties"]] for r in matched],"llrrr"))
    write_text(output/"variance_ratios.tex", "% Same256CONFIRM paths; undefined positive/0 and0/0 ratios are counted, never replaced by1.\n"+
        tabular(["Target/profile","Control","Median ratio","Finite n","0/0","positive/0"],[[r["target"].replace("_",r"\_")+"/"+r["profile"],LABELS[r["learned_method"]],number(r["variance_ratio_median"]),r["finite_variance_ratios"],r["variance_both_zero"],r["variance_zero_pair_positive_learned"]] for r in matched],"llrrrr"))
    write_text(output/"chosen_controls.tex", "% Auxiliary settings are identical toprimaryalpha0, not independent fitting/tuning.\n"+
        tabular(["Target/profile","Control",r"$B>0$",r"$\beta>0$",r"Both$>0$"],[[r["target"].replace("_",r"\_")+"/"+r["profile"],LABELS[r["method"]],r["nonzero_budget"],r["nonzero_beta"],r["nonzero_both"]] for r in choice_summaries],"llrrr"))
    numeric_compute = [[key.replace("_",r"\_"),number(value)] for key,value in compute["actual_work"].items()]
    numeric_compute += [[key.replace("_",r"\_"),number(value)] for key,value in compute["model_fit"].items()]
    write_text(output/"compute.tex", "% Measured sharedwork; extraCPU/fitting, scope exclusions andTUNEcalls inDESCRIPTIVE_RESULTS.json.\n"+tabular(["Quantity","Value"],numeric_compute))
    body = []
    for profile in ("strict","vulnerable"):
        group = [r for r in native_activity if r["profile"]==profile]
        for family in ("plain_mappo","feasible_mappo"):
            rows = [r for r in group if r["policy_family"]==family]
            body.append([profile+"/"+family.replace("_",r"\_"),sum(r["episodes_with_any_maturity"] for r in rows),
                sum(r["episodes_with_new_maturity"] for r in rows),sum(r["episodes_with_initial_prepaid_claims"] for r in rows),10*256])
    write_text(output/"native_activity.tex", "% Episodecounts; actor-request activity isseparateCSV andnotexecutedexposure. Oldstateeffectsretained.\n"+
        tabular(["Profile/cohort","Any maturity","New maturity","Prepaid initial","Episodes"],body))
    note = """# Local descriptive fragments for root integration

Both private audits and both independent public replays passed before any endpoint
was read. Fragments use booktabs and do not edit manuscript sources. LaTeX bounds
are in DKK with four significant digits; CSV/JSON preserve original float values.
Counts are out of20 policies per profile. Primary520 and auxiliary240 are separate
95% families; no joint95% statement. PrimaryRAW CI uses320 paths, other primary
methods256; ALL auxiliary methods256. Learned-versus-pairing comparisons and
variance ratios use common confirmation paths, are descriptive, and have no
formal superiority test. Undefined0/0 andpositive/0 variance ratios are retained
as statuses. Nonzeroactor requests are not executed trades. Prepaidclaims and
their actual state effects remain; auxiliary excludes only their direct maturity
contribution. ModelFIT CPU/query scope and source hashes are in the JSON report.
"""
    write_text(output/"README.md", note)
    manifest = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    write_text(output/"COMPLETION.json", json.dumps(dict(complete=True, file_sha256=manifest, inputs=provenance),indent=2,sort_keys=True)+"\n")
    return dict(status=report["status"],output="report_tables/"+run_id,primary_limits=520,auxiliary_limits=240,manuscript_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run-id",default="run_001");args=parser.parse_args()
    require(args.run_id.startswith("run_") and args.run_id.replace("_","").isalnum(),"Simple frozen run ID")
    print(json.dumps(generate(args.run_id),indent=2,sort_keys=True));return 0


if __name__=="__main__":
    raise SystemExit(main())
