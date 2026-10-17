"""DRAFT source-only runner contract; native execution and freezing unavailable.

The module uses only the standard library and never imports the native adapter,
actors, regressors, NumPy, or original runner. Completion requires later review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent
SEEDS = (7,42,123,2025,3007,5001,8102,9005,10001,11202)
PHASE_N = {"fit":32,"tune":32,"confirm":256}
PHASE_BANDS = {"fit":36300000,"tune":37300000,"confirm":38300000}
METHODS = ("raw","capped_raw","capped_residual","paired","ridge_control","mlp_control")
ALPHAS = (0.,1.)
CAPS = (1000.,4000.,16000.,64000.,256000.,1024000.,4096000.,16384000.)
BETAS = (0.,.25,.5,1.,2.)
BUDGET_RATIOS = (0.,1/16,1/4,1.,4.)
FIT_SEED = 39300000
N_CASES,HORIZON,BURN_IN,CONTEXTS,MAX_WORKERS = 20,96,384,8,3
FAMILY_LIMITS = 260
DELTA = .05/FAMILY_LIMITS
ORIGINAL_FREEZE_SHA = "6152949392bd8de0f494bc21ffb5fd7d9be4a16a6c8866f4eb5595254da523d4"
PENDING_COMPONENTS = ("new_native_joint_fault_adapter", "new_native_joint_fault_worker",
    "cap0_broad_and_calendar_support_review", "authentic_post_intervention_observation_review",
    "independent_source_and_software_validation_release", "optional_alpha0_method_decision",
    "final_runner_implementation_and_independent_review", "coordinator_freeze_and_launch_release")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def cases():
    """Read only the original fixed registry; do not inspect any outcome row."""
    old = read(ORIGINAL/"execution/run_001/cases.json")
    require(len(old)==40, "Complete original fixed registry required")
    registry = {int(row["case_id"]):row for row in old}
    output=[]
    for policy in range(N_CASES):
        source = registry[2*policy+1]
        require(source["profile"]=="vulnerable" and int(source["policy_index"])==policy,
                "Original vulnerable-policy lineage")
        require(int(source["checkpoint_seed"])==SEEDS[policy%10], "Original complete actor order")
        require(source["policy_family"]==("plain_mappo" if policy<10 else "feasible_mappo"), "Original actor family")
        output.append(dict(case_id=policy,policy_index=policy,policy_family=source["policy_family"],
            checkpoint_seed=source["checkpoint_seed"],checkpoint_dir=source["checkpoint_dir"],
            source_run001_case_id=2*policy+1,profile="joint_fault",warm_profile="vulnerable",
            feature_profile_index=1,cash_fraction=1e-4,horizon=HORIZON,burn_in=BURN_IN,
            warm_liquidity_participation_cap_fraction=.25,audit_liquidity_participation_cap_fraction=0.,
            enable_trading_sleeve_margin=False,mtm_loss_exit_disabled=True,
            intervention_timing="fresh_clone_after_original_warm_before_support_observation_and_actor"))
    require(len(output)==len({row["case_id"] for row in output})==20,"All twenty unique policies")
    return output


def dependency_identity():
    """Hash frozen source contracts only, never model/data deserialization."""
    freeze_path=ORIGINAL/"execution/run_001/SOURCE_FREEZE.json"
    require(sha(freeze_path)==ORIGINAL_FREEZE_SHA,"Preserved original freeze identity")
    original=read(freeze_path)
    checked={}
    for name,digest in original["source_sha256"].items():
        p=ORIGINAL/name
        require(sha(p)==digest,"Original frozen source unchanged: "+name)
        checked[name]=digest
    return dict(original_freeze_sha256=sha(freeze_path), original_source_sha256=checked,
        original_cases_sha256=sha(ORIGINAL/"execution/run_001/cases.json"))


def protocol_manifest():
    audit_steps=N_CASES*sum(PHASE_N.values())*HORIZON
    warm_steps=N_CASES*CONTEXTS*BURN_IN*2
    return dict(status="DRAFT PREPARED; NOT FROZEN; EXECUTION UNAVAILABLE",created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),plan_sha256=sha(HERE/"EXPLORATORY_PROTOCOL_DRAFT.md"),
        dependency_identity=dependency_identity(),cases=cases(),methods=list(METHODS),alphas=list(ALPHAS),
        phase_n=PHASE_N,phase_bands=PHASE_BANDS,proposed_audit_regressor_fit_seed=FIT_SEED,
        caps=CAPS,betas=BETAS,budget_ratios=BUDGET_RATIOS,family_limits=FAMILY_LIMITS,total_delta=.05,delta_per_limit=DELTA,
        optional_method=dict(status="DISABLED_PENDING_SOURCE_PROOF",alphas=[0.],
            family_if_enabled=280,not_enabled=True),
        planned_work=dict(audit_native_steps=audit_steps,two_stage_warm_native_steps=warm_steps,
            total_native_steps=audit_steps+warm_steps,audit_actor_predict_calls=2*audit_steps,
            warm_actor_predict_calls=2*warm_steps,zero_step_support_initializations=N_CASES*CONTEXTS*2),
        max_workers=MAX_WORKERS,numerical_threads_per_worker=1,pending=list(PENDING_COMPONENTS),
        original_outcomes_used_for_case_or_method_selection=False,
        followup_motivation="original null known; source-driven joint fault utility validation; all original actors retained",
        new_native_queries=0,new_model_fits=0,new_statistical_draws=0,source_freeze_created=False)


def validate_source_contract():
    proposal=protocol_manifest()
    require(FAMILY_LIMITS==N_CASES*len(ALPHAS)*len(METHODS)+N_CASES==260,"Base family arithmetic")
    require(FAMILY_LIMITS+N_CASES==280,"Optional alpha0-only representation arithmetic")
    require(len(set(PHASE_BANDS.values()))==3 and set(PHASE_BANDS.values()).isdisjoint({26300000,27300000,28300000}),"Fresh separated streams")
    require(proposal["planned_work"]["audit_native_steps"]==614400,"Fixed audit budget")
    require(proposal["planned_work"]["two_stage_warm_native_steps"]==122880,"Fixed warm budget")
    require(proposal["planned_work"]["total_native_steps"]==737280,"Total proposed native work")
    require(all(r["warm_liquidity_participation_cap_fraction"]==.25 and r["audit_liquidity_participation_cap_fraction"]==0 for r in proposal["cases"]),"Audit-only cap fault")
    require(proposal["source_freeze_created"] is False and proposal["new_native_queries"]==0,"Held source-only contract")
    return proposal


def run_phase(*_args, **_kwargs):
    raise RuntimeError("DRAFT ONLY: no native worker implementation/release; no execution authorized")


def freeze(*_args, **_kwargs):
    raise RuntimeError("DRAFT ONLY: independent review and coordinator decision precede any freeze")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",nargs="?",choices=("describe","check"),default="describe")
    args=parser.parse_args()
    proposal=validate_source_contract() if args.command=="check" else protocol_manifest()
    print(json.dumps(proposal,indent=2,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
