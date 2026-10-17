"""Standard-library source/registry checks only; no native execution or freeze."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def main():
    source=HERE/"run_joint_fault_v2.py"
    parsed=ast.parse(source.read_text(encoding="utf-8"))
    module_spec=importlib.util.spec_from_file_location("joint_fault_source_contract",source)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    before=module.dependency_identity()
    proposal=module.validate_source_contract()
    registry=proposal["cases"]
    check(len(registry)==20,"all20 fixed policies")
    check([r["case_id"] for r in registry]==list(range(20)),"uniquecaseIDequalsfixedpolicyID")
    check([r["source_run001_case_id"] for r in registry]==list(range(1,40,2)),"originalvulnerablelineage")
    check(sum(r["policy_family"]=="plain_mappo" for r in registry)==10,"all10plainactors")
    check(sum(r["policy_family"]=="feasible_mappo" for r in registry)==10,"all10feasibleactors")
    check(all(r["warm_liquidity_participation_cap_fraction"]==.25 for r in registry),"warmcappreserved")
    check(all(r["audit_liquidity_participation_cap_fraction"]==0 for r in registry),"auditonlymissingcap")
    check(all(r["enable_trading_sleeve_margin"] is False and r["mtm_loss_exit_disabled"] is True for r in registry),"jointfaultguardsdeclared")
    check(proposal["phase_n"]=={"fit":32,"tune":32,"confirm":256},"fixedwholeepisodebudgets")
    check(proposal["family_limits"]==260 and proposal["delta_per_limit"]==.05/260,"base260family")
    check(proposal["optional_method"]["not_enabled"] is True and proposal["optional_method"]["family_if_enabled"]==280,"optionalalpha0methodheld")
    check(20*2*7+20==300,"bothalpha7methodswouldneed300")
    check(proposal["planned_work"]["total_native_steps"]==737280,"source-derivedtotalplannedwork")
    check(proposal["max_workers"]==3 and proposal["numerical_threads_per_worker"]==1,"reviewedresourceceiling")
    for function in (module.run_phase,module.freeze):
        try:function()
        except RuntimeError:check(True,function.__name__+"disabled")
        else:raise AssertionError(function.__name__+"unexpectedlyenabled")
    imports=[]
    for node in ast.walk(parsed):
        if isinstance(node,ast.Import):imports.extend(n.name.split('.')[0] for n in node.names)
        if isinstance(node,ast.ImportFrom) and node.module:imports.append(node.module.split('.')[0])
    check(set(imports)<=set(("__future__","argparse","datetime","hashlib","json","pathlib")),"standardlibraryonlynoengineinferenceorarrays")
    check(before==module.dependency_identity(),"originalprotectedsourcehashesunchanged")
    check(proposal["new_native_queries"]==proposal["new_model_fits"]==proposal["new_statistical_draws"]==0 and
        proposal["source_freeze_created"] is False,"noexecutionorfreeze")
    report=dict(status="PASS SOURCE-ONLY DRAFT CHECKS",created_utc=datetime.now(timezone.utc).isoformat(),checks=len(checks),
        check_names=checks,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        plan_sha256=hashlib.sha256((HERE/"EXPLORATORY_PROTOCOL_DRAFT.md").read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protected_original=before,
        new_native_steps=0,new_actor_calls=0,new_regressor_fits=0,new_statistical_draws=0,
        freeze_created=False,classification="source-only proposal checks; adapter/support/runtime validation remains pending")
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=HERE/("SOURCE_CONTRACT_CHECK_"+stamp+".json")
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(dict(status=report['status'],checks=len(checks),report=output.name,source_sha256=report['source_sha256'],plan_sha256=report['plan_sha256'],new_native_steps=0)))


if __name__=="__main__":main()
