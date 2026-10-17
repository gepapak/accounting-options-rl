"""Metadata/hashes only; copies no external source, checkpoints, or raw inputs."""
from __future__ import annotations
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[3]/"Prototype5"

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()

def record(path,role):
    return {"path_from_native_engine":path.relative_to(ENGINE).as_posix(),"role":role,
        "bytes":path.stat().st_size,"sha256":sha(path),"included_in_public_code_repository":False}

def main():
    frozen=json.loads((HERE/"execution/run_001/SOURCE_FREEZE.json").read_text(encoding="utf-8"))
    inventory=json.loads((HERE/"checkpoint_inventory.json").read_text(encoding="utf-8"))
    core=[]
    for name,digest in frozen["native_source_sha256"].items():
        path=ENGINE/name;assert sha(path)==digest
        core.append({"path_from_native_engine":Path(name).as_posix(),"sha256":digest})
    checkpoints=[]
    for name,digest in frozen["checkpoint_sha256"].items():
        path=ENGINE/name;assert sha(path)==digest
        checkpoints.append({"path_from_native_engine":Path(name).as_posix(),"sha256":digest,
            "role":"frozen native actor" if name.endswith("investor_0_policy.zip") else
                   "matching frozen battery controller" if name.endswith("battery_operator_0_policy.zip") else
                   "saved native training/runtime contract"})
    resources=[]
    for stem in ("unseendata","unseendata_v2"):
        for suffix,role in ((".csv","chronological physical/price context"),("_settlement_real_v2.csv","aligned financial settlement"),("_liquidity_volume_real_v1.csv","activated liquidity and impact reference")):
            resources.append(record(ENGINE/"evaluation_dataset_ffill"/(stem+suffix),role))
    for path in sorted((ENGINE/"rolling_past_history_dataset_ffill").glob("*.csv")):
        resources.append(record(path,"rolling-past native normalization history or history manifest"))
    forecast_root=ENGINE/"forecast_cache_settlement_hourly_v2/forecast_cache_eval_episode20_2025"
    for folder in ("forecast_cache_eval_episode20_2025-full","forecast_cache_eval_episode20_2025-unseendata_v2"):
        files=[path for path in (forecast_root/folder).glob("precomputed_forecasts_*.csv") if "metadata" not in path.name]
        assert len(files)==1
        resources.append(record(files[0],"trained forecast/constraint actor's native exogenous feature source"))
    # Advisory static dependency closure; pickle/dynamic imports can need more.
    pending=["evaluation","environment","config","generator","forecast_prior","policy","logger"]
    closure=set()
    while pending:
        module=pending.pop()
        if module in closure or not (ENGINE/(module+".py")).is_file():continue
        closure.add(module)
        source=ast.parse((ENGINE/(module+".py")).read_text(encoding="utf-8-sig"))
        for node in ast.walk(source):
            names=([item.name for item in node.names] if isinstance(node,ast.Import) else
                   [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            pending.extend(name.split(".")[0] for name in names)
    output={"schema":"native_audit_public_reproduction_dependencies_v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"honest dependency/access manifest; no scientific endpoint results and no private input copies",
        "source_freeze_sha256":sha(HERE/"execution/run_001/SOURCE_FREEZE.json"),
        "metadata_builder_sha256":sha(__file__),"public_audit_source_sha256":frozen["source_sha256"],
        "python_version_from_preexecution_inventory":inventory["python_version"],
        "installed_packages_from_preexecution_inventory":inventory["installed_package_versions"],
        "external_engine_root":"<Paper1>/Prototype5",
        "required_public_code_location":"<Paper1>/github/accounting-options-rl/exploratory_novelty_2026-10-16/sample_only_learned_audit",
        "full_frozen_native_module_bundle":core,
        "advisory_static_import_closure":[module+".py" for module in sorted(closure)],
        "minimal_bundle_caveat":"Static closure is advisory: dynamic imports and saved custom-policy deserialization are not established by AST analysis. Full frozen native module bundle is the currently verified source dependency set; no standalone public-native bundle has been tested.",
        "checkpoint_artifacts":checkpoints,"market_history_forecast_resources":resources,
        "derived_runtime_forecast_views":"8 deterministic480-row views; generated from the two hashed forecast sources; source/view hashes and zero-step preparation recorded in native_validation/forecast_cache_prep_attempt_1/preparation_report.json; not copied into this public manifest",
        "access_limit":"Native source/checkpoints/context/history/forecast files are outside this code repository. Hashes establish identity and do not supply access or redistribution rights. Independent full-pipeline reproduction currently requires an authorized copy of these exact external dependencies; the public repository alone is insufficient.",
        "public_vs_private":"Only filenames, role/size/version metadata and SHA256 values are included here. No raw data, actor/battery weights, feature rows, derived private forecasts or fitted nuisance-model files are copied.",
        "remaining_packaging_work":["Provide or archive an authorized versioned native source bundle, checkpoint bundles and complete input-access instructions.","Validate a clean installation of the saved custom policy classes and native observation/normalization contract.","Publish aggregate frozen choices, all case/method outcomes, support and arithmetic-verifier records after completion; keep licensed raw/derived inputs private.","Perform a separate clean-room re-execution when authorized dependencies can be supplied; matching a hash manifest is not that re-execution."]}
    destination=HERE/"public_reproducibility_manifest.json"
    with destination.open("x",encoding="utf-8") as stream:json.dump(output,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({"output":str(destination),"native_modules":len(core),"advisory_static_modules":len(closure),"checkpoint_artifacts":len(checkpoints),"external_resource_files":len(resources),"actor_calls":0,"native_steps":0}))

if __name__=="__main__":main()
