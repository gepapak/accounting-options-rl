"""Construct NEW v2 implementation from exact original source; no native work.

This source transformation is retained as provenance. It never edits an old
source, imports an engine, freezes an experiment, or launches a worker.
"""
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent


def build(name, original, transform):
    source=(OLD/original).read_text(encoding='utf-8')
    target=HERE/name
    if target.exists():raise FileExistsError(target)
    changed=transform(source)
    with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(changed)
    return dict(original=original,original_sha256=hashlib.sha256((OLD/original).read_bytes()).hexdigest(),
        new=name,new_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def edit(source, old, new):
    if old not in source:raise ValueError('Missing exact source fragment: '+old[:100])
    return source.replace(old,new)


def estimators(source):
    source=edit(source,"METHODS=('raw','capped_raw','capped_residual','paired','ridge_control','mlp_control')", "METHODS=('raw','capped_raw','capped_residual','paired','ridge_control','mlp_control','cash_capped_raw')")
    source=edit(source,"    require(np.isfinite(alpha) and 0<=alpha<=1,'Charge fraction in [0,1]')", "    require(np.isfinite(alpha) and 0<=alpha<=1,'Charge fraction in [0,1]')\n    require(method!='cash_capped_raw' or alpha==0.,'Cash-bound method is alpha0 only')")
    source=edit(source,"    pair_contribution=weights*(paired-costs)", "    old=(coins!=0)&(steps<6)\n    # Retain every cost and coin; remove only direct inherited maturity G/I/A.\n    target_paired=np.where(old,0.,paired)\n    target_gross=np.where(old,0.,gross)\n    target_injections=np.where(old,0.,injections)\n    pair_contribution=weights*(target_paired-costs)")
    source=edit(source,"return dict(gross=float(weights@gross),cost=float(weights@costs),forgiveness=float(weights@injections),\n                paired_forgiveness=float(weights@paired)","return dict(gross=float(weights@target_gross),cost=float(weights@costs),forgiveness=float(weights@target_injections),\n                paired_forgiveness=float(weights@target_paired),\n                excluded_old_gross=float(weights@(gross-target_gross)),\n                excluded_old_forgiveness=float(weights@(injections-target_injections)),\n                excluded_old_paired_forgiveness=float(weights@(paired-target_paired)),\n                excluded_old_maturity_events=int(np.count_nonzero(old))")
    source=edit(source,"if method in ('raw','capped_raw'):","if method in ('raw','capped_raw','cash_capped_raw'):")
    source=edit(source,"    if method=='raw':return rawlo,rawhi", "    if method=='cash_capped_raw':\n        K=float(bounds['cash_inflow_bound'])\n        require(np.isfinite(K) and K>=0,'Source-derived excluded-inflow bound')\n        return -K,min(float(choice['cap']),rawhi)\n    if method=='raw':return rawlo,rawhi")
    source += "\n\ndef valid_pairs():\n    return tuple((alpha,method) for alpha in ALPHAS for method in METHODS if method!='cash_capped_raw' or alpha==0.)\n"
    return source


def worker(source):
    source=source.replace('native_sample_only_phase_v1','native_joint_fault_phase_v2')
    source=source.replace('26300000','36300000').replace('27300000','37300000').replace('28300000','38300000')
    source=edit(source,"    if not (0 <= policy < 20 and profile in (0, 1) and cid == 2*policy+profile):", "    if not (0 <= policy < 20 and profile == 1 and cid == policy):")
    source=edit(source,"    if case[\"profile\"] != (\"strict\", \"vulnerable\")[profile]:", "    if case[\"profile\"] != 'joint_fault':")
    source=source.replace('native_engine_adapter','joint_fault_adapter').replace('native_support','joint_fault_support').replace('sample_estimators','joint_fault_estimators')
    source=edit(source,'names = ("native_phase_worker.py", "joint_fault_adapter.py", "joint_fault_support.py",\n             "joint_fault_estimators.py", "audit_models.py", "checkpoint_inventory.json")', 'names = ("joint_fault_worker.py", "joint_fault_adapter.py", "joint_fault_support.py",\n             "joint_fault_estimators.py", "audit_models.py", "checkpoint_inventory.json")')
    source=edit(source,'    event_offsets = [0]', '''    event_offsets = [0]
    cash_fields=("transfer_step","transfer_kind","transfer_cash_change_inside_method","transfer_returned_cashflow",
        "path_native_cash_before","path_native_cash_after","path_excluded_native_cash_change")
    cash_arrays={key:[] for key in cash_fields}
    transfer_offsets=[0]''')
    source=edit(source,'                summarized = estimators.summarize(trace)', '''                summarized = estimators.summarize(trace)
                for key in cash_fields:cash_arrays[key].append(np.asarray(trace[key]))
                transfer_offsets.append(transfer_offsets[-1]+len(trace["transfer_step"]))''')
    source=edit(source,'                    event_count=summarized["events"], maturity_coins=coins.tolist(),', '''                    excluded_old_gross=summarized["excluded_old_gross"],
                    excluded_old_forgiveness=summarized["excluded_old_forgiveness"],
                    excluded_old_paired_forgiveness=summarized["excluded_old_paired_forgiveness"],
                    excluded_old_maturity_events=summarized["excluded_old_maturity_events"],
                    event_count=summarized["events"], maturity_coins=coins.tolist(),''')
    source=edit(source,'        fit_path = None', '''        cash_path=directory/"cash_flow_checks.npz"
        publish_npz(cash_path, **{key:concatenate(values) for key,values in cash_arrays.items()},
                    transfer_offsets=np.asarray(transfer_offsets,dtype=np.int64),
                    path_offsets=np.arange(n+1,dtype=np.int64)*HORIZON)
        fit_path = None''')
    source=edit(source,'artifact_paths = [row_path, branch_path, event_path, metadata_path]', 'artifact_paths = [row_path, branch_path, event_path, cash_path, metadata_path]')
    return source


def runner(source):
    source=edit(source,'CODE=HERE.parents[1]','CODE=HERE.parents[2]')
    source=edit(source,'CAPS=(1000,4000,16000,64000,256000,1024000,4096000,16384000)','CAPS=tuple(1000.*4**j for j in range(11))')
    source=source.replace('.05/520','.05/280').replace("family_limits=520","family_limits=280")
    start=source.index('def cases():')
    end=source.index('\n\ndef freeze(',start)
    source=source[:start]+'''def cases():
    from run_joint_fault_v2 import cases as source_cases
    return [dict(row,profile_index=1) for row in source_cases()]
'''+source[end:]
    source=edit(source,"source_names=('EXPLORATORY_EXECUTION_PLAN.md','run_native_audit.py','native_phase_worker.py',\n        'native_engine_adapter.py','native_support.py','sample_estimators.py','audit_models.py',\n        'SAMPLE_ONLY_THEORY.md','exact_sample_only_checks.py','exact_sample_only_checks.json',\n        'checkpoint_inventory.json')", "source_names=('EXPLORATORY_EXECUTION_PLAN.md','run_joint_fault_audit.py','run_joint_fault_v2.py','joint_fault_worker.py',\n        'joint_fault_adapter.py','joint_fault_support.py','joint_fault_estimators.py','audit_models.py',\n        'checkpoint_inventory.json','V2_INDEPENDENT_SOURCE_REVIEW.md')")
    source=edit(source,"    native={str(p.relative_to(ENGINE)):sha(p) for p in sorted(ENGINE.glob('*.py'))}","    native={str(p.relative_to(ENGINE)):sha(p) for p in sorted(ENGINE.glob('*.py'))}\n    original_freeze=read(HERE.parent/'execution/run_001/SOURCE_FREEZE.json')\n    original={name:sha(HERE.parent/name) for name in original_freeze['source_sha256']}\n    assert original==original_freeze['source_sha256']")
    source=edit(source,'source_sha256=source,native_source_sha256=native,checkpoint_sha256=artifacts,','source_sha256=source,native_source_sha256=native,checkpoint_sha256=artifacts,\n        original_source_sha256=original,original_freeze_sha256=sha(HERE.parent/\'execution/run_001/SOURCE_FREEZE.json\'),')
    source=edit(source,'    return record\n\n\ndef job',"    for name,digest in record['original_source_sha256'].items():\n        assert sha(HERE.parent/name)==digest\n    return record\n\n\ndef job")
    source=source.replace('from native_phase_worker import run_case','from joint_fault_worker import run_case').replace('from sample_estimators import','from joint_fault_estimators import')
    source=source.replace('n_audit_episodes=12800','n_audit_episodes=6400').replace('n_audit_steps=1228800','n_audit_steps=614400').replace('n_warmup_steps_two_stages=245760','n_warmup_steps_two_stages=122880')
    source=source.replace('n_cases=40','n_cases=20').replace('n_native_audit_steps=1228800','n_native_audit_steps=614400').replace('==1228800','==614400').replace('==245760','==122880').replace('total_cases=40','total_cases=20')
    source=source.replace('seed=29300000','seed=39300000')
    source=source.replace("    write(directory/'SOURCE_FREEZE.json',manifest)","    write(directory/'SOURCE_FREEZE.json',manifest)")
    source=edit(source,'        for alpha in ALPHAS:\n            for method in METHODS:', "        for alpha in ALPHAS:\n            for method in METHODS:\n                if method=='cash_capped_raw' and alpha!=0.:continue")
    source=source.replace("ALL MODELS AND 480 CHOICES", "ALL MODELS AND 260 CHOICES")
    source=source.replace("for profile in ('strict','vulnerable')", "for profile in ('joint_fault',)")
    source=edit(source,"    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','aggregate'))", "    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','aggregate'))\n    parser.add_argument('--coordinator-release',default='')")
    source=edit(source,"    directory=HERE/'execution'/args.run_id", "    if not args.coordinator_release.strip():raise RuntimeError('HELD: explicit coordinator release required; no freeze/native execution authorized yet')\n    directory=HERE/'execution'/args.run_id")
    return source


def main():
    records=[]
    for name,old,transform in (("joint_fault_estimators.py","sample_estimators.py",estimators),
        ("joint_fault_worker.py","native_phase_worker.py",worker),
        ("run_joint_fault_audit.py","run_native_audit.py",runner),
        ("audit_models.py","audit_models.py",lambda s:s)):
        records.append(build(name,old,transform))
    source=OLD/'checkpoint_inventory.json';target=HERE/'checkpoint_inventory.json'
    with target.open('xb') as stream:stream.write(source.read_bytes())
    records.append(dict(original=source.name,original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new=target.name,new_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    with (HERE/'IMPLEMENTATION_SOURCE_LINEAGE.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(status='NEW UNFROZEN SOURCE CONSTRUCTION ONLY',records=records,new_native_queries=0,new_freeze=False),stream,indent=2)
    print(json.dumps(dict(status='NEW PIPELINE SOURCES WRITTEN; NO NATIVE EXECUTION OR FREEZE',files=len(records))))


if __name__=='__main__':main()
