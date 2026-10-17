"""New joint-fault exploratory native audit; no simulations on import.

Each command requires --coordinator-release with an explicit coordinator note:
python run_joint_fault_audit.py freeze
python run_joint_fault_audit.py run
python run_joint_fault_audit.py aggregate
Every generated experiment artifact is exclusive-created and source-bound.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
CODE=HERE.parents[2]
PAPER1=CODE.parents[1]
ENGINE=PAPER1/'Prototype5'
SEEDS=(7,42,123,2025,3007,5001,8102,9005,10001,11202)
CAPS=tuple(1000.*4**j for j in range(11))
BETAS=(0.,.25,.5,1.,2.)
BUDGET_RATIOS=(0.,1/16,1/4,1.,4.)
N_FIT,N_TUNE,N_CONFIRM=32,32,256
DELTA=.05/340
WORKERS=3


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def rows(path):
    with Path(path).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]


def cases():
    from joint_fault_registry import cases as source_cases
    return [dict(row,profile_index=1) for row in source_cases()]


def context_identity(contexts):
    # Warm CPU/wall timers legitimately differ across stages; economic state,
    # resources, contracts and the WHOLE context support must remain identical.
    return [dict(context=r['context'],support=r['support'],resource_sha256=r['metadata']['resource_sha256'],
        config_snapshot_sha256=r['metadata']['config_snapshot_sha256'],warm_snapshot_sha256=r['metadata']['warm_snapshot_sha256'],
        spec=r['metadata']['spec'],warm_spec=r['metadata']['warm_spec'],
        profile=r['metadata']['profile_intervention'],fault=r['metadata']['joint_fault_intervention']) for r in contexts]


def freeze(directory):
    directory.mkdir(parents=True,exist_ok=True)
    source_names=('EXPLORATORY_EXECUTION_PLAN.md','run_joint_fault_audit.py','joint_fault_registry.py','joint_fault_worker.py',
        'joint_fault_adapter.py','joint_fault_support.py','joint_fault_estimators.py','audit_models.py',
        'checkpoint_inventory.json','V2_INDEPENDENT_SOURCE_REVIEW.md','SOURCE_SUPPORT_REVIEW_DRAFT.md',
        'prepare_joint_fault_cache.py','validate_joint_fault_smoke.py','TECHNICAL_SMOKE_PLAN.md',
        'PRE_STATISTICAL_CASH_CONTROL_REVIEW.md','validation/exact_cash_control_checks.py','validation/exact_cash_control_checks.json',
        'public_export_joint_fault.py','reproduce_public_joint_fault.py','validation/verify_joint_fault_results.py',
        'post_joint_fault_diagnostics.py',
        'theory/CASH_INFLOW_BUDGET_THEORY.md','theory/exact_cash_inflow_budget_checks.py','theory/exact_cash_inflow_budget_checks.json')
    source={name:sha(HERE/name) for name in source_names}
    native={str(p.relative_to(ENGINE)):sha(p) for p in sorted(ENGINE.glob('*.py'))}
    original_freeze=read(HERE.parent/'execution/run_001/SOURCE_FREEZE.json')
    assert native==original_freeze['native_source_sha256']
    assert sha(HERE.parent/'execution/run_001/cases.json')==original_freeze['cases_sha256']
    original={name:sha(HERE.parent/name) for name in original_freeze['source_sha256']}
    assert original==original_freeze['source_sha256']
    preparation_path=HERE/'validation/cache_preparation_attempt_1/preparation_report.json'
    preparation=read(preparation_path)
    assert preparation['status']=='PASS' and preparation['native_steps']==preparation['actor_calls']==0
    assert sha(HERE/'prepare_joint_fault_cache.py')==preparation['source_sha256']
    runtime_inputs={str(Path(path).relative_to(ENGINE)):digest for path,digest in preparation['verified_input_sha256'].items() if Path(path).suffix.lower()=='.csv'}
    assert len(runtime_inputs)==10
    forecast_views={str(Path(row['cache_path']).relative_to(HERE)):row['cache_sha256'] for row in preparation['contexts']}
    assert len(forecast_views)==8 and {r['context_id'] for r in preparation['contexts']}==set(range(8))
    for root,mapping in ((ENGINE,runtime_inputs),(HERE,forecast_views)):
        for name,digest in mapping.items():assert sha(root/name)==digest
    smoke_path=HERE/'validation/technical_smoke_attempt_1/smoke_report.json'
    smoke=read(smoke_path)
    assert smoke['status']=='PASS' and len(smoke['cases'])==2
    for name,digest in smoke['source_sha256'].items():assert sha(HERE/name)==digest
    artifacts={}
    for case in cases():
        root=Path(case['checkpoint_dir'])
        for name in ('investor_0_policy.zip','battery_operator_0_policy.zip','training_config.json'):
            path=root/name;artifacts[str(path.relative_to(ENGINE))]=sha(path)
    assert artifacts==original_freeze['checkpoint_sha256']
    snapshot=directory/'source_snapshot';snapshot.mkdir()
    for name in source:
        (snapshot/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/name,snapshot/name)
    write(directory/'cases.json',cases())
    manifest=dict(status='FROZEN BEFORE STATISTICAL DEVELOPMENT',created_utc=utc(),
        classification='exploratory, internally hash-frozen; not a registered learner study',
        source_sha256=source,native_source_sha256=native,checkpoint_sha256=artifacts,
        original_source_sha256=original,original_freeze_sha256=sha(HERE.parent/'execution/run_001/SOURCE_FREEZE.json'),
        original_cases_sha256=original_freeze['cases_sha256'],
        runtime_input_sha256=runtime_inputs,forecast_view_sha256=forecast_views,
        cache_preparation_report_path=str(preparation_path.relative_to(HERE)),cache_preparation_report_sha256=sha(preparation_path),
        cache_preparation_source_sha256=preparation['source_sha256'],input_manifest_sha256=preparation['input_manifest_sha256'],
        technical_smoke_report_path=str(smoke_path.relative_to(HERE)),technical_smoke_report_sha256=sha(smoke_path),
        cases_sha256=sha(directory/'cases.json'),workers=WORKERS,n_fit=N_FIT,n_tune=N_TUNE,
        n_confirm=N_CONFIRM,n_audit_episodes=6400,n_audit_steps=614400,
        n_warmup_steps_two_stages=122880,family_limits=340,delta_per_limit=DELTA,
        python=sys.version,platform=sys.platform)
    write(directory/'SOURCE_FREEZE.json',manifest)
    print(json.dumps(dict(status='FROZEN',directory=str(directory),sha256=sha(directory/'SOURCE_FREEZE.json'))),flush=True)


def verify_freeze(directory):
    record=read(directory/'SOURCE_FREEZE.json')
    assert record['cases_sha256']==sha(directory/'cases.json')
    assert read(directory/'cases.json')==cases()
    for group,root in (('source_sha256',HERE),('native_source_sha256',ENGINE),('checkpoint_sha256',ENGINE),('runtime_input_sha256',ENGINE),('forecast_view_sha256',HERE)):
        for name,digest in record[group].items():
            if sha(root/name)!=digest:raise RuntimeError(f'Frozen source/input changed: {group}/{name}')
    assert sha(HERE.parent/'execution/run_001/SOURCE_FREEZE.json')==record['original_freeze_sha256']
    assert sha(HERE.parent/'execution/run_001/cases.json')==record['original_cases_sha256']
    for name,digest in record['original_source_sha256'].items():
        assert sha(HERE.parent/name)==digest
    assert sha(HERE/record['cache_preparation_report_path'])==record['cache_preparation_report_sha256']
    assert sha(HERE/'prepare_joint_fault_cache.py')==record['cache_preparation_source_sha256']
    assert sha(HERE/record['technical_smoke_report_path'])==record['technical_smoke_report_sha256']
    return record


def job(case,phase,directory,models_dir):
    from joint_fault_worker import run_case
    return run_case(case,phase=phase,output_dir=str(directory),
        n=N_FIT if phase=='development' else N_CONFIRM,models_dir=models_dir)


def run_phase(directory,phase):
    verify_freeze(directory)
    import psutil
    available=int(psutil.virtual_memory().available)
    workers=min(WORKERS,max(0,(available-512*1024**2)//(750*1024**2)))
    if workers<1:raise RuntimeError('Memory guard: retain512MiB reserve plus750MiB per native worker')
    start=time.perf_counter();cpu=time.process_time()
    model_dir=str(directory/'models') if phase=='confirm' else None
    if phase=='confirm':
        selection=read(directory/'SELECTION_FREEZE.json')
        assert selection['selection_sha256']==sha(directory/'selection.json')
        assert selection['model_manifest_sha256']==sha(directory/'models/audit_models_manifest.json')
        assert selection['raw_model_manifest_sha256']==sha(directory/'models_raw/audit_models_manifest.json')
    # Preserve incomplete attempts: a worker refuses overwriting partial files.
    completed=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(job,case,phase,str(directory),model_dir):case for case in cases()}
        for future in as_completed(futures):
            case=futures[future]
            result=future.result();completed.append(result)
            print(json.dumps(dict(utc=utc(),phase=phase,case_id=case['case_id'],
                completed_cases=len(completed),total_cases=20)),flush=True)
    write(directory/f'{phase}_stage_complete.json',dict(complete=True,utc=utc(),
        phase=phase,results=completed,wall_seconds=time.perf_counter()-start,
        parent_cpu_seconds=time.process_time()-cpu,workers=workers,available_memory_at_dispatch=available))
    verify_freeze(directory)


def phase_path(directory,case,phase):return directory/f"case_{case['case_id']:02d}"/phase


def evaluate_saved_branches(phase_dir,models,base_rows):
    import numpy as np
    data=np.load(phase_dir/'branch_features.npz',allow_pickle=False)
    offsets=data['episode_offsets']
    assert len(offsets)==len(base_rows)+1
    predictions={name:(model.predict(data['plus'])-model.predict(data['minus']))/2+
        (data['known_current_raw_delta'] if name.startswith('cash_') else 0.) for name,model in models.items()}
    output=[]
    for i,row in enumerate(base_rows):
        row=dict(row);lo,hi=int(offsets[i]),int(offsets[i+1])
        row['maturity_coins']=data['coins'][lo:hi].tolist()
        row['delta_cash_local']=data['known_current_raw_delta'][lo:hi].tolist()
        for name,values in predictions.items():row['delta_'+name]=values[lo:hi].tolist()
        output.append(row)
    return output


def bound_for_case(directory,case,phase='fit'):
    metadata=read(phase_path(directory,case,phase)/'metadata.json')
    return metadata['bounds']


def fit_and_select(directory):
    import numpy as np
    from audit_models import fit_models,load_models
    from joint_fault_estimators import METHODS,ALPHAS,select_choice
    verify_freeze(directory)
    assert (directory/'development_stage_complete.json').exists()
    fitX=[];fity=[];fity_raw=[];inputs={}
    for case in cases():
        path=phase_path(directory,case,'fit')/'fit_rows.npz'
        data=np.load(path,allow_pickle=False);fitX.append(data['X']);fity.append(data['y'])
        fity_raw.append(data['y_raw'])
        inputs[str(case['case_id'])]=sha(path)
    model_metadata=fit_models(np.concatenate(fitX),np.concatenate(fity),directory/'models',seed=39300000,
        target_label='GLOBAL after-event future NEW paired-injection-minus-cost; old directmaturity excluded')
    raw_model_metadata=fit_models(np.concatenate(fitX),np.concatenate(fity_raw),directory/'models_raw',seed=40300000,
        target_label='GLOBAL after-event future NEW raw financial component G-C+I; old directmaturity excluded')
    models=load_models(directory/'models')
    models.update({'cash_'+name:model for name,model in load_models(directory/'models_raw').items()})
    selections=[];search_records=[]
    cpu=time.process_time();start=time.perf_counter()
    for case in cases():
        fitdir=phase_path(directory,case,'fit');tunedir=phase_path(directory,case,'tune')
        fitrows=rows(fitdir/'rows.jsonl');tunerows=rows(tunedir/'rows.jsonl')
        learned_tune=evaluate_saved_branches(tunedir,models,tunerows)
        assert len(fitrows)==N_FIT and len(tunerows)==N_TUNE
        bounds=bound_for_case(directory,case)
        assert bounds==bound_for_case(directory,case,'tune')
        assert context_identity(read(fitdir/'metadata.json')['cases_contexts'])==context_identity(read(tunedir/'metadata.json')['cases_contexts'])
        for alpha in ALPHAS:
            for method in METHODS:
                if method.startswith('cash_') and alpha!=0.:continue
                development=learned_tune if method.endswith('_control') else fitrows+tunerows
                choice,search=select_choice(development,method,alpha,bounds,DELTA,N_CONFIRM,CAPS,BETAS,BUDGET_RATIOS)
                selections.append(dict(case_id=case['case_id'],alpha=alpha,method=method,choice=choice))
                search_records.append(dict(case_id=case['case_id'],alpha=alpha,method=method,candidates=search))
    write(directory/'selection.json',selections)
    write(directory/'search_records.json',search_records)
    write(directory/'SELECTION_FREEZE.json',dict(complete=True,created_utc=utc(),
        confirmation_not_started=True,selection_sha256=sha(directory/'selection.json'),
        search_sha256=sha(directory/'search_records.json'),
        model_manifest_sha256=sha(directory/'models/audit_models_manifest.json'),
        raw_model_manifest_sha256=sha(directory/'models_raw/audit_models_manifest.json'),
        fit_input_sha256=inputs,raw_fit_input_sha256=inputs,model_metadata=model_metadata,raw_model_metadata=raw_model_metadata,
        additional_parent_tune_critic_predict_calls=160,
        selection_cpu_seconds=time.process_time()-cpu,selection_wall_seconds=time.perf_counter()-start,
        source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json')))
    assert len(selections)==320
    print(json.dumps(dict(status='ALL FOUR MODELS AND 320 CHOICES FROZEN BEFORE CONFIRM',utc=utc())),flush=True)


def aggregate(directory):
    import numpy as np
    from joint_fault_estimators import METHODS,ALPHAS,sample_values,deterministic_support,empirical_bernstein,numerical_pad
    verify_freeze(directory)
    assert (directory/'confirm_stage_complete.json').exists()
    selection={(row['case_id'],row['alpha'],row['method']):row['choice'] for row in read(directory/'selection.json')}
    charged=[];cost_results=[];case_diagnostics=[]
    actual_work=dict(audit_native_transitions=0,warm_native_transitions=0,actor_forward_calls=0,
        warm_actor_forward_calls=0,branch_injection_arithmetic_evaluations=0,
        known_old_payoff_point_evaluations=0,
        critic_branch_rows_per_model=0,critic_predict_calls=0,fresh_audit_clones=0,
        zero_step_support_initializations=0,phase_cpu_seconds=0.,preparation_cpu_seconds=0.)
    for case in cases():
        phase_rows={phase:rows(phase_path(directory,case,phase)/'rows.jsonl') for phase in ('fit','tune','confirm')}
        for phase in phase_rows:
            phase_metadata=read(phase_path(directory,case,phase)/'metadata.json')
            for key,value in phase_metadata['query_counts'].items():actual_work[key]+=value
            actual_work['phase_cpu_seconds']+=phase_metadata['cpu_seconds']
        for stage in ('development','confirm'):
            preparation=read(directory/f"case_{case['case_id']:02d}"/f'{stage}_completion.json')['preparation']
            for key in ('warm_native_transitions','warm_actor_forward_calls','zero_step_support_initializations','known_old_payoff_point_evaluations'):
                actual_work[key]+=preparation[key]
            actual_work['preparation_cpu_seconds']+=preparation['cpu_seconds']
        assert [len(phase_rows[k]) for k in ('fit','tune','confirm')]==[32,32,256]
        bounds=bound_for_case(directory,case)
        for phase in ('tune','confirm'):
            current=bound_for_case(directory,case,phase)
            assert current==bounds,(case['case_id'],phase,'whole support identity inclK andanalyticfee')
            assert context_identity(read(phase_path(directory,case,phase)/'metadata.json')['cases_contexts'])==context_identity(read(phase_path(directory,case,'fit')/'metadata.json')['cases_contexts'])
        allrows=sum(phase_rows.values(),[]);confirm=phase_rows['confirm']
        for alpha in ALPHAS:
            for method in METHODS:
                if method.startswith('cash_') and alpha!=0.:continue
                choice=selection[case['case_id'],alpha,method]
                used=allrows if method=='raw' else confirm
                values=sample_values(used,method,alpha,choice)
                lo,hi=deterministic_support(bounds,method,alpha,choice)
                certificate=empirical_bernstein(values,lo,hi,DELTA,numerical_pad(bounds,choice))
                # Same256 unselected RAW variance aids a paired descriptive comparison.
                confirmation_values=sample_values(confirm,method,alpha,choice)
                charged.append(dict(case_id=case['case_id'],policy_family=case['policy_family'],
                    policy_index=case['policy_index'],checkpoint_seed=case['checkpoint_seed'],profile=case['profile'],
                    alpha=alpha,method=method,choice=choice,certificate=certificate,
                    positive=certificate['lower']>0,confirmation_mean=float(confirmation_values.mean()),
                    confirmation_variance=float(confirmation_values.var(ddof=1)),bounds=bounds))
        pad=numerical_pad(bounds)
        cost=empirical_bernstein([row['cost'] for row in allrows],0,bounds['discounted_cost_bound'],DELTA,pad)
        access_lower=float(bounds['unavoidable_discounted_access_cost'])
        cost_results.append(dict(case_id=case['case_id'],profile=case['profile'],certificate=cost,
            positive=cost['lower']>0,analytic_unavoidable_cost_lower=access_lower,
            booked_expectation_upper=-max(cost['lower'],access_lower)))
        case_diagnostics.append(dict(case_id=case['case_id'],policy_family=case['policy_family'],
            profile=case['profile'],n_episodes=320,n_confirmation=256,
            mean_maturities=float(np.mean([row['native_activity']['maturity_events'] for row in confirm])),
            confirmation_episodes_with_maturity=sum(row['native_activity']['maturity_events']>0 for row in confirm),
            max_event_identity_error=max(row['identity_error'] for row in allrows),
            source_and_phase_metadata_sha256={phase:sha(phase_path(directory,case,phase)/'metadata.json') for phase in phase_rows}))
    assert len(charged)==320 and len(cost_results)==20
    counts={str(alpha):{method:sum(row['positive'] for row in charged if row['alpha']==alpha and row['method']==method)
        for method in METHODS if not method.startswith('cash_') or alpha==0.} for alpha in ALPHAS}
    profile_counts={profile:{str(alpha):{method:sum(row['positive'] for row in charged if row['profile']==profile and row['alpha']==alpha and row['method']==method)
        for method in METHODS if not method.startswith('cash_') or alpha==0.} for alpha in ALPHAS} for profile in ('joint_fault',)}
    result=dict(status='COMPLETE EXPLORATORY AUDIT',created_utc=utc(),n_existing_learned_policies=20,
        n_cases=20,n_audit_episodes=6400,n_native_audit_steps=614400,method_counts=counts,
        profile_counts=profile_counts,positive_cost_limits=sum(x['positive'] for x in cost_results),
        charged_results=charged,cost_results=cost_results,case_diagnostics=case_diagnostics,
        source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json'),selection_freeze_sha256=sha(directory/'SELECTION_FREEZE.json'),
        development_stage=read(directory/'development_stage_complete.json'),
        confirmation_stage=read(directory/'confirm_stage_complete.json'),
        actual_work=actual_work,model_and_selection=read(directory/'SELECTION_FREEZE.json'),
        multiplicity=dict(family_limits=340,total_delta=.05,delta_per_limit=DELTA),
        scope='Fixed contextual simulator fair-sign intervention; financial component only; exploratory; original policies are frozen')
    assert actual_work['audit_native_transitions']==614400 and actual_work['warm_native_transitions']==122880
    output=HERE/f'joint_fault_results_{directory.name}.json'
    write(output,result)
    csvpath=HERE/f'joint_fault_results_{directory.name}.csv'
    with csvpath.open('x',encoding='utf-8',newline='') as stream:
        fields=('case_id','policy_family','checkpoint_seed','profile','alpha','method','n','mean','variance','radius','lower','positive','cap','beta','budget','support_lower','support_upper')
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in charged:
            record={key:row[key] for key in ('case_id','policy_family','checkpoint_seed','profile','alpha','method','positive')}
            record.update(row['certificate']);record.update(row['choice'])
            writer.writerow({key:record[key] for key in fields})
    write(HERE/f'joint_fault_completion_{directory.name}.json',dict(complete=True,utc=utc(),
        results_sha256=sha(output),csv_sha256=sha(csvpath),source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json')))
    print(json.dumps(dict(status='COMPLETE',method_counts=counts,profile_counts=profile_counts,
        positive_cost_limits=result['positive_cost_limits'],result=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','aggregate'))
    parser.add_argument('--coordinator-release',default='')
    parser.add_argument('--run-id',default='run_001');args=parser.parse_args()
    if not args.run_id.startswith('run_') or not args.run_id.replace('_','').isalnum():raise ValueError('Simple run_id required')
    if not args.coordinator_release.strip():raise RuntimeError('HELD: explicit coordinator release required; no freeze/native execution authorized yet')
    directory=HERE/'execution'/args.run_id
    if args.command=='freeze':freeze(directory);return
    if args.command=='aggregate':aggregate(directory);return
    write(directory/'LAUNCH.json',dict(pid=os.getpid(),utc=utc(),command=sys.argv,workers=WORKERS))
    try:
        run_phase(directory,'development')
        fit_and_select(directory)
        run_phase(directory,'confirm')
        aggregate(directory)
    except BaseException:
        write(directory/f'FAILURE_{os.getpid()}.json',dict(utc=utc(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
