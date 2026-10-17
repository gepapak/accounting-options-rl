"""Frozen exploratory native-policy audit; no simulations on import.

python run_native_audit.py freeze
python run_native_audit.py run
python run_native_audit.py aggregate
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
CODE=HERE.parents[1]
PAPER1=CODE.parents[1]
ENGINE=PAPER1/'Prototype5'
SEEDS=(7,42,123,2025,3007,5001,8102,9005,10001,11202)
CAPS=(1000,4000,16000,64000,256000,1024000,4096000,16384000)
BETAS=(0.,.25,.5,1.,2.)
BUDGET_RATIOS=(0.,1/16,1/4,1.,4.)
N_FIT,N_TUNE,N_CONFIRM=32,32,256
DELTA=.05/520
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
    output=[]
    for family in ('plain_mappo','feasible_mappo'):
        for seed in SEEDS:
            policy_index=len(output)//2
            if family=='plain_mappo':
                checkpoint=(ENGINE/'batch_tier_phase_runs/prototype5_mappo_marl_final_v1'/
                    f'seed{seed}'/f'tier1_seed{seed}'/'final_models')
            else:
                checkpoint=(ENGINE/'Ablations/batch_tier_phase_runs/prototype5_mechanism_ablations_final_v1/feasible_action_full'/
                    f'seed{seed}'/f'tier1_forecast_utilization_seed{seed}'/'final_models')
            for profile_index,profile in enumerate(('strict','vulnerable')):
                output.append(dict(case_id=len(output),policy_index=policy_index,policy_family=family,
                    checkpoint_seed=seed,checkpoint_dir=str(checkpoint),profile_index=profile_index,
                    profile=profile,cash_fraction=1e-4,horizon=96,burn_in=384))
    assert len(output)==40 and len({x['policy_index'] for x in output})==20
    return output


def freeze(directory):
    directory.mkdir(parents=True,exist_ok=True)
    source_names=('EXPLORATORY_EXECUTION_PLAN.md','run_native_audit.py','native_phase_worker.py',
        'native_engine_adapter.py','native_support.py','sample_estimators.py','audit_models.py',
        'SAMPLE_ONLY_THEORY.md','exact_sample_only_checks.py','exact_sample_only_checks.json',
        'checkpoint_inventory.json')
    source={name:sha(HERE/name) for name in source_names}
    native={str(p.relative_to(ENGINE)):sha(p) for p in sorted(ENGINE.glob('*.py'))}
    artifacts={}
    for case in cases():
        root=Path(case['checkpoint_dir'])
        for name in ('investor_0_policy.zip','battery_operator_0_policy.zip','training_config.json'):
            path=root/name;artifacts[str(path.relative_to(ENGINE))]=sha(path)
    snapshot=directory/'source_snapshot';snapshot.mkdir()
    for name in source:shutil.copyfile(HERE/name,snapshot/name)
    write(directory/'cases.json',cases())
    manifest=dict(status='FROZEN BEFORE STATISTICAL ENDPOINTS',created_utc=utc(),
        classification='exploratory, internally hash-frozen; not a registered learner study',
        source_sha256=source,native_source_sha256=native,checkpoint_sha256=artifacts,
        cases_sha256=sha(directory/'cases.json'),workers=WORKERS,n_fit=N_FIT,n_tune=N_TUNE,
        n_confirm=N_CONFIRM,n_audit_episodes=12800,n_audit_steps=1228800,
        n_warmup_steps_two_stages=245760,family_limits=520,delta_per_limit=DELTA,
        python=sys.version,platform=sys.platform)
    write(directory/'SOURCE_FREEZE.json',manifest)
    print(json.dumps(dict(status='FROZEN',directory=str(directory),sha256=sha(directory/'SOURCE_FREEZE.json'))),flush=True)


def verify_freeze(directory):
    record=read(directory/'SOURCE_FREEZE.json')
    assert record['cases_sha256']==sha(directory/'cases.json')
    assert read(directory/'cases.json')==cases()
    for group,root in (('source_sha256',HERE),('native_source_sha256',ENGINE),('checkpoint_sha256',ENGINE)):
        for name,digest in record[group].items():
            if sha(root/name)!=digest:raise RuntimeError(f'Frozen source/input changed: {group}/{name}')
    return record


def job(case,phase,directory,models_dir):
    from native_phase_worker import run_case
    return run_case(case,phase=phase,output_dir=str(directory),
        n=N_FIT if phase=='development' else N_CONFIRM,models_dir=models_dir)


def run_phase(directory,phase):
    verify_freeze(directory)
    start=time.perf_counter();cpu=time.process_time()
    model_dir=str(directory/'models') if phase=='confirm' else None
    if phase=='confirm':
        selection=read(directory/'SELECTION_FREEZE.json')
        assert selection['selection_sha256']==sha(directory/'selection.json')
        assert selection['model_manifest_sha256']==sha(directory/'models/audit_models_manifest.json')
    # Preserve incomplete attempts: a worker refuses overwriting partial files.
    completed=[]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(job,case,phase,str(directory),model_dir):case for case in cases()}
        for future in as_completed(futures):
            case=futures[future]
            result=future.result();completed.append(result)
            print(json.dumps(dict(utc=utc(),phase=phase,case_id=case['case_id'],
                completed_cases=len(completed),total_cases=40)),flush=True)
    write(directory/f'{phase}_stage_complete.json',dict(complete=True,utc=utc(),
        phase=phase,results=completed,wall_seconds=time.perf_counter()-start,
        parent_cpu_seconds=time.process_time()-cpu,workers=WORKERS))
    verify_freeze(directory)


def phase_path(directory,case,phase):return directory/f"case_{case['case_id']:02d}"/phase


def evaluate_saved_branches(phase_dir,models,base_rows):
    import numpy as np
    data=np.load(phase_dir/'branch_features.npz',allow_pickle=False)
    offsets=data['episode_offsets']
    assert len(offsets)==len(base_rows)+1
    predictions={name:(model.predict(data['plus'])-model.predict(data['minus']))/2 for name,model in models.items()}
    output=[]
    for i,row in enumerate(base_rows):
        row=dict(row);lo,hi=int(offsets[i]),int(offsets[i+1])
        row['maturity_coins']=data['coins'][lo:hi].tolist()
        for name,values in predictions.items():row['delta_'+name]=values[lo:hi].tolist()
        output.append(row)
    return output


def bound_for_case(directory,case,phase='fit'):
    metadata=read(phase_path(directory,case,phase)/'metadata.json')
    return metadata['bounds']


def fit_and_select(directory):
    import numpy as np
    from audit_models import fit_models,load_models
    from sample_estimators import METHODS,ALPHAS,select_choice
    verify_freeze(directory)
    assert (directory/'development_stage_complete.json').exists()
    fitX=[];fity=[];inputs={}
    for case in cases():
        path=phase_path(directory,case,'fit')/'fit_rows.npz'
        data=np.load(path,allow_pickle=False);fitX.append(data['X']);fity.append(data['y'])
        inputs[str(case['case_id'])]=sha(path)
    model_metadata=fit_models(np.concatenate(fitX),np.concatenate(fity),directory/'models',seed=29300000)
    models=load_models(directory/'models')
    selections=[];search_records=[]
    cpu=time.process_time();start=time.perf_counter()
    for case in cases():
        fitdir=phase_path(directory,case,'fit');tunedir=phase_path(directory,case,'tune')
        fitrows=rows(fitdir/'rows.jsonl');tunerows=rows(tunedir/'rows.jsonl')
        learned_tune=evaluate_saved_branches(tunedir,models,tunerows)
        assert len(fitrows)==N_FIT and len(tunerows)==N_TUNE
        bounds=bound_for_case(directory,case)
        assert bounds['discounted_cost_bound']==bound_for_case(directory,case,'tune')['discounted_cost_bound']
        for alpha in ALPHAS:
            for method in METHODS:
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
        fit_input_sha256=inputs,model_metadata=model_metadata,
        selection_cpu_seconds=time.process_time()-cpu,selection_wall_seconds=time.perf_counter()-start,
        source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json')))
    print(json.dumps(dict(status='ALL MODELS AND 480 CHOICES FROZEN BEFORE CONFIRM',utc=utc())),flush=True)


def aggregate(directory):
    import numpy as np
    from sample_estimators import METHODS,ALPHAS,sample_values,deterministic_support,empirical_bernstein,numerical_pad
    verify_freeze(directory)
    assert (directory/'confirm_stage_complete.json').exists()
    selection={(row['case_id'],row['alpha'],row['method']):row['choice'] for row in read(directory/'selection.json')}
    charged=[];cost_results=[];case_diagnostics=[]
    actual_work=dict(audit_native_transitions=0,warm_native_transitions=0,actor_forward_calls=0,
        warm_actor_forward_calls=0,branch_injection_arithmetic_evaluations=0,
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
            for key in ('warm_native_transitions','warm_actor_forward_calls','zero_step_support_initializations'):
                actual_work[key]+=preparation[key]
            actual_work['preparation_cpu_seconds']+=preparation['cpu_seconds']
        assert [len(phase_rows[k]) for k in ('fit','tune','confirm')]==[32,32,256]
        bounds=bound_for_case(directory,case)
        for phase in ('tune','confirm'):
            current=bound_for_case(directory,case,phase)
            for name in ('discounted_cost_bound','discounted_gross_bound'):
                assert current[name]==bounds[name],(case['case_id'],phase,name)
        allrows=sum(phase_rows.values(),[]);confirm=phase_rows['confirm']
        for alpha in ALPHAS:
            for method in METHODS:
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
    counts={str(alpha):{method:sum(row['positive'] for row in charged if row['alpha']==alpha and row['method']==method)
        for method in METHODS} for alpha in ALPHAS}
    profile_counts={profile:{str(alpha):{method:sum(row['positive'] for row in charged if row['profile']==profile and row['alpha']==alpha and row['method']==method)
        for method in METHODS} for alpha in ALPHAS} for profile in ('strict','vulnerable')}
    result=dict(status='COMPLETE EXPLORATORY AUDIT',created_utc=utc(),n_existing_learned_policies=20,
        n_cases=40,n_audit_episodes=12800,n_native_audit_steps=1228800,method_counts=counts,
        profile_counts=profile_counts,positive_cost_limits=sum(x['positive'] for x in cost_results),
        charged_results=charged,cost_results=cost_results,case_diagnostics=case_diagnostics,
        source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json'),selection_freeze_sha256=sha(directory/'SELECTION_FREEZE.json'),
        development_stage=read(directory/'development_stage_complete.json'),
        confirmation_stage=read(directory/'confirm_stage_complete.json'),
        actual_work=actual_work,model_and_selection=read(directory/'SELECTION_FREEZE.json'),
        multiplicity=dict(family_limits=520,total_delta=.05,delta_per_limit=DELTA),
        scope='Fixed contextual simulator fair-sign intervention; financial component only; exploratory; original policies are frozen')
    assert actual_work['audit_native_transitions']==1228800 and actual_work['warm_native_transitions']==245760
    output=HERE/f'native_results_{directory.name}.json'
    write(output,result)
    csvpath=HERE/f'native_results_{directory.name}.csv'
    with csvpath.open('x',encoding='utf-8',newline='') as stream:
        fields=('case_id','policy_family','checkpoint_seed','profile','alpha','method','n','mean','variance','radius','lower','positive','cap','beta','budget','support_lower','support_upper')
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in charged:
            record={key:row[key] for key in ('case_id','policy_family','checkpoint_seed','profile','alpha','method','positive')}
            record.update(row['certificate']);record.update(row['choice'])
            writer.writerow({key:record[key] for key in fields})
    write(HERE/f'native_completion_{directory.name}.json',dict(complete=True,utc=utc(),
        results_sha256=sha(output),csv_sha256=sha(csvpath),source_freeze_sha256=sha(directory/'SOURCE_FREEZE.json')))
    print(json.dumps(dict(status='COMPLETE',method_counts=counts,profile_counts=profile_counts,
        positive_cost_limits=result['positive_cost_limits'],result=str(output))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','aggregate'))
    parser.add_argument('--run-id',default='run_001');args=parser.parse_args()
    if not args.run_id.startswith('run_') or not args.run_id.replace('_','').isalnum():raise ValueError('Simple run_id required')
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
