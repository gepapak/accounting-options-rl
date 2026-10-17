"""Fixed descriptive cash-control ablations after complete private/public PASS.

No additional confidence limit, simulator query, fitting or setting selection.
"""
from collections import defaultdict
from datetime import datetime,timezone
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def require(value,message):
    if not value:raise AssertionError(message)


def control(deltas,coins,beta,budget):
    remaining=float(budget);terms=[]
    for delta,coin in zip(deltas,coins):
        amplitude=min(abs(float(beta)*float(delta)),remaining)
        terms.append(coin*(1 if delta>0 else -1 if delta<0 else 0)*amplitude)
        remaining=max(remaining-amplitude,0.)
    return math.fsum(terms)


def moments(values):
    require(len(values)==256 and all(math.isfinite(v) for v in values),'same256finiteCONFIRMepisodes')
    mean=math.fsum(values)/len(values)
    return dict(mean=mean,variance=math.fsum((v-mean)**2 for v in values)/(len(values)-1))


def ratio(a,b):
    if b>0:return a/b,'finite'
    return None,'both_zero' if a==0 else 'zero_reference_positive_full'


def execute(run_id):
    require(run_id.startswith('run_') and run_id.replace('_','').isalnum(),'simple run ID')
    private_path=HERE/'validation'/('JOINT_FAULT_VERIFICATION_'+run_id+'.json')
    public=HERE/'results'/run_id
    required=[private_path,HERE/('joint_fault_completion_'+run_id+'.json'),HERE/'execution'/run_id/'confirm_stage_complete.json',public/'MANIFEST.json',public/'REPRODUCTION.json']
    require(all(p.is_file() for p in required),'NOT_READY: complete private/public gates before endpoint reads')
    private,manifest,replay=read(private_path),read(public/'MANIFEST.json'),read(public/'REPRODUCTION.json')
    require(private['status']=='PASS' and private['bounds_checked']==340 and private['nuisance_regressor_prediction_rechecked'] is True,'full private340/fourcriticPASS')
    require(private['verifier_sha256']==sha(HERE/'validation/verify_joint_fault_results.py'),'executed independent verifier source')
    require(manifest['complete'] is True and replay['status']=='PASS' and replay['bounds_recomputed']==340 and replay['manifest_sha256']==sha(public/'MANIFEST.json'),'complete archived public340 replay')
    module_path=public/'reproduce_public_joint_fault.py'
    require(sha(module_path)==manifest['file_sha256']['reproduce_public_joint_fault.py'],'archived reproducer identity')
    spec=importlib.util.spec_from_file_location('joint_fault_diag_public',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fresh=module.verify_directory(public)
    require(fresh['status']=='PASS' and fresh['bounds_recomputed']==340,'fresh public340 replay')
    require(read(public/'provenance.json')['private_result_sha256']==private['result_sha256'],'public/private result identity')
    result=read(public/'results.json')
    with gzip.open(public/'paths.jsonl.gz','rt',encoding='utf-8') as stream:paths=[json.loads(s) for s in stream if s.strip()]
    grouped=defaultdict(list)
    for row in paths:
        if row['phase']=='confirm':grouped[row['case_id']].append(row)
    lookup={(r['case_id'],r['alpha'],r['method']):r for r in result['charged_results']}
    records=[]
    for cid in range(20):
        rows=sorted(grouped[cid],key=lambda r:r['episode_id'])
        require([r['episode_id'] for r in rows]==list(range(256)),'all fixed commonCONFIRMpaths')
        raw=[r['gross']-r['cost']+r['forgiveness'] for r in rows]
        for method in ('cash_ridge_control','cash_mlp_control'):
            saved=lookup[cid,0.,method];choice=saved['choice']
            cap,beta,B=float(choice['cap']),float(choice['beta']),float(choice['budget'])
            full=[w-control(r['delta_'+method.removesuffix('_control')],r['maturity_coins'],beta,B) for w,r in zip(raw,rows)]
            local=[w-control(r['delta_cash_local'],r['maturity_coins'],beta,B) for w,r in zip(raw,rows)]
            arrays={'uncapped_full':full,'uncapped_local':local,'uncapped_zero':raw,
                'same_cap_full':[min(v,cap) for v in full],
                'same_cap_local':[min(v,cap) for v in local],'same_cap_zero':[min(v,cap) for v in raw]}
            record=dict(case_id=cid,policy_family=saved['policy_family'],method=method,n=256,
                beta=beta,budget=B,cap=cap,selected_certified=saved['positive'],
                separately_selected_local_certified=lookup[cid,0.,'cash_local_control']['positive'],
                separately_selected_cashraw_certified=lookup[cid,0.,'cash_capped_raw']['positive'])
            for key,values in arrays.items():
                stats=moments(values)
                record[key+'_mean']=stats['mean'];record[key+'_variance']=stats['variance']
            for kind in ('uncapped','same_cap'):
                for reference in ('local','zero'):
                    value,status=ratio(record[kind+'_full_variance'],record[kind+'_'+reference+'_variance'])
                    record[kind+'_variance_ratio_full_to_'+reference]=value
                    record[kind+'_ratio_status_full_to_'+reference]=status
            for key,values in (('full',full),('local',local),('zero',raw)):
                record['same_cap_'+key+'_clipped_fraction']=sum(v>cap for v in values)/256
            records.append(record)
    require(len(records)==40,'both cashlearned controls for all20actors')
    output=HERE/'diagnostics'/run_id;output.mkdir(parents=True,exist_ok=False)
    provenance=dict(source_sha256=sha(__file__),plan_sha256=sha(HERE/'EXPLORATORY_EXECUTION_PLAN.md'),private_verification_sha256=sha(private_path),
        public_manifest_sha256=sha(public/'MANIFEST.json'),public_paths_sha256=sha(public/'paths.jsonl.gz'),fresh_public_bounds=340)
    payload=dict(status='COMPLETE DESCRIPTIVE CASH CONTROL ABLATIONS',created_utc=datetime.now(timezone.utc).isoformat(),
        provenance=provenance,records=records,new_native_queries=0,new_model_fits=0,new_confidence_limits=0,
        interpretation='Same frozen learned cap/beta/B on identical256CONFIRM paths; zero-future means known current RAWdelta only, zero-control means none. No retuning or formal superiority test. Ratios controlled/reference preserve undefined0/0 andpositive/0; selected/capped and uncapped sample variances are descriptive.')
    with (output/'CONTROL_ABLATIONS.json').open('x',encoding='utf-8') as stream:json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False)
    with (output/'control_ablations.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (output/'README.md').open('x',encoding='utf-8') as stream:stream.write('# Frozen-setting descriptive ablations\n\nForty cashlearned choices, all20actors, same256CONFIRMpaths. Hold cap/beta/B fixed and replace only RAWfuture q byzero or removeallcontrol. Theseare descriptive sample statistics, withnoadditionalCI/family orretuning. See JSONforfullinput/sourcebindings.\n')
    hashes={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    with (output/'COMPLETION.json').open('x',encoding='utf-8') as stream:json.dump(dict(complete=True,file_sha256=hashes,input_sha256=provenance),stream,indent=2,sort_keys=True)
    print(json.dumps(dict(status=payload['status'],records=len(records),output=str(output))))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--run-id',default='run_001');args=parser.parse_args()
    if args.execute:execute(args.run_id)
    else:print(json.dumps(dict(status='HELD UNTIL COMPLETE VERIFIED PUBLIC340',new_native_queries=0,source_sha256=sha(__file__))))
