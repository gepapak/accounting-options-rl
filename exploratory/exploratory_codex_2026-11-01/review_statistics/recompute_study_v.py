"""Independent saved-result Study V review; never imports registered study code.

Reads only V JSON, T CSV, source/protocol text and the published V summaries.
Writes exclusively beside this script. No environments, fits, policies or Git.
The rank-sum p-values are independently calculated and cross-checked against
SciPy; bootstrap draws preserve the registered global RNG/call ordering.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY = REPO / 'ACCOUNTING_OPTIONS_2026-10-14/study_v'
T = STUDY.parent / 'study_t'
A_ARMS = ('PLAIN','ORACLE','PMR','RUIN1','RUIN5','RUIN25','MV0.05','MV0.2','MV1')
C_ARMS = ('PLAIN','ORACLE','TIR','PMR','RUIN1','RUIN5','RUIN25')
LOW = ('0','0.02','0.05','0.1')
RNG = np.random.default_rng(20261101)
HASHES: dict[str,str] = {}
CHECKS = 0


def check(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked(path):
    HASHES[path.relative_to(REPO).as_posix()] = digest(path)
    return path.read_text(encoding='utf-8')


def mean(x):
    return float(np.mean(np.asarray(x,dtype=float)))


def quantile(x,q):
    return float(np.quantile(np.asarray(x,dtype=float),q,method='linear'))


def rank_test(left,right):
    """P(U <= observed), tied normal approximation + continuity correction."""
    n,m=len(left),len(right)
    ordered=sorted([(x,0) for x in left]+[(x,1) for x in right])
    rank_sum=0.; ties=0; i=0
    while i<n+m:
        j=i+1
        while j<n+m and ordered[j][0]==ordered[i][0]: j+=1
        average_rank=(i+1+j)/2
        rank_sum += sum(label==0 for _,label in ordered[i:j])*average_rank
        t=j-i; ties += t**3-t; i=j
    u=rank_sum-n*(n+1)/2
    variance=n*m/12*((n+m+1)-ties/((n+m)*(n+m-1)))
    if variance==0:
        p=1.
    else:
        z=(u-n*m/2+.5)/math.sqrt(variance)
        p=min(1.,.5*math.erfc(-z/math.sqrt(2)))
    reference=float(mannwhitneyu(left,right,alternative='less').pvalue)
    check(math.isclose(reference,p,rel_tol=5e-13,abs_tol=1e-300), 'Independent tied-rank probability disagrees with SciPy')
    return dict(U=u,p=p,scipy_crosscheck=reference,crosscheck_absolute_difference=abs(reference-p),n_left=n,n_right=m,
                alternative='less',method='tied asymptotic normal, continuity correction')


def ratio_result(arm,oracle):
    ea=np.asarray([r['E'] for r in arm]); eo=np.asarray([r['E'] for r in oracle])
    draws=[]; nonpositive=0
    for _ in range(10000):
        ia=RNG.integers(len(ea),size=len(ea)); io=RNG.integers(len(eo),size=len(eo))
        denominator=float(np.mean(eo[io]))
        nonpositive += denominator<=0
        draws.append(float(np.mean(ea[ia]))/denominator)
    ratio=float(np.mean(ea)/np.mean(eo)); lower=quantile(draws,.05)
    return dict(ratio=ratio,lower_95=lower,arm_mean=mean(ea),oracle_mean=mean(eo),n_arm=len(ea),n_oracle=len(eo),
        bootstrap_samples=10000,nonpositive_bootstrap_denominators=nonpositive,
        supported=bool(mean(eo)>0 and ratio>=.8 and lower>=.6))


def remove_result(arm,plain,oracle):
    result=rank_test([r['X'] for r in arm],[r['X'] for r in plain])
    result.update(arm_X=mean([r['X'] for r in arm]),plain_X=mean([r['X'] for r in plain]),oracle_X=mean([r['X'] for r in oracle]))
    result['supported']=bool(result['p']<.05 and result['arm_X']<=result['oracle_X']+.1)
    return result


def both_result(rows,arms,oracle):
    out={}; xo=mean([r['X'] for r in oracle]); eo=mean([r['E'] for r in oracle])
    for arm in arms:
        selected=[r for r in rows if r['arm']==arm]
        x=mean([r['X'] for r in selected]); e=mean([r['E'] for r in selected])
        out[arm]=dict(X=x,E=e,both=bool(x<=xo+.1 and e>=.8*eo),X_ceiling=xo+.1,E_floor=.8*eo)
    return dict(arms=out,supported=bool(eo>0 and not any(r['both'] for r in out.values())))


def expected_roster():
    a={('S_DK1',arm,seed) for arm in A_ARMS for seed in range(150,158)}
    b={(s,'PMR',seed) for s in ('T_DK2','T_DK1M') for seed in range(160,168)}
    c={('S_DK1+Z',arm,seed) for arm in C_ARMS for seed in range(170,178)}
    return a|b|c


def load_results():
    rows=[]
    for path in sorted((STUDY/'results_v').glob('*.json')):
        d=json.loads(tracked(path)); setting,arm,seed=d['setting'],d['arm'],d['seed']
        check(path.name==f'{setting}__{arm}__s{seed}.json','Filename/run identity')
        check(d['steps']==1000000,'Registered requested PPO budget')
        expected={f'{e}|{sign}|{rem}' for e in ('0','0.02','0.05','0.1','0.25','0.5','1','2') for sign in ('-1','+1') for rem in ('0.05','0.5')}
        check(set(d['lev'])==expected and all(v in (-16.,-4.,-1.,-.25,0.,.25,1.,4.,16.) for v in d['lev'].values()),'All32 valid policy probes')
        x=sum(abs(d['lev'][f'{e}|{sign}|{rem}'])==16 for e in LOW for sign in ('-1','+1') for rem in ('0.05','0.5'))/16
        row=dict(setting=setting,arm=arm,seed=seed,X=x,E=d['test_edge']['booked_pct_K0'],Ez=d['test_zero']['booked_pct_K0'],
            placebo_reported=d['test_placebo']['reported_pct_K0'],placebo_forgiven=d['test_placebo']['forgiven_pct_K0'],
            pool_mean=d.get('pool_mean_eur_per_mwh'),audit={})
        for market,key in (('edge','audit_edge'),('placebo','audit_placebo')):
            audit=d[key]; episodes=audit['episodes']; booked=episodes['booked']
            check(set(episodes)=={'booked','reported','spr','tir','pmr'} and all(len(v)==100 for v in episodes.values()),'All100 audit episodes/estimators retained')
            check(all(math.isfinite(v) for values in episodes.values() for v in values),'No nonfinite saved audit values')
            check(mean(booked)==audit['booked'],'Independent mean of saved booked ledger episode returns')
            check(abs(audit['booked']-d['test_'+market]['booked_pct_K0'])<1e-6,'Registered replay booked assertion')
            check(abs(audit['reported']-d['test_'+market]['reported_pct_K0'])<1e-6,'Registered replay reported assertion')
            measure={}
            for est in ('reported','spr','tir','pmr'):
                values=episodes[est]
                check(mean(values)==audit[est],'Saved estimator mean from100 episode values')
                errors=np.abs(np.asarray(values)-np.asarray(booked))
                check(mean(errors)==audit['mae_'+est],'Saved mean per-episode absolute error')
                measure[est]=dict(error_of_mean=abs(mean(values)-mean(booked)),episode_MAE=mean(errors),largest_episode_error=float(np.max(errors)))
            row['audit'][market]=measure
        check(all(math.isfinite(row[k]) for k in ('X','E','Ez','placebo_reported','placebo_forgiven')),'Finite registered summaries')
        row['probe_map']=d['lev']; rows.append(row)
    check(len(rows)==144 and {(r['setting'],r['arm'],r['seed']) for r in rows}==expected_roster(),'Exact144 registered roster without substitution/smoke')
    check(len({(r['setting'],r['arm'],r['seed']) for r in rows})==144,'No duplicate agent')
    return rows


def main(output_name='RECOMPUTATION_STUDY_V.json'):
    output=HERE/output_name
    check(output.resolve().parent==HERE.resolve() and output.suffix=='.json','Report stays in owned review directory')
    for path in (STUDY/'PREREGISTRATION_V.md',STUDY/'aggregate_v.py',STUDY/'run_v.py',STUDY/'rewards_v.py',T/'envs_t.py',T/'spr_t.py',STUDY.parent/'twin_reward/twin_reward.py'):
        tracked(path)
    rows=load_results()
    reference=[]
    for r in csv.DictReader(tracked(T/'runs_t.csv').splitlines()):
        if r['setting'] in ('T_DK2','T_DK1M') and r['arm'] in ('PLAIN','ORACLE'):
            reference.append(dict(setting=r['setting'],arm=r['arm'],seed=int(r['seed']),X=float(r['X']),E=float(r['E'])))
    check(len(reference)==40 and len({(r['setting'],r['arm'],r['seed']) for r in reference})==40,'Both10+10 frozen StudyT reference groups')
    get=lambda data,setting,arm:[r for r in data if r['setting']==setting and r['arm']==arm]
    verdicts={}
    p=get(rows,'S_DK1','PLAIN');o=get(rows,'S_DK1','ORACLE');m=get(rows,'S_DK1','PMR')
    v=rank_test([r['X'] for r in o],[r['X'] for r in p]);v['supported']=v['p']<.05;verdicts['VA1']=v
    verdicts['VA2']=remove_result(m,p,o);verdicts['VA3']=ratio_result(m,o)
    part=[r for r in rows if r['setting']=='S_DK1']
    verdicts['VA4']=both_result(part,('MV0.05','MV0.2','MV1'),o);verdicts['VA5']=both_result(part,('RUIN1','RUIN5','RUIN25'),o)
    if not verdicts['VA1']['supported']:
        for key in ('VA2','VA3','VA4','VA5'): verdicts[key]['supported']=None
    for setting in ('T_DK2','T_DK1M'):
        m=get(rows,setting,'PMR');p=get(reference,setting,'PLAIN');o=get(reference,setting,'ORACLE')
        check(len(m)==8 and len(p)==len(o)==10,'Unpaired8vs10 PartB sample sizes')
        verdicts[f'VB-{setting}-1']=remove_result(m,p,o);verdicts[f'VB-{setting}-2']=ratio_result(m,o)
    p=get(rows,'S_DK1+Z','PLAIN');o=get(rows,'S_DK1+Z','ORACLE');m=get(rows,'S_DK1+Z','PMR');t=get(rows,'S_DK1+Z','TIR')
    v=rank_test([r['X'] for r in o],[r['X'] for r in p]);v['supported']=v['p']<.05;verdicts['VC1']=v
    verdicts['VC2']=remove_result(m,p,o);verdicts['VC3']=ratio_result(m,o);verdicts['VC4']=remove_result(t,p,o);verdicts['VC5']=ratio_result(t,o)
    verdicts['VC6']=both_result([r for r in rows if r['setting']=='S_DK1+Z'],('RUIN1','RUIN5','RUIN25'),o)
    if not verdicts['VC1']['supported']:
        for key in ('VC2','VC3','VC4','VC5','VC6'): verdicts[key]['supported']=None
    audited=[r for r in rows if r['setting'] in ('S_DK1','S_DK1+Z')]; gamblers=[r for r in audited if r['X']>=.25]
    vd1={market:max(r['audit'][market]['tir']['error_of_mean'] for r in audited) for market in ('edge','placebo')}
    verdicts['VD1']=dict(max_error=vd1,supported=max(vd1.values())<=1.)
    reported_errors=[r['audit']['placebo']['reported']['error_of_mean'] for r in gamblers]
    check(all(x>0 for x in reported_errors),'VD2 denominators positive')
    medrep=statistics.median(reported_errors)
    medratio=statistics.median(r['audit']['placebo']['tir']['error_of_mean']/r['audit']['placebo']['reported']['error_of_mean'] for r in gamblers)
    verdicts['VD2']=dict(gamblers=len(gamblers),median_reported_error=medrep,median_TIR_to_reported=medratio,
        supported=bool(len(gamblers)>=5 and medrep>=5 and medratio<=.05) if len(gamblers)>=5 else None)
    check(len(verdicts)==17,'All17 independently computed')
    published={}
    for line in tracked(STUDY/'VERDICTS_V.md').splitlines():
        if line.startswith('| V'):
            cells=[s.strip() for s in line.split('|')[1:-1]]
            if cells[0] in verdicts:published[cells[0]]=cells[-1]
    check(set(published)==set(verdicts),'All17 stored verdict labels present')
    for key,value in verdicts.items():
        own='NOT TESTABLE' if value['supported'] is None else 'SUPPORTED' if value['supported'] else 'NOT SUPPORTED'
        check(own==published[key],'Recomputed verdict differs: '+key);value['stored_verdict']=published[key]
    csv_errors=defaultdict(float)
    csv_rows=list(csv.DictReader(tracked(STUDY/'runs_v.csv').splitlines()))
    by_id={(r['setting'],r['arm'],r['seed']):r for r in rows}
    check(len(csv_rows)==144,'Stored CSV complete144')
    for record in csv_rows:
        r=by_id[(record['setting'],record['arm'],int(record['seed']))]
        for key in ('X','E','Ez','placebo_reported','placebo_forgiven'):
            csv_errors[key]=max(csv_errors[key],abs(float(record[key])-r[key]))
    check(max(csv_errors.values())<1e-12,'StoredCSV vsrawJSON summary mismatch')
    group_summaries={}
    for setting,arms in (('S_DK1',A_ARMS),('T_DK2',('PMR',)),('T_DK1M',('PMR',)),('S_DK1+Z',C_ARMS)):
        for arm in arms:
            group=get(rows,setting,arm)
            group_summaries[f'{setting}/{arm}']=dict(n=len(group),mean_X=mean([r['X'] for r in group]),mean_E=mean([r['E'] for r in group]),
                runs_X_positive=sum(r['X']>0 for r in group),runs_X_ge_quarter=sum(r['X']>=.25 for r in group),
                mean_Ez=mean([r['Ez'] for r in group]),unique_test_E=len({r['E'] for r in group}),
                distinct_full_probe_maps=len({json.dumps(r['probe_map'],sort_keys=True) for r in group}))
    maxima={est:{market:max(r['audit'][market][est]['error_of_mean'] for r in audited) for market in ('edge','placebo')} for est in ('reported','spr','tir','pmr')}
    episode_limits={market:dict(max_agent_MAE=max(r['audit'][market]['tir']['episode_MAE'] for r in audited),
        largest_single_episode_TIR_error=max(r['audit'][market]['tir']['largest_episode_error'] for r in audited)) for market in ('edge','placebo')}
    posthoc={}
    for label,setting,arm,reported in (('PMR_over_RUIN5_A','S_DK1','PMR','0.113'),('PMR_over_RUIN5_C','S_DK1+Z','PMR','0.042'),('TIR_over_RUIN5_C','S_DK1+Z','TIR','0.193')):
        # Reverse the order so the independently implemented less test asks
        # whether RUIN5 is smaller than the named candidate.
        value=rank_test([r['E'] for r in get(rows,setting,'RUIN5')],[r['E'] for r in get(rows,setting,arm)])
        check(f"{value['p']:.3f}"==reported,'Posthoc probability differs from reported rounding')
        posthoc[label]=dict(**value,status='POSTHOC_NOT_A_REGISTERED_VERDICT',reported_rounding=reported)
    check(all(digest(REPO/name)==value for name,value in HASHES.items()),'All read inputs remain unchanged')
    report=dict(status='PASS_INDEPENDENT_STUDY_V_17_VERDICTS',checks=CHECKS,
        supported=sum(v['supported'] is True for v in verdicts.values()),unsupported=[k for k,v in verdicts.items() if v['supported'] is False],
        verdicts=verdicts,group_summaries=group_summaries,audit_max_error_of_mean=maxima,audit_TIR_episode_error_limits=episode_limits,
        gamblers_by_arm=dict(Counter(r['arm'] for r in gamblers)),csv_vs_json_maximum_errors=dict(csv_errors),
        PMR_zero_X=sum(r['X']==0 for r in rows if r['arm']=='PMR'),PMR_total=sum(r['arm']=='PMR' for r in rows),
        plain_positive_X=sum(r['X']>0 for r in audited if r['arm']=='PLAIN'),plain_total=sum(r['arm']=='PLAIN' for r in audited),
        posthoc_edge_MW=posthoc,
        inputs_sha256=HASHES,reviewer_sha256=digest(Path(__file__)),versions={p:importlib.metadata.version(p) for p in ('numpy','scipy')},
        source_environment_model_training_calls=0,
        scope='Independent reconstruction from saved agent JSON and StudyT CSV. No registered module imported or executed. Verifies saved100-episode means/MAEs and registered replay-summary assertions; no saved transition ledger/weights exist here for independent trajectory replay. Timing/hash-prefix audit owned separately by root. No new hypotheses or multiplicity-adjusted claims.')
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],checks=CHECKS,supported=report['supported'],unsupported=report['unsupported'],report_sha256=digest(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='RECOMPUTATION_STUDY_V.json')
    main(parser.parse_args().output)
