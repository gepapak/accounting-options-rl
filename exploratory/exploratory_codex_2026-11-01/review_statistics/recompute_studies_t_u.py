"""Independent T16/U6 saved-result analysis and labelled seed-cluster sensitivity.

Never imports a registered runner, aggregate, environment or trained model.
Uses the previously independently written rank-sum arithmetic in this folder.
"""
from __future__ import annotations
import csv
from collections import defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from recompute_study_v import rank_test

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=REPO/'ACCOUNTING_OPTIONS_2026-10-14'
T=BASE/'study_t';U=BASE/'study_u'
CHECKS=0;HASHES={}
ARMS_T=('PLAIN','ORACLE','SPR','TIR','OM1','OM0.1','OM0.05','OM0.02','OM0.01')
ARMS_U=('ZFG','ZBK','ZSPR','ZTIR','TFG','TBK','TSPR','TTIR')


def check(ok,label):
    global CHECKS
    CHECKS+=1
    if not ok:raise AssertionError(label)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    HASHES[p.relative_to(REPO).as_posix()]=sha(p)
    return p.read_text(encoding='utf-8')


def avg(values):return float(np.mean(values))


def bootstrap(a,b,rng,ratio=False):
    a=np.asarray(a);b=np.asarray(b);values=[];bad=0
    for _ in range(10000):
        am=avg(a[rng.integers(len(a),size=len(a))]);bm=avg(b[rng.integers(len(b),size=len(b))])
        bad+=bool(ratio and bm<=0)
        values.append(am/bm if ratio else am-bm)
    return dict(point=avg(a)/avg(b) if ratio else avg(a)-avg(b),lower_95=float(np.quantile(values,.05)),
                upper_95=float(np.quantile(values,.95)),n_a=len(a),n_b=len(b),nonpositive_ratio_denominators=bad)


def load_t():
    rows=[]
    for p in sorted((T/'results_t').glob('*.json')):
        d=json.loads(read(p));st,arm,seed=d['setting'],d['arm'],d['seed']
        check(p.name==f'{st}__{arm}__s{seed}.json' and d['steps']==1000000,'T run identity/requested budget')
        low=[abs(d['lev'][f'{e}|{z}|{r}'])==16 for e in ('0','0.02','0.05','0.1') for z in ('-1','+1') for r in ('0.05','0.5')]
        rows.append(dict(setting=st,arm=arm,seed=seed,X=sum(low)/16,E=d['test_edge']['booked_pct_K0'],
            placebo_reported=d['test_placebo']['reported_pct_K0'],placebo_forgiven=d['test_placebo']['forgiven_pct_K0']))
    expected={('S_DK1',a,s) for a in ARMS_T for s in range(120,130)}
    expected|={(st,a,s) for st in ('T_DK2','T_DK1M') for a in ('PLAIN','ORACLE','SPR','TIR') for s in range(130,140)}
    check(len(rows)==170 and {(r['setting'],r['arm'],r['seed']) for r in rows}==expected,'Exact170 T roster')
    return rows


def load_u():
    rows=[]
    for p in sorted((U/'results_u').glob('*.json')):
        d=json.loads(read(p));arm,seed,region=d['arm'],d['seed'],d['region']
        check(p.name==f'{arm}__PPO__s{seed}__{region}.json' and d['steps']==1000000 and d['eval_split']=='test','U run identity/evaluation split/budget')
        keys=([f'zombie_n{n:g}_k{k}' for n in (-.5,-.25,-.1,-.02) for k in (96,48,12)] if arm[0]=='Z'
              else [f'L{l:g}_tau{tau}' for l in (.02,.1,.25) for tau in (1,4,16,48,96)])
        values=[d['probes'][key] for key in keys]
        check(all(x in (0.,.25,1.,4.,16.) for x in values),'U absolute leverage outcomes')
        check(len(d['draws'])==20,'All20 evaluation sign draws')
        check(avg([v['booked_pct'] for v in d['draws']])==d['drawmean_booked_pct']
              and avg([v['reported_pct'] for v in d['draws']])==d['drawmean_reported_pct'],'U independently recomputed20-draw means')
        rows.append(dict(arm=arm,setting=arm[0],seed=seed,region=region,X=avg(values),
                         booked=d['drawmean_booked_pct'],reported=d['drawmean_reported_pct']))
    expected={(a,s,r) for a in ARMS_U for s in range(140,145) for r in ('DK1','DK2')}
    check(len(rows)==80 and {(r['arm'],r['seed'],r['region']) for r in rows}==expected,'Exact80 U roster')
    return rows


def verdict_check(path,verdicts):
    labels={}
    for line in read(path).splitlines():
        if line.startswith('| '):
            cells=[x.strip() for x in line.split('|')[1:-1]]
            if cells and cells[0] in verdicts:labels[cells[0]]=cells[-1]
    check(set(labels)==set(verdicts),'Complete stored hypothesis roster')
    for key,row in verdicts.items():
        result='NOT TESTABLE' if row['supported'] is None else 'SUPPORTED' if row['supported'] else 'NOT SUPPORTED'
        check(result==labels[key],'Independent verdict differs '+key)
        row['stored_verdict']=labels[key]


def csv_check(path,rows,keys,values):
    source=list(csv.DictReader(read(path).splitlines()));check(len(source)==len(rows),'CSV roster count')
    index={tuple(str(r[k]) for k in keys):r for r in rows};errors=defaultdict(float)
    for old in source:
        row=index[tuple(old[k] for k in keys)]
        for key in values:errors[key]=max(errors[key],abs(float(old[key])-row[key]))
    check(max(errors.values())<1e-12,'CSV vs raw values')
    return dict(errors)


def analyse_t(rows):
    rng=np.random.default_rng(20261031);out={};groups={}
    for setting,tag in (('S_DK1','TA'),('T_DK2','TB-T_DK2-'),('T_DK1M','TB-T_DK1M-')):
        d=[r for r in rows if r['setting']==setting]
        get=lambda arm:[r for r in d if r['arm']==arm]
        plain,oracle=get('PLAIN'),get('ORACLE')
        test=rank_test([r['X'] for r in oracle],[r['X'] for r in plain])
        out[tag+'1']=dict(**test,supported=test['p']<.05)
        for arm,rid,eid,minratio,minlower in (('SPR','2','3',.8,.6),('TIR','4','5',.9,.75)):
            a=get(arm);test=rank_test([r['X'] for r in a],[r['X'] for r in plain]);x=avg([r['X'] for r in a]);xo=avg([r['X'] for r in oracle])
            out[tag+rid]=dict(**test,mean_X=x,oracle_X=xo,supported=bool(test['p']<.05 and x<=xo+.1))
            ratio=bootstrap([r['E'] for r in a],[r['E'] for r in oracle],rng,ratio=True)
            out[tag+eid]=dict(**ratio,minratio=minratio,minlower=minlower,
                supported=bool(avg([r['E'] for r in oracle])>0 and ratio['point']>=minratio and ratio['lower_95']>=minlower))
        if not out[tag+'1']['supported']:
            for i in ('2','3','4','5'):out[tag+i]['supported']=None
        if setting=='S_DK1':
            both={}
            for arm in ARMS_T[4:]:
                a=get(arm);x=avg([r['X'] for r in a]);e=avg([r['E'] for r in a])
                both[arm]=dict(X=x,E=e,both=x<=avg([r['X'] for r in oracle])+.1 and e>=.8*avg([r['E'] for r in oracle]))
            out['TA6']=dict(arms=both,supported=bool(not any(v['both'] for v in both.values())) if avg([r['E'] for r in oracle])>0 else None)
        for arm in sorted({r['arm'] for r in d}):
            a=get(arm);groups[setting+'/'+arm]=dict(n=len(a),mean_X=avg([r['X'] for r in a]),mean_E=avg([r['E'] for r in a]))
    check(len(out)==16,'All16 T decisions')
    verdict_check(T/'VERDICTS_T.md',out)
    return dict(verdicts=out,group_means=groups,csv_errors=csv_check(T/'runs_t.csv',rows,('setting','arm','seed'),('X','E','placebo_reported','placebo_forgiven')))


def analyse_u(rows):
    rng=np.random.default_rng(20261031);out={};groups={};contrasts=[]
    for setting in ('Z','T'):
        get=lambda arm:[r for r in rows if r['arm']==arm]
        fg=get(setting+'FG');bk=get(setting+'BK');xb=avg([r['X'] for r in bk])
        for index,arm in enumerate((setting+'BK',setting+'SPR',setting+'TIR'),1):
            a=get(arm);b=bootstrap([r['X'] for r in fg],[r['X'] for r in a],rng)
            near=avg([r['X'] for r in a])-xb
            ok=(b['point']>=8 and b['lower_95']>4) if index==1 else (b['point']>=4 and b['lower_95']>0 and near<4)
            key=f'U{index} {setting}'
            out[key]=dict(**b,near_booked=near,supported=bool(ok) if index==1 or out[f'U1 {setting}']['supported'] else None)
            contrasts.append((key,fg,a,near))
        for suffix in ('FG','BK','SPR','TIR'):
            a=get(setting+suffix);groups[setting+suffix]=dict(n=len(a),mean_X=avg([r['X'] for r in a]),
                mean_booked=avg([r['booked'] for r in a]),mean_reported=avg([r['reported'] for r in a]))
    check(len(out)==6,'All6 U decisions')
    verdict_check(U/'VERDICTS_U.md',out)
    # Sensitivity requested after the completed study: never replaces verdicts.
    # Paired exhaustive bootstrap preserves both zones and matched seed across
    # arms. A second independent-arm seed-block bootstrap exposes the pairing
    # choice rather than claiming a uniquely justified dependence structure.
    sensitivity={};cluster_rng=np.random.default_rng(20261031)
    for key,left,right,near in contrasts:
        def blocks(data):
            return np.asarray([avg([r['X'] for r in data if r['seed']==s]) for s in range(140,145)])
        l,r=blocks(left),blocks(right);d=l-r
        exact=[avg(d[list(index)]) for index in itertools.product(range(5),repeat=5)]
        paired=dict(point=avg(d),lower_95=float(np.quantile(exact,.05)),upper_95=float(np.quantile(exact,.95)),exhaustive_resamples=len(exact))
        independent=bootstrap(l,r,cluster_rng)
        def rule(b):
            return bool(b['point']>=8 and b['lower_95']>4) if key.startswith('U1') else bool(b['point']>=4 and b['lower_95']>0 and near<4)
        sensitivity[key]=dict(paired_seed_blocks=paired,independent_arm_seed_blocks=independent,near_booked=near,
            same_numerical_rule_paired=rule(paired),same_numerical_rule_independent=rule(independent),
            per_seed_difference={str(s):float(v) for s,v in zip(range(140,145),d)},
            status='POSTHOC_DEPENDENCE_SENSITIVITY_NOT_REGISTERED_VERDICT')
    return dict(verdicts=out,group_means=groups,seed_cluster_sensitivity=sensitivity,
        csv_errors=csv_check(U/'runs_u.csv',rows,('arm','seed','region'),('X','booked','reported')),
        cluster_scope='Five learner-seed blocks retaining both fixed zones; not ten independently sampled markets. Exhaustive paired5^5 bootstrap plus independent-arm5seed-block MonteCarlo diagnostic. Neither changes the six registered run-level decisions; five clusters remain limited evidence for population inference.')


def main():
    for p in (T/'PREREGISTRATION_T.md',T/'aggregate_t.py',T/'run_t.py',T/'orpo_t.py',T/'spr_t.py',T/'envs_t.py',T/'envs_tn.py',U/'PREREGISTRATION_U.md',U/'aggregate_u.py',U/'run_u.py'):
        read(p)
    t=analyse_t(load_t());u=analyse_u(load_u())
    check(all(sha(REPO/p)==h for p,h in HASHES.items()),'Inputs unchanged through review')
    out=dict(status='PASS_INDEPENDENT_T16_U6_REGISTERED_RECOMPUTATION',checks=CHECKS,T=t,U=u,source_environment_model_training_calls=0,
        input_sha256=HASHES,reviewer_sha256=sha(Path(__file__)),independent_rank_code_sha256=sha(HERE/'recompute_study_v.py'),
        bootstrap_registered_seed=20261031,registered_resamples=10000,registered_generator_reset_separately_for_each_study=True,
        scope='No registered aggregate/runner imported. All170T/80U retained. Five-seed U sensitivity is posthoc and cannot retroactively alter preregistered decisions. Timing/hash-prefix checks owned by parent reviewer.')
    path=HERE/'RECOMPUTATION_T_U.json'
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=out['status'],checks=CHECKS,T_supported=sum(v['supported'] is True for v in t['verdicts'].values()),
        U_supported=sum(v['supported'] is True for v in u['verdicts'].values()),report_sha256=sha(path))))


if __name__=='__main__':main()
