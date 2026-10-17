from pathlib import Path,PureWindowsPath
import json,sys,hashlib,csv,os
import numpy as np
BASE=Path(__file__).parent/'continuation_experiment'
ROOT=Path(os.environ.get('RCT_ROOT',r'C:\Users\gepap\Documents\GitHub\rl-trading-reality-check'))
if not(ROOT/'GENERALIZATION_STUDY_2026-09-29').exists(): ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
from envs_b import load_market,K0
rows=[json.loads(p.read_text()) for p in sorted((BASE/'jobs').glob('*/result.json'))]
assert len(rows)==20, f'Need all20 results, got{len(rows)}'
manifest=json.loads((BASE/'prospective_manifest.json').read_text())
for file,digest in manifest['files'].items():
    name=PureWindowsPath(file).name
    candidate=(ROOT/'GENERALIZATION_STUDY_2026-09-29/data_panel'/name
               if name.endswith('.pkl') else Path(__file__).parent/name)
    if not candidate.exists(): candidate=Path(file)
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==digest,file
lookup={(r['region'],r['seed'],r['alpha']):r for r in rows}
assert len(lookup)==20
assert all(r['split']=='test' and r['steps']==1007616 for r in rows)
diff=np.array([[lookup[(region,seed,0)]['leverage'][3]-lookup[(region,seed,1)]['leverage'][3]
                for region in ['DK1','DK2']] for seed in range(5)])
rng=np.random.default_rng(20261010)
samples=diff[rng.integers(0,5,(20000,5))].mean(axis=(1,2))
summary=dict(primary_gap=float(diff.mean()),primary_seed_cluster_ci=np.percentile(samples,[2.5,97.5]).tolist(),
             primary_pair_differences=diff.tolist(),positive_pairs=int((diff>0).sum()),zones={})
for region in ['DK1','DK2']:
    market=load_market(region,'test','RM')
    bound=float(np.sqrt(np.sum((np.minimum(4,market.cap_mwh[200:])*np.abs(market.spread[200:]))**2)))
    cutoff=100*bound/K0*np.sqrt(2*np.log(20/.05)/100)
    summary['zones'][region]={}
    for alpha in [0,1]:
        r=[lookup[(region,s,alpha)] for s in range(5)]
        means={key:float(np.mean([np.mean(x[key]) for x in r]))
               for key in ['reported_returns','booked_returns','injections_pct','costs_pct','mean_leverage']}
        means['leverage_grid_mean']=np.mean([x['leverage'] for x in r],axis=0).tolist()
        means['max_abs_reconciliation_error_pct']=float(max(np.max(np.abs(np.array(x['reported_returns'])-np.array(x['booked_returns'])-np.array(x['injections_pct']))) for x in r))
        means['simultaneous_gross_cutoff_pct']=cutoff
        means['maximum_agent_reconciled_gross_mean_pct']=float(max(np.mean(np.array(x['booked_returns'])+np.array(x['costs_pct'])) for x in r))
        summary['zones'][region][str(alpha)]=means
with (BASE/'agents.csv').open('w',newline='') as f:
    fields=['region','seed','alpha','low_leverage','high_leverage','reported_return_pct','booked_return_pct','injections_pct','mean_leverage']
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for r in rows:
        writer.writerow(dict(region=r['region'],seed=r['seed'],alpha=r['alpha'],low_leverage=r['leverage'][3],high_leverage=r['leverage'][7],
                             reported_return_pct=np.mean(r['reported_returns']),booked_return_pct=np.mean(r['booked_returns']),
                             injections_pct=np.mean(r['injections_pct']),mean_leverage=np.mean(r['mean_leverage'])))
(BASE/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(8,3.3),sharey=True,layout='constrained')
for ax,region in zip(axes,['DK1','DK2']):
    for alpha,color,label in [(0,'#c64b38','Uncharged injections'),(1,'#2466a6','Fully charged injections')]:
        r=[lookup[(region,s,alpha)] for s in range(5)]
        for item in r:ax.plot(item['grid'],item['leverage'],color=color,alpha=.25,lw=.8)
        ax.plot(r[0]['grid'],np.mean([x['leverage'] for x in r],axis=0),color=color,lw=2,marker='o',ms=3,label=label)
    ax.set(title=region,xlabel='Reported equity / initial allocation',ylim=(-.4,16.8),xlim=(0,3))
    ax.grid(alpha=.2);ax.set_yticks([0,4,8,12,16])
axes[0].set_ylabel('Absolute chosen leverage');axes[1].legend(fontsize=8)
fig.savefig(BASE/'fig_continuation.pdf');fig.savefig(BASE/'fig_continuation.png',dpi=180)
print(json.dumps(summary,indent=2))
