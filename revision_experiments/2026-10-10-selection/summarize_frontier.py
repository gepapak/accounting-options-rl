"""Reproduce exact accounting frontiers and prospective joint certificates."""
from pathlib import Path
import os,sys,json,csv,hashlib
import numpy as np
ROOT=Path(os.environ.get('RCT_ROOT',r'C:\Users\gepap\Documents\GitHub\rl-trading-reality-check'))
sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
from envs_b import load_market,FEE,LEV,Q1,K0
BASE=Path(__file__).parent
OUT=BASE/'frontier_certification'

def frontier(r,f):
    # Include an explicit flat comparator. Pair intersections partition [0,1]
    # into intervals on which all rankings are constant.
    knots=[0.,1.]
    for i in range(len(r)):
        for j in range(i):
            if abs(f[i]-f[j])>1e-10:
                x=(r[i]-r[j])/(f[i]-f[j])
                if 0<x<1:knots.append(x)
    knots=sorted(set(knots));segments=[]
    for lo,hi in zip(knots[:-1],knots[1:]):
        winner=int(np.argmax(r-((lo+hi)/2)*f))
        if segments and segments[-1]['winner']==winner:segments[-1]['end']=hi
        else:segments.append(dict(start=lo,end=hi,winner=winner))
    # Verify endpoints/midpoints against exhaustive scores and monotonicities.
    for seg in segments:
        for x in [seg['start']+1e-10,(seg['start']+seg['end'])/2,seg['end']-1e-10]:
            assert (r-x*f)[seg['winner']] >= (r-x*f).max()-1e-7
    ordered=[s['winner'] for s in segments]
    assert np.all(np.diff(f[ordered])<=1e-7)
    assert np.all(np.diff((r-f)[ordered])>=-1e-7)
    return segments

def main():
    manifest=json.loads((OUT/'prospective_manifest.json').read_text());plan=manifest['certificate_plan']
    for name,digest in manifest['hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((BASE/'audit_frontier.py').read_bytes()).hexdigest()==manifest['runner_sha256']
    assert hashlib.sha256((BASE/'certify_frontier.py').read_bytes()).hexdigest()==manifest['certification_runner_sha256']
    rows=[];zone_summary={}
    for zone in ['DK1','DK2']:
        results=[json.loads(p.read_text()) for p in sorted((OUT/'jobs').glob(zone+'*/result.json'))]
        assert len(results)==10
        m=load_market(zone,'test','RM');cap=100*FEE*np.minimum(max(abs(LEV))*Q1,m.cap_mwh[200:]).sum()/K0
        log=np.log(4*20/plan['cost_error'])
        for x in results:
            assert x['draws']==manifest['draws'] and x['sign_seed']==manifest['evaluation_seeds'][zone]
            r=np.array(x['reported_returns']);b=np.array(x['booked_returns']);c=np.array(x['costs_pct']);f=np.array(x['injections_pct']);n=len(c)
            assert np.allclose(r-b,f,atol=1e-7)
            assert c.min()>=0 and c.max()<=cap+1e-8
            h=np.sqrt(2*c.var(ddof=1)*log/n)+7*cap*log/(3*(n-1))
            lo=max(0,c.mean()-h);hi=min(cap,c.mean()+h)
            reported_lower=np.minimum(r,plan['reported_truncation_pct']).mean()-300*np.sqrt(np.log(20/plan['positive_profit_error'])/(2*n))
            row=dict(label=x['label'],zone=zone,training_charge=int(x['label'][-1]),reported_mean=float(r.mean()),booked_realized_mean=float(b.mean()),
                injection_mean=float(f.mean()),cost_mean=float(c.mean()),expected_booked_estimate=float(-c.mean()),expected_booked_lower=float(-hi),expected_booked_upper=float(-lo),
                reported_mean_mcse=float(r.std(ddof=1)/np.sqrt(n)),booked_realized_mean_mcse=float(b.std(ddof=1)/np.sqrt(n)),
                expected_reported_lower=float(reported_lower),cost_lower=float(lo),cost_upper=float(hi),
                bound_cap_pct=float(cap),lookup_fallback_fraction=x['exact_network_fallback_fraction'])
            rows.append(row)
        zr=[x for x in rows if x['zone']==zone]
        winner=next(x for x in zr if x['label']==manifest['discovery_winners'][zone]);others=[x for x in zr if x!=winner]
        r=np.array([x['reported_mean'] for x in zr]+[0.]);f=np.array([x['injection_mean'] for x in zr]+[0.]);labels=[x['label'] for x in zr]+['Flat']
        segments=frontier(r,f)
        for seg in segments:seg['label']=labels[seg['winner']]
        idx=labels.index(winner['label']);critical=min([(r[idx]-r[j])/(f[idx]-f[j]) for j in range(len(r)) if f[idx]>f[j]+1e-9]+[1.])
        assert labels[np.argmax(r)]==winner['label']
        zone_summary[zone]=dict(discovery_winner=winner['label'],winner=winner,all_worst_booked_certified=bool(winner['cost_lower']>max(x['cost_upper'] for x in others)),
            positive_expected_reported_certified=bool(winner['expected_reported_lower']>0),
            next_largest_cost_upper=max(x['cost_upper'] for x in others),critical_metric_charge=float(critical),
            empirical_booked_rank=1+sum(x['booked_realized_mean']>winner['booked_realized_mean'] for x in zr),
            cost_identity_rank=1+sum(x['cost_mean']<winner['cost_mean'] for x in zr),frontier=segments)
    assert len(rows)==20
    summary=dict(draws=manifest['draws'],total_error=plan['total_error'],zones=zone_summary,
        scope='Joint probability >=.95 for all cost intervals and positive-profit lower bounds, conditional on fixed panels, policies and no-edge model. Frontier and mean ranks are descriptive plug-in quantities, not confidence sets for breakpoints.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'agents.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained');colors={0:'#d34b32',1:'#2266ad'}
    for col,zone in enumerate(['DK1','DK2']):
        zr=[x for x in rows if x['zone']==zone];s=zone_summary[zone];ax=axes[0,col]
        for x in zr:
            ax.errorbar(x['reported_mean'],x['expected_booked_estimate'],
                yerr=[[x['expected_booked_estimate']-x['expected_booked_lower']],[x['expected_booked_upper']-x['expected_booked_estimate']]],
                fmt='o',color=colors[x['training_charge']],capsize=2,markersize=5)
        w=s['winner'];ax.scatter(w['reported_mean'],w['expected_booked_estimate'],s=170,facecolors='none',edgecolors='black',linewidths=1.5)
        ax.annotate('Selected winner',xy=(w['reported_mean'],w['expected_booked_estimate']),xytext=(-85,22),textcoords='offset points',arrowprops={'arrowstyle':'->'},fontsize=9)
        ax.axhline(0,color='gray',lw=.7);ax.set_title(zone);ax.set_xlabel('Mean reported return (%)');ax.set_ylabel('Expected booked return (%)');ax.grid(alpha=.2)
        ax=axes[1,col];grid=np.linspace(0,1,501)
        for x in zr:
            ax.plot(grid,x['reported_mean']-grid*x['injection_mean'],color=colors[x['training_charge']],alpha=.45,lw=1)
        ax.plot(grid,w['reported_mean']-grid*w['injection_mean'],color='black',lw=2,label='Original selected winner')
        ax.axhline(0,color='#32823b',lw=1.5,label='Flat comparator')
        ax.axvline(s['critical_metric_charge'],color='gray',ls='--',lw=1)
        ax.text(s['critical_metric_charge']-.02,ax.get_ylim()[1]*.6,f"{100*s['critical_metric_charge']:.1f}%",rotation=90,ha='right',fontsize=9)
        ax.set_xlabel(r'Evaluation charge $\beta$');ax.set_ylabel(r'Re-scored return $R-\beta F$ (%)');ax.grid(alpha=.2)
    handles=[plt.Line2D([0],[0],color=colors[a],marker='o',label=label) for a,label in [(0,'Uncharged training reward'),(1,'Fully charged training reward')]]
    axes[0,0].legend(handles=handles,loc='lower left',fontsize=10)
    axes[1,0].legend(fontsize=10)
    fig.savefig(OUT/'fig_frontier.pdf');fig.savefig(OUT/'fig_frontier.png',dpi=150)
    for zone,s in zone_summary.items():print(zone,json.dumps({k:v for k,v in s.items() if k not in ['winner','frontier']}),json.dumps(s['winner']))
if __name__=='__main__':main()
