"""Independent certification cohort, frozen after exploratory frontier evaluation.

Prospective primary certificates: positive expected reported profit for the two
original discovery winners, and lowest expected booked profit in each zone.
All 20 cost intervals and all 20 positive-profit lower bounds are simultaneous
with total error <= .05. Reuses frozen evaluator without modifying its source.
"""
from pathlib import Path
import audit_frontier as evaluator
evaluator.OUT=Path(__file__).parent/'frontier_certification'
evaluator.N=2000
evaluator.SEEDS={'DK1':2026101011,'DK2':2026101012}
if __name__=='__main__':
    import sys,json,hashlib
    if '--freeze' in sys.argv:
        evaluator.freeze()
        path=evaluator.OUT/'prospective_manifest.json'
        manifest=json.loads(path.read_text())
        manifest['certification_runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        manifest['certificate_plan']={
            'total_error':.05,'cost_error':.025,'positive_profit_error':.025,'policies':20,
            'reported_truncation_pct':200.,
            'positive_bound':'For each policy Y=min(reported_return,200) in [-100,200]. E[R]>=E[Y]>=mean(Y)-300*sqrt(log(20/.025)/(2*n)). Union bound over 20.',
            'cost_bound':'Costs in [0,Cmax], Cmax=100*fee*sum(executed maximum cap)/K. h=sqrt(2*sample_variance*log(4*20/.025)/n)+7*Cmax*log(4*20/.025)/(3*(n-1)). Union bound over both tails and 20 policies, clipping intervals to [0,Cmax].',
            'ranking':'A discovery winner is certified worst in expected booked profit if its cost lower bound exceeds every other cost upper bound in its zone. No-edge identity: expected booked return=-expected cost.',
            'selection':'Discovery winners remain DK1_s0_a0 and DK2_s2_a0 from original L evaluation. No reselection from either new cohort. Bounds and truncation chosen after exploratory M1 and before independent M2.',
            'limits':'Conditional on fixed historical panels, model no-edge signs, frozen exported policies and fee model. Not real-market skill, not a guarantee on realized booked rank.'}
        path.write_text(json.dumps(manifest,indent=2))
    else:
        manifest=json.loads((evaluator.OUT/'prospective_manifest.json').read_text())
        assert manifest['certification_runner_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        evaluator.main()
