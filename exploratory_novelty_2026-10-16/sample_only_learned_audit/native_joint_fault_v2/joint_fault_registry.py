"""Outcome-free twenty-policy registry for the new joint-fault experiment."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SEEDS=(7,42,123,2025,3007,5001,8102,9005,10001,11202)


def cases():
    original=json.loads((HERE.parent/'execution/run_001/cases.json').read_text(encoding='utf-8'))
    assert len(original)==40
    by_id={int(row['case_id']):row for row in original}
    rows=[]
    for policy in range(20):
        previous=by_id[2*policy+1]
        assert previous['profile']=='vulnerable' and previous['policy_index']==policy
        assert previous['checkpoint_seed']==SEEDS[policy%10]
        assert previous['policy_family']==('plain_mappo' if policy<10 else 'feasible_mappo')
        rows.append(dict(case_id=policy,policy_index=policy,policy_family=previous['policy_family'],
            checkpoint_seed=previous['checkpoint_seed'],checkpoint_dir=previous['checkpoint_dir'],
            source_run001_case_id=2*policy+1,profile='joint_fault',profile_index=1,warm_profile='vulnerable',
            cash_fraction=1e-4,horizon=96,burn_in=384,
            warm_liquidity_participation_cap_fraction=.25,audit_liquidity_participation_cap_fraction=0.,
            enable_trading_sleeve_margin=False,mtm_loss_exit_disabled=True))
    return rows
