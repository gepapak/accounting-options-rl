"""Portable path adapter for the unchanged, hashed revision runner.
RCT_ROOT optionally names the companion repository; no scientific settings change.
"""
import os,sys,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_continuation as frozen
ROOT=Path(os.environ.get('RCT_ROOT',r'C:\Users\gepap\Documents\GitHub\rl-trading-reality-check'))
if not(ROOT/'GENERALIZATION_STUDY_2026-09-29').exists(): ROOT=Path(__file__).resolve().parents[2]
def dispatch(region,seed,alpha,smoke):
    sys.path.insert(0,str(ROOT/'LIABILITY_STUDY_2026-09-29'))
    return frozen.job(region,seed,alpha,smoke)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    assert (ROOT/'GENERALIZATION_STUDY_2026-09-29').exists(),'Set RCT_ROOT to the companion repository'
    jobs=[(r,s,a,args.smoke) for s in range(1 if args.smoke else 5) for r in (['DK1'] if args.smoke else ['DK1','DK2']) for a in [0,1]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in as_completed([pool.submit(dispatch,*j) for j in jobs]):print(result.result(),flush=True)
