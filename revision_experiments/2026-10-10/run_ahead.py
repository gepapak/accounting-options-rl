"""Schedule the already planned final seed on two spare workers; no design change.
Separate outputs prevent two schedulers from writing the same checkpoint.
"""
import os,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import run_continuation as runner
def ahead(region,alpha):
    runner.OUT=Path(__file__).parent/'continuation_ahead'
    return runner.job(region,4,alpha,False)
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(ahead,r,a) for r in ['DK1','DK2'] for a in [0,1]]):
            print(future.result(),flush=True)
    origin=Path(__file__).parent/'continuation_ahead/jobs'
    target=Path(__file__).parent/'continuation_experiment/jobs'
    for folder in origin.iterdir():
        if (target/folder.name).exists():
            print('Original scheduler already started',folder.name,'keeping separate copy',flush=True)
        else:
            shutil.copytree(folder,target/folder.name)
            print('Copied completed planned job',folder.name,flush=True)
