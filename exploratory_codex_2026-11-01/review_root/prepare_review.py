"""Read-only input receipts and a scratch-only manuscript rebuild for review 3."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
WORK = CODE.parents[1]
LATEX = WORK / 'github' / 'accounting-options-rl-latex-latex'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    protected = {}
    for study in ('study_t', 'study_u', 'study_v'):
        for p in (CODE / 'ACCOUNTING_OPTIONS_2026-10-14' / study).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                protected[str(p.relative_to(WORK))] = digest(p)
    for p in LATEX.rglob('*'):
        if p.is_file() and not any(x in p.parts for x in ('.git', 'build_tmp', '__pycache__')):
            protected[str(p.relative_to(WORK))] = digest(p)
    receipt = {'utc':datetime.now(timezone.utc).isoformat(), 'sha256':protected,
               'note':'Local content snapshot; not external timestamp attestation. No Git commands.'}
    target = HERE / 'INPUT_RECEIPT.json'
    if target.exists():
        raise RuntimeError('Never overwrite input receipt')
    target.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    out = subprocess.run([sys.executable, '-B', '-X', 'utf8', 'tools/verify_studies.py'],
                         cwd=CODE, capture_output=True, text=True, errors='replace')
    (HERE / 'verify_studies_stdout.txt').write_text(out.stdout + out.stderr, encoding='utf-8')
    print(json.dumps({'snapshot_files':len(protected), 'verifier_exit':out.returncode,
                      'verifier_tail':out.stdout.splitlines()[-3:]}))
    scratch = WORK / 'review_3_scratch' / 'latex_build'
    shutil.copytree(LATEX, scratch, ignore=shutil.ignore_patterns('.git','build_tmp','__pycache__'))
    print('Scratch manuscript prepared: ' + str(scratch), flush=True)
    out = subprocess.run([sys.executable, '-B', '-X', 'utf8', 'tools/build_review.py','--draft'],
                         cwd=scratch, capture_output=True, text=True, errors='replace')
    (HERE / 'scratch_build_stdout.txt').write_text(out.stdout + out.stderr, encoding='utf-8')
    print(json.dumps({'build_exit':out.returncode,'output':out.stdout[-5000:],'stderr':out.stderr[-1500:]}))

if __name__ == '__main__':
    prepare()
