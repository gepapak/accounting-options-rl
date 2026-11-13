"""Read-only original-input check; write one fresh research receipt only."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[3]
receipt=json.loads((HERE/'INPUT_RECEIPT.json').read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
mismatches=[]
for rel,digest in receipt['sha256'].items():
    p=PAPER/rel
    if not p.is_file() or sha(p)!=digest:mismatches.append(rel)
review=PAPER/'CODEX_REVIEW_3.md'
expected_review='dea51c42897305b739bb926bddd74eaf3e8f6b1e42110c4d573704088baef1b6'
result=dict(original_inputs_checked=len(receipt['sha256']),mismatches=mismatches,
    prior_review_sha256=sha(review),prior_review_unchanged=sha(review)==expected_review,
    new_PPO_runs=0,new_registered_studies=0,new_Env_steps=0,
    new_original_source_kernel_hand_top_calls=54,
    scope='Original T/U/V and manuscript inputs pinned by prior review; additional old-native source/result preservation checked in their separate audit receipts. No Git operation.')
with (HERE/'RESEARCH_PRESERVATION_CHECK_V1.json').open('x',encoding='utf-8') as out:
    json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
print(json.dumps(result))
if mismatches or not result['prior_review_unchanged']:raise SystemExit(1)
