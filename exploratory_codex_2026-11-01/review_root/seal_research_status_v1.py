"""Seal the completed continuation's top-level report and supporting records."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
report=ROOT/'EAAI_RESEARCH_STATUS_2026-11-01.md'
broken=[]
for target in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
    if '://' not in target and not (ROOT/target).is_file():broken.append(target)
paths=[report,ROOT/'.gitignore',HERE/'RESEARCH_PRESERVATION_CHECK_V1.json']
for folder in ('action_probe_identification_001','conditional_identification_method_001',
               'native_saved_outcome_audit_001','native_reward_identification_001',
               'identification_extension_research_001','passive_cohort_audit_001','passive_unpaid_claim_audit_001'):
    paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths=sorted(set(paths))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(status='SEALED_COMPLETED_RESEARCH_CONTINUATION',
    report_sha256=sha(report),local_links_checked=True,broken_local_links=broken,
    sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},
    new_training_runs=0,new_registered_studies=0,new_Env_steps=0,new_model_loads=0,new_raw_market_reads=0,
    original_source_hand_top_calls=54,original_source_hand_helper_calls=462,
    substantive_conclusion='Specialized source-correct/missing-claim audit gain; strong new AI learning-method novelty not established. Full-policy event observer remains held.')
with (HERE/'RESEARCH_COMPLETION_MANIFEST_V1.json').open('x',encoding='utf-8') as out:
    json.dump(result,out,indent=2,sort_keys=True);out.write('\n')
print(json.dumps(dict(status=result['status'],sealed_files=len(paths),broken_links=broken,report_sha256=result['report_sha256'])))
if broken:raise SystemExit(1)
