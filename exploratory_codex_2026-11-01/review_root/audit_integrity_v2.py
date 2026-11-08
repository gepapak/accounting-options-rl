"""Correct the review checker's marker parser; preserve its failed V1 receipt."""
from pathlib import Path
import json, hashlib

HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
b=CODE/'ACCOUNTING_OPTIONS_2026-10-14/study_v'
d=json.loads((HERE/'integrity.json').read_text())
for row in d['registered_hashes']:
    if row['file']=='RUN_LOG_V.md':
        raw=(b/row['file']).read_bytes()
        # The text also quotes the marker in its first heading. Only the
        # subsequent standalone heading delimits the registered byte prefix.
        prefix=raw[:raw.index(b'\n## After hashing')]
        row.update(actual=hashlib.sha256(prefix).hexdigest(),matches=hashlib.sha256(prefix).hexdigest()==row['expected'],prefix_bytes=len(prefix))
d['review_checker_erratum']='V1 split on the marker quoted in the opening title. V2 requires the actual newline-prefixed heading. No study artifact changed.'
(HERE/'integrity_v2.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
assert all(r['matches'] for r in d['registered_hashes'])
print(json.dumps({'all_seven_hashes_match':True,'prefix_bytes':next(r['prefix_bytes'] for r in d['registered_hashes'] if r['file']=='RUN_LOG_V.md')}))
