"""Only two local repository-map corrections; preserve all other input bytes."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent
P=H.parents[1]/'README.md'
raw=P.read_bytes(); before=raw
changes=[
 (b'requirements*.txt                standalone analyses; engine stack; Studies A-S; Study G (gym-mtsim)',
  b'requirements*.txt                standalone analyses; engine stack; Studies A-V; Study G (gym-mtsim)'),
 (b'ACCOUNTING_OPTIONS_2026-10-14/   option-ladder theory checks, historical settlement-component analysis, Studies N-S',
  b'ACCOUNTING_OPTIONS_2026-10-14/   option-ladder theory checks, historical settlement-component analysis, Studies N-V')]
for old,new in changes:
    assert raw.count(old)==1
    raw=raw.replace(old,new)
P.write_bytes(raw)
(H/'LOCAL_FIXES.json').write_text(json.dumps({'file':str(P),'before_sha256':hashlib.sha256(before).hexdigest(),
 'after_sha256':hashlib.sha256(raw).hexdigest(),'changes':[{'before':a.decode(),'after':b.decode()} for a,b in changes]},indent=2),encoding='utf-8')
print('Made two README repository-map corrections, A-S to A-V and N-S to N-V. No claim/result/reward changes.')
