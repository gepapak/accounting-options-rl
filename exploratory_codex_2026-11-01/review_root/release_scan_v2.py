"""Read working-tree release candidates with all three root/nested ignore files."""
from pathlib import Path,PurePath
import fnmatch,hashlib,json,os,re
H=Path(__file__).resolve().parent; C=H.parents[1]; W=C.parents[1]
rules=[]
for p in [C/'.gitignore',C/'exploratory_novelty_2026-10-16/sample_only_learned_audit/.gitignore',C/'exploratory_novelty_2026-10-16/sample_only_learned_audit/native_joint_fault_v2/.gitignore']:
    for x in p.read_text().splitlines():
        x=x.strip()
        if x and not x.startswith('#'):rules.append((p.parent,x))
def ignored(p):
    for root,rule in rules:
        if not p.is_relative_to(root):continue
        rel=p.relative_to(root); s=rel.as_posix()
        if rule.endswith('/'):
            q=rule.rstrip('/')
            if ('/' not in q and q in rel.parts) or s==q or s.startswith(q+'/'):return True
        elif fnmatch.fnmatch(s,rule) or rel.match(rule) or ('/' not in rule and fnmatch.fnmatch(rel.name,rule)):return True
    return False
patterns={
 'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
 'aws_access':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
 'openai_token':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b')}
o={}
for name,base in [('code',C),('manuscript',W/'github/accounting-options-rl-latex-latex')]:
    sizes=[]; hits=[]; ntext=0; excluded=0; review=0; data=[]
    for folder,dirs,files in os.walk(base):
        dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache','build_tmp')]
        for file in files:
            p=Path(folder)/file; rel=p.relative_to(base)
            if name=='code' and rel.parts[0]=='exploratory_codex_2026-11-01':review+=1;continue
            if name=='code' and ignored(p):excluded+=1;continue
            size=p.stat().st_size; sizes.append((size,str(rel)))
            if p.suffix.lower() in ('.py','.md','.txt','.toml','.yaml','.yml','.json','.csv','.cff','.tex','.bib','.ini','.env') and size<20_000_000:
                ntext+=1; raw=p.read_bytes()
                for kind,pattern in patterns.items():
                    if pattern.search(raw):hits.append({'file':str(rel),'kind':kind})
            if p.suffix.lower() in ('.gz','.csv','.pkl','.parquet'):data.append(p)
    o[name]={'files':len(sizes),'text_scanned':ntext,'ignored_files':excluded,'new_review_files_excluded':review,
      'over_50MB':[(s,f) for s,f in sizes if s>50_000_000],'largest':sorted(sizes,reverse=True)[:3],'credential_pattern_hits':hits}
    if name=='code':
        raws=[p for p in C.rglob('*.csv.gz') if 'esett' in str(p.relative_to(C)).lower()]
        hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in raws}
        o[name].update(local_esett_raw_files=len(raws),identical_nonignored_copies=[str(p.relative_to(C)) for p in data if hashlib.sha256(p.read_bytes()).hexdigest() in hashes],esett_named_nonignored_data=[str(p.relative_to(C)) for p in data if 'esett' in str(p.relative_to(C)).lower()])
o['limitations']='All three local ignore files honored. No Git commands/index/history inspection; ignored-but-already-tracked files cannot be excluded by this audit. Current-content credential patterns and exact raw-file duplication only; arbitrary binary secrets/derivative licensing not proved absent. New review folder is nonrelease material and was deliberately excluded, not asserted ignored by Git.'
o['v1_erratum']='V1 omitted nested .gitignore files, so falsely treated the existing private 53.56MB review ZIP as a release candidate. V2 includes the explicit review_bundle_private exclusion; the private ZIP was preserved unchanged.'
(H/'release_scan_v2.json').write_text(json.dumps(o,indent=2),encoding='utf-8'); print(json.dumps(o,indent=2))
