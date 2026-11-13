"""Independent local chronology, release-content and PDF checks. Never mutates inputs."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, re, fnmatch, os, subprocess
import fitz
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
WORK = CODE.parents[1]
LATEX = WORK / 'github/accounting-options-rl-latex-latex'
SCRATCH = WORK / 'review_3_scratch'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(t): return datetime.fromtimestamp(t, timezone.utc).isoformat()

def integrity():
    b = CODE / 'ACCOUNTING_OPTIONS_2026-10-14/study_v'
    matches = []
    for line in (b/'PREREGISTRATION_HASHES_V.txt').read_text().splitlines()[1:]:
        h, name = line.split(None, 1)
        p = (b/name).resolve()
        data = p.read_bytes()
        if name.startswith('RUN_LOG'):
            # The writer froze all bytes before the marker existed.
            data = data.split(b'## After hashing')[0]
        matches.append({'file':name,'expected':h,'actual':hashlib.sha256(data).hexdigest(),
                        'matches':hashlib.sha256(data).hexdigest()==h})
    fs = sorted((b/'results_v').glob('*.json'))
    rows = [json.loads(p.read_text()) for p in fs]
    expected = {(setting,arm,seed) for setting,arms,seeds in [
      ('S_DK1',['PLAIN','ORACLE','PMR','RUIN1','RUIN5','RUIN25','MV0.05','MV0.2','MV1'],range(150,158)),
      ('T_DK2',['PMR'],range(160,168)), ('T_DK1M',['PMR'],range(160,168)),
      ('S_DK1+Z',['PLAIN','ORACLE','TIR','PMR','RUIN1','RUIN5','RUIN25'],range(170,178))]
      for arm in arms for seed in seeds}
    actual = [(d['setting'],d['arm'],d['seed']) for d in rows]
    d = {'registered_hashes':matches,'run_count':len(rows),'roster_matches':set(actual)==expected,
         'duplicates':len(actual)-len(set(actual)), 'steps':dict(Counter(x['steps'] for x in rows)),
         'log_created_utc':stamp((b/'run_full.log').stat().st_ctime),
         'earliest_result_modified_utc':min(stamp(p.stat().st_mtime) for p in fs),
         'latest_result_modified_utc':max(stamp(p.stat().st_mtime) for p in fs),
         'stderr_bytes':(b/'run_full.err').stat().st_size,
         'log_done_count':sum(' done ' in x for x in (b/'run_full.log').read_text().splitlines()),
         'log_fail_count':sum('FAILED' in x for x in (b/'run_full.log').read_text().splitlines()),
         'note':'Local timestamps and self-authored hashes support chronology, not independent registration or proof of a descriptive read time.'}
    (HERE/'integrity.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
    print('INTEGRITY',json.dumps(d))

def pdfs():
    out = SCRATCH/'rendered'; out.mkdir(exist_ok=True)
    summaries={}
    for stem in ('main','supplementary'):
        p=SCRATCH/'latex_build/build_tmp'/f'{stem}.pdf'
        subprocess.run(['pdftoppm','-r','26','-png',str(p),str(out/stem)],check=True,capture_output=True)
        ims=sorted(out.glob(stem+'-*.png'))
        for k in range(0,len(ims),16):
            sheet=Image.new('RGB',(1000,1510),'#d9d9d9'); draw=ImageDraw.Draw(sheet)
            for j,f in enumerate(ims[k:k+16]):
                im=Image.open(f).convert('RGB'); im.thumbnail((242,346))
                x=(j%4)*250+(250-im.width)//2; y=(j//4)*377+20
                sheet.paste(im,(x,y)); draw.text(((j%4)*250+6,(j//4)*377+3),f.stem,fill='black')
            sheet.save(out/f'{stem}_contact_{k//16+1:02}.png')
        doc=fitz.open(p)
        summaries[stem]={'pages':len(doc),'toc':doc.get_toc(),
          'first_lines':{str(i+1):' | '.join(doc[i].get_text().splitlines()[:5]) for i in range(len(doc))}}
    (HERE/'pdf_pages.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print('PDFS', {k:v['pages'] for k,v in summaries.items()})

def release_scan():
    # Working-tree content check only. No Git index/history inspected and no claim
    # that ignored paths are absent from an existing index or old commits.
    rules=[x.strip() for x in (CODE/'.gitignore').read_text().splitlines() if x.strip() and not x.startswith('#')]
    def ignored(rel):
        s=rel.as_posix()
        for rule in rules:
            if rule.endswith('/'):
                root=rule.rstrip('/')
                if ('/' not in root and root in rel.parts) or s==root or s.startswith(root+'/'): return True
            elif fnmatch.fnmatch(s,rule) or ('/' not in rule and fnmatch.fnmatch(rel.name,rule)): return True
        return False
    patterns={
      'private_key':re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
      'github_token':re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
      'aws_access':re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
      'openai_token':re.compile(rb'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b')}
    result={}
    for name,base in [('code',CODE),('manuscript',LATEX)]:
        sizes=[]; hits=[]; ntext=0; included=[]; excluded=[]
        for folder,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ('.git','__pycache__','.pytest_cache','build_tmp')]
            for filename in files:
                p=Path(folder)/filename; rel=p.relative_to(base)
                if name=='code' and (ignored(rel) or rel.parts[0]=='exploratory_codex_2026-11-01'):
                    excluded.append(str(rel)); continue
                size=p.stat().st_size; sizes.append((size,str(rel))); included.append(p)
                if p.suffix.lower() in ('.py','.md','.txt','.toml','.yaml','.yml','.json','.csv','.cff','.tex','.bib','.ini','.env') and size<20_000_000:
                    ntext+=1; data=p.read_bytes()
                    for kind,pattern in patterns.items():
                        if pattern.search(data): hits.append({'file':str(rel),'kind':kind})
        result[name]={'files':len(sizes),'text_scanned':ntext,'over_50MB':[(s,f) for s,f in sizes if s>50_000_000],
                      'largest':sorted(sizes,reverse=True)[:5],'credential_pattern_hits':hits,'excluded_files':len(excluded)}
        if name=='code':
            raw=[p for p in base.rglob('*.csv.gz') if 'esett' in str(p.relative_to(base)).lower()]
            raw_hashes={sha(p) for p in raw}
            possible=[str(p.relative_to(base)) for p in included if p.suffix.lower() in ('.gz','.csv','.pkl','.parquet') and sha(p) in raw_hashes]
            result[name].update({'local_esett_raw_gz':len(raw),'nonignored_identical_raw_copies':possible,
                'nonignored_esett_named_data':[str(p.relative_to(base)) for p in included if 'esett' in str(p.relative_to(base)).lower() and p.suffix.lower() in ('.csv','.gz','.pkl','.parquet')]})
    result['limitations']='Current-content and root ignore-pattern check; no Git commands, index/history audit, general binary secret detection or proof that every derivative is redistributable. New review scratch folder excluded as non-release material.'
    (HERE/'release_scan.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RELEASE',json.dumps(result))

if __name__=='__main__':
    integrity(); pdfs(); release_scan()
