"""Read-only, in-memory scan (no third-party code is executed): for the storage and bidding environments of the paper's
electricity-RL audit, fetch the sources at HEAD and extract every step() body plus the lines that compute the reward, the state of
charge, clipping and penalties. Writes one evidence file per repository into ./delivery_evidence/ (scratch)."""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = Path(__file__).resolve().parent / "delivery_evidence"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "code-audit-script"}
REPOS = ["ADGEfficiency/energy-py", "aws-samples/sagemaker-rl-energy-storage-system", "RichardFindlay/battery-optimisation-with-drl",
         "masa2203/battery_arbitrage_with_drl", "masa2203/battery-dispatch-with-DRL", "albanpuech/energy-arbitrage-via-reinforcement-learning",
         "deep-daya/Grid_Scale_Energy_Storage_Q_Learning", "zoewr/RL-bess-control", "kim-mskw/BessBidder",
         "vaid75/RL-Based-_Smart_Energy_optimization_System", "xiufengliu/feature-driven-rl-pv-trading", "liu-jinyu/electricity_market_RL_HDB"]
KEY = re.compile(r"reward|soc|state_of_charge|charge|discharg|clip|np\.min|np\.max|min\(|max\(|penal|imbalanc|deliver|feasib|capacity|energy|power", re.I)


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def sources(repo):
    sha = subprocess.run(["git", "ls-remote", f"https://github.com/{repo}", "HEAD"], capture_output=True, text=True, timeout=60).stdout.split()
    sha = sha[0] if sha else "HEAD"
    tree = json.loads(get(f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
    files = [t["path"] for t in tree.get("tree", []) if t["type"] == "blob" and t["path"].endswith((".py", ".ipynb")) and t.get("size", 0) < 400_000]
    return sha, files


def code_of(raw, path):
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(raw)
            return "\n".join("".join(c.get("source", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
        except Exception:
            return ""
    return raw.decode("utf-8", errors="replace")


def step_bodies(code):
    lines, out = code.splitlines(), []
    for i, ln in enumerate(lines):
        m = re.match(r"(\s*)def\s+(step|_step|reward|_reward|get_reward|calc\w*reward\w*|compute\w*reward\w*|_get_reward|update_soc|charge\w*|discharge\w*)\s*\(", ln)
        if m:
            ind = len(m.group(1))
            body = [ln]
            for ln2 in lines[i + 1:i + 160]:
                if ln2.strip() and (len(ln2) - len(ln2.lstrip())) <= ind and not ln2.lstrip().startswith(("#", ")", "]")):
                    break
                body.append(ln2)
            out.append((i + 1, body))
    return out


for repo in REPOS:
    try:
        sha, files = sources(repo)
    except Exception as ex:
        print(f"{repo}: tree failed: {ex}")
        continue
    ev = [f"# {repo} @ {sha}", f"# {len(files)} python/notebook files"]
    n_bodies = 0
    for path in files:
        if re.search(r"test|setup\.py|docs?/", path, re.I):
            continue
        try:
            code = code_of(get(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"), path)
        except Exception:
            continue
        bodies = step_bodies(code)
        if not bodies:
            continue
        ev.append(f"\n## {path}")
        for lineno, body in bodies:
            n_bodies += 1
            ev.append(f"### line {lineno}")
            ev.extend(body)
        time.sleep(0.2)
    (OUT / (repo.replace("/", "__") + ".txt")).write_text("\n".join(ev), encoding="utf-8")
    print(f"{repo}: {len(files)} files, {n_bodies} step/reward bodies")
