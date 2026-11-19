"""Read-only, in memory: how do mainstream open-source backtesting engines treat an account whose loss exceeds its equity?
Fetches named source files at HEAD of the default branch and prints the lines around ruin / liquidation / negative-balance logic.
No third-party code is executed."""
import re
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = {"User-Agent": "code-audit-script"}
TARGETS = [
    ("kernc/backtesting.py", "master", "backtesting/backtesting.py", r"OutOfMoney|equity\s*<=\s*0|_cash\s*=\s*0|margin_available"),
    ("mementum/backtrader", "master", "backtrader/brokers/bbroker.py", r"cash\s*<\s*0|Margin|margin|getvalue|negative"),
    ("freqtrade/freqtrade", "develop", "freqtrade/optimize/backtesting.py", r"liquidat"),
    ("freqtrade/freqtrade", "develop", "freqtrade/persistence/trade_model.py", r"liquidation_price|def calc_profit\b|stake_amount \* "),
    ("polakowo/vectorbt", "master", "vectorbt/portfolio/nb.py", r"cash\s*<\s*0|insufficient|allow_partial|lock_cash"),
    ("jesse-ai/jesse", "master", "jesse/models/Position.py", r"liquidat|bankrupt"),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


for repo, branch, path, pat in TARGETS:
    try:
        src = get(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}").splitlines()
    except Exception as ex:
        print(f"\n===== {repo}/{path}: fetch failed ({ex})")
        continue
    rx = re.compile(pat, re.I)
    hits = [i for i, ln in enumerate(src) if rx.search(ln)]
    print(f"\n===== {repo} {path} ({len(src)} lines, {len(hits)} hits)")
    shown = set()
    for i in hits[:14]:
        lo, hi = max(0, i - 3), min(len(src), i + 6)
        if i in shown:
            continue
        print(f"--- line {i + 1}")
        for j in range(lo, hi):
            shown.add(j)
            print(f"{j + 1:5d}: {src[j][:150]}")
