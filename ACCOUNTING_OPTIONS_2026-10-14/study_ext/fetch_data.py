"""Fetch the official example file of Gym-Trading-Env at the pinned commit and check its hash. The repository of this paper does not redistribute it (no licence is stated).

    python fetch_data.py
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

COMMIT = "a86fc03d29759374fae2db7df4d878485808a730"
URL = f"https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/{COMMIT}/examples/data/BTC_USD-Hourly.csv"
SHA256 = "b5ccfdef2aa852ded1ea03378c1777757d2c27a81d53e0fc81af0011fed9cccf"
OUT = Path(__file__).resolve().parent / "data" / "BTC_USD-Hourly.csv"


def main() -> int:
    if OUT.exists() and hashlib.sha256(OUT.read_bytes()).hexdigest() == SHA256:
        print("already present, hash matches:", OUT)
        return 0
    with urllib.request.urlopen(URL) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"the downloaded file differs from the pinned bytes: {digest}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(raw)
    print("wrote", OUT, len(raw), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
