"""Download eSett Open Data EXP14 imbalance prices for the twelve Nordic zones, 2021-11-01 .. 2025-03-04
(before the window of the repository's fetch_esett.py), for the cross-regime coverage analysis.
eSett terms: public without authorization, no explicit redistribution license -> files stay local (data_esett_pre/),
with a manifest of per-call SHA-256 hashes."""
import hashlib, json, time, urllib.parse, urllib.request
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent / "data_esett_pre"; OUT.mkdir(exist_ok=True)
API = "https://api.opendata.esett.com/EXP14/Prices"
MBAS = {"FI": "10YFI_1________U", "NO1": "10YNO_1________2", "NO2": "10YNO_2________T", "NO3": "10YNO_3________J",
        "NO4": "10YNO_4________9", "NO5": "10Y1001A1001A48H", "DK1": "10YDK-1--------W", "DK2": "10YDK-2--------M",
        "SE1": "10Y1001A1001A44P", "SE2": "10Y1001A1001A45N", "SE3": "10Y1001A1001A46L", "SE4": "10Y1001A1001A47J"}
START, END = pd.Timestamp("2021-11-01", tz="UTC"), pd.Timestamp("2025-03-04", tz="UTC")
edges = [START] + [x for x in pd.date_range(START, END, freq="MS", tz="UTC") if x > START] + [END]
man = {"source": API, "fetched_utc": pd.Timestamp.utcnow().isoformat(), "period": [str(START), str(END)], "areas": {}}
for zone, code in MBAS.items():
    frames, calls = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        q = {"start": a.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "end": b.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "mba": code}
        url = API + "?" + urllib.parse.urlencode(q)
        for attempt in range(5):
            try:
                raw = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "paper1-research-fetch"}), timeout=120).read(); break
            except Exception:  # noqa: BLE001
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(url)
        calls.append({"query": q, "sha256": hashlib.sha256(raw).hexdigest()})
        frames.append(pd.DataFrame(json.loads(raw)))
        time.sleep(0.5)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("timestampUTC").sort_values("timestampUTC")
    f = OUT / f"esett_EXP14_{zone}_pre.csv.gz"
    df.to_csv(f, index=False)
    man["areas"][zone] = {"mba": code, "rows": len(df), "first": str(df.timestampUTC.iloc[0]), "last": str(df.timestampUTC.iloc[-1]),
                          "file_sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "calls": calls}
    print(zone, len(df), df.timestampUTC.iloc[0], df.timestampUTC.iloc[-1], flush=True)
(OUT / "manifest.json").write_text(json.dumps(man, indent=1))
