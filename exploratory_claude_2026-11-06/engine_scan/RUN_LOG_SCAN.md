# Run log: reward diagram of the running engine (append-only after the hash)

- 2026-11-06, evening: pilots of `line_launcher.py` analysed; `reward_diagram.py` written; one tolerance error found and fixed on the pilots (see PROTOCOL_SCAN.md); `scan_launcher.py`, `run_scan.py`, `analyze_scan.py`, `SCAN_JOBS.txt`, `PROTOCOL_SCAN.md` written.

## After hashing

- 2026-11-06 18:59 local: launched run_scan.py --workers 4 --out scan_campaign --manifest SCAN_JOBS.txt, PID 24484.
- 2026-11-06 19:12-20:06 local: laptop lid closed, Modern Standby; 20:11 restarted by the user from the Start menu. The first launch had reached step 29,000 of 39,311 on two jobs and had finished none; scans are written at exit, so none exists and no outcome was read.
- 2026-11-06 20:15 local: relaunched unchanged (same hashed files), run_scan.py --workers 4 --out scan_campaign --manifest SCAN_JOBS.txt.
