# Study FT run log (append-only after "## After hashing")

- 6 November 2026, afternoon: environments, rewards, exact checks and `dp_lambda.py` written in an earlier session; no registration.
- 6 November 2026, evening, to 7 November: `run_ft.py`, `aggregate_ft.py` written; pilot at 10^6 steps, seed 98, parts B and A, arms REP, LEDGER, TIR, under heavy load on the machine (`pilot.log`). The laptop's lid closed during it, which only delayed it. The pilot contradicted a first, unhashed prediction, that SPR and TIR would lose the edge from a deficit start with half of the training funded. Part C was added and the seeds reduced from eight to six per arm, before hashing; see PREREGISTRATION_FT.md.

## After hashing
- 2026-11-07 00:47 local: hashes written; launched run_ft.py --workers 4, PID 8856.
- 2026-11-07 07:15 local: all 81 runs complete, 0 failed; aggregate_ft.py run once: 20 of 23 decisions supported; FB4 (p = 0.0505 with TSR's three seeds), FB6 and FB7 not. make_tab_ft.py added after the analysis; it computes nothing new.
