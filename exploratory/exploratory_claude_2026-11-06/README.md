# Exploratory work of 6 November 2026: the reward diagram

Not registered unless stated. Nothing here changes a registered study. The registered studies of this round are in `ACCOUNTING_OPTIONS_2026-10-14/study_ext/` (an unmodified public simulator) and `study_ft/` (accountings that a sign query cannot read).

| Folder | What it is |
|---|---|
| `reward_diagram/` | `sleeve_readout.py`: what the reading recovers in every study sleeve, where the truth is known (no learning). `make_fig_diagram.py`: the paper's figure of the diagram. The theory's exact checks are in `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/test_reward_diagram.py`, the reader in `twin_reward/reward_diagram.py` |
| `engine_scan/` | The reading on real clones of the audit engine: protocol and rules, hashed before the full evaluation (`PROTOCOL_SCAN.md`, `PROTOCOL_SCAN_HASHES.txt`), launcher, runner, analysis, run log, run table and summary. The 26 scan files and traces stay outside the repository for their size; five are shipped |
| `sign_cost/` | A certificate without a clone (`sign_cost_audit.py`), developed in an AI-assisted session and re-run independently with identical results, on the engine twin check's traces |
