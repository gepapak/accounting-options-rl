# Study X pre-registration: do the twin rewards survive a violated single-floor assumption?

Written 2 November 2026, before any full run of this study. Internal pre-registration: the files listed in `PREREGISTRATION_HASHES_X.txt`
are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every hypothesis is reported, whatever the result.

## Question and what precedes it
SPR and TIR are derived under Assumption C: the reported reward is a single monotone censor of the booked profit, `r = max(x, tau)`. The public
engine does not satisfy it: it writes the opening fee with its own floor and settles afterwards, so from an account that cannot pay the fee a
twin pair reports cash increments `(1, 0)` where the single-floor formula reads a booked profit of `+1` and the booked value is `-9`. The paper
adds a cost-shift test that detects this (`exploratory_claude_2026-11-02/contract_test.py`: no flag in 5,122 states of the single-floor sleeve;
761 of 761 informative states with a fee above the account flagged in a two-write variant; the engine's own source methods flagged from starting
cash 0 and 5 DKK). The learning question is open: when the contract fails, do SPR and TIR still remove the exploit and keep the edge, or does
the misidentification at insolvent accounts reintroduce it?

**Theory (exploratory; `exploratory_claude_2026-11-02/`).** For an account with equity `e < c` that writes the fee first with its own floor, the
twin pair equals that of a single-floor world with threshold `-e` and magnitude `g + c - e` (the proposition on testing Assumption C). TIR then
returns `g - e` on the favourable branch and `-g - 2c + e` on the unfavourable one, so its expectation exceeds the booked one by `(2p-1)(c-e)`
(a small overpayment, no asymmetry between branches, and zero in a fair market); SPR's expectation falls short by `(p - 1/2) g`, as
in the single-floor case. The reported reward, by contrast, pays the whole forgiven loss and fee. The prediction is graceful degradation: TIR's
error grows (mean absolute error on credited steps 0.062 against 0.008 in the single-floor sleeve; SPR 0.39; reported reward 0.57, percent of
the allocation per quarter-hour) but no incentive to gamble is restored.

**Development (pilot seed 90, four runs, exploratory; not part of this study).** One run per arm (1,000,000 steps, `pilot_results/`, 22:28-22:50). PLAIN gambled: `X` 0.875, booked edge -0.9% per episode (-21.7% from zero equity), 71% of the allocation forgiven per placebo episode. ORACLE booked +22.1%; SPR +22.0% and TIR +22.0%, both with no probe at maximal leverage. The TIR audit was exact (error at most 0.003 point, both markets) for the ORACLE, SPR and TIR agents, which never reached an account that could not pay the fee, and off by 1.03 (edge market) and 0.12 (placebo market) points for the gambler, whose SPR audit was off by 3.3 points and whose reported-return audit was off by 71 points in the placebo market. The pilot showed that the exploit is acquired in the two-write sleeve, so the full study is run; it did not change an arm, a measure or a decision rule except that XB1's bound was tightened from 2.0 to 1.0 point (Study V's bound under a single floor) after the honest agents' audit errors turned out to be negligible.

## Design (`run_x.py`, `envs_x.py`)
- **Environment.** The Study T edge-market sleeve (`S_DK1`: DK1, p_edge 0.60, kappa 4, the Study T market draw) with the two-write floor
  `EdgeEnvMF` (fee written first with its own floor, gross P&L second; the ledger books the whole fee). Half of the training episodes start
  at zero reported equity (`EdgeEnvMFZ`, as Study V's Part C), so that accounts that cannot pay the fee are visited. Probes, test windows and
  start equities of the evaluation are those of Studies T and V, run in the same two-write environment.
- **Learner.** PPO as in Study V: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Arms.** PLAIN (reported reward), ORACLE (booked reward; reads the ledger), SPR and TIR (built from the reported rewards of the real step
  and of a sign-reversed clone and the own cost, never the ledger; `../study_t/spr_t.py`).
- **Seeds.** 190-197 per arm: 4 arms x 8 seeds = 32 runs. A missing or failed run is reported and not replaced.
- **Measures per run.** `X` = share of the 16 reachable low-equity probes (equity in {0, 0.02, 0.05, 0.1} allocations, both signals, share of
  the episode remaining 0.05 and 0.5) at maximal leverage (greedy policy); `E` = booked P&L in the edge market on the test split, percent of K0
  per one-week episode, mean of 100 episodes. Ledger-free audit of every run as in Study V (sum of the reported, SPR and TIR step rewards over
  the 100 test episodes, twin queries at evaluation time only).

## Hypotheses and decision rules (`aggregate_x.py`)
"Removes the exploit": `X(arm) < X(PLAIN)` by a one-sided Mann-Whitney test, `p < 0.05`, and `mean X(arm) <= mean X(ORACLE) + 0.10`. "Keeps the
edge": mean `E` of the arm over mean `E` of ORACLE at least 0.80, with a one-sided 95% bootstrap lower bound (10,000 resamples of runs within each
arm, generator seed 20261102) of at least 0.60.
- **XA1.** The exploit is present in the two-write sleeve: `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`.
- **XA2.** TIR removes it. **XA3.** TIR keeps the edge.
- **XA4.** SPR removes it. **XA5.** SPR keeps the edge.
- **XB1.** For the ORACLE, SPR and TIR agents (24 of the 32), the largest error of the TIR estimate of the 100-episode booked return is at most
  1.0 point (percent of K0 per episode) in the edge market and in the placebo market (Study V's bound under a single floor).
If XA1 fails, XA2-XA5 are reported as not testable. Six hypotheses, each with its own decision; no multiplicity correction.

*Prior belief (written before the runs):* XA1 holds (the second floor forgives the fee as well). XA2-XA5 probably hold (graceful degradation), TIR's
ratio close to 1 because its bias at insolvent accounts is an overpayment of about 20% of the fee; SPR's edge ratio may be lower than TIR's. XB1 probably
holds: the audit accumulates the misidentification over every insolvent step of an episode, which is rare for honest agents (the pilot's honest
agents were exact) and common for gamblers, which XB1 excludes.

## Exploratory outputs (not registered)
Booked edge from zero equity; placebo-market reported and forgiven returns; the TIR-SPR difference in booked edge with a bootstrap interval; the audit
errors of the gambling agents (`X >= 0.25`) against the single-floor values of Study V.

## Software validation, integrity and failures
`run_x.py --smoke` (20,000 steps; PLAIN and TIR, one seed) ran end to end and its output is excluded; the pilot is exploratory. The greedy NumPy
forward pass is checked against `model.predict` for every run and the audit asserts that it replays the test episodes. Files hashed before launch:
this file, `RUN_LOG_X.md`, `envs_x.py`, `run_x.py`, `aggregate_x.py`, and (reused unchanged from Study T) `../study_t/envs_t.py` and
`../study_t/spr_t.py`. The run log is append-only after "## After hashing".
