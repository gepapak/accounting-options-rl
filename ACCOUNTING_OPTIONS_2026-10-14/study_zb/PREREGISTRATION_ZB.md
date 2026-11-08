# Study Z, Part B pre-registration: when the trainer cannot see where losses are floored (isolated margin)

Written 3 November 2026, before any full run of this part. Internal pre-registration: the files listed in `PREREGISTRATION_HASHES_ZB.txt` are
hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every hypothesis is reported, whatever the result.

## Question and what precedes it
In Study Z (registered at 16:03 UTC the same day, running while this part was written) two accounts with their own floors are both visible to
the agent and the trainer. An AI-assisted review of Study Z's design (GitHub Copilot, 3 November) objected that a per-account ruin penalty is
then always available, so the portfolio penalty's blindness is built into the design rather than forced by it. Part B removes the per-account
option: one account and one balance, and an action that chooses a margin mode, as on derivatives exchanges where a position in isolated margin
can lose at most its margin and a pooled fund covers the rest. The trainer sees the balance, the action index and the reward. Does the ruin
penalty, which removed the one-account exploit in Studies V and W, still work, and do the twin rewards?

**Theory and witness (`THEORY_ZB.md`, `test_envs_zb.py`, `witness_zb.py`; all before any Part B agent was trained).** Isolated margin is a
single censor with threshold -min(m, P), so TIR is exact at every account that can pay its own cost; a position cannot take an account with
P > m to zero, so the ruin penalty is not incurred there; and with a margin of the own cost plus 100 EUR/MWh (covering 84% of quarter-hour
moves) maximal leverage with the signal reports 22.6 EUR per quarter-hour more than leverage 4 and books 14.3 EUR less, at every funded
equity. 322,396 exact checks passed. On the registered test weeks, isolated maximal leverage while equity is at least 0.25 allocations books
-9.5% of the allocation per week against +22.0% for the honest optimum, reports +38.7% against +22.0%, with no account ruin and 21
liquidations per week, and keeps its reward under the ruin penalty at 5 and 25 (+38.7), while a liquidation penalty at 5 gives it -66.3.

**Development (pilot seed 99, exploratory; not part of this part).** One run each of REP and RUINP25 (10^6 steps, `pilot_results/`,
17:56-18:22, about 24 minutes per run while sharing the processors with Study Z). REP gambled at funded equity: maximal leverage in isolated
margin at 63% of the funded probes; booked -15.4% of the allocation per edge-market week, reported +35.3%, with 2.6 account ruins and 36.5
liquidations per week; placebo booked -92.6%. RUINP25 did not gamble (`X` 0.06) and booked +20.5%, but chose isolated margin at the honest
leverage at every probe, collecting forgiven tail losses that leave its booked P&L unchanged: reported +38.5% against booked +20.5% (30.5
liquidations, no ruin per week). For both runs the TIR audit was within 0.05 points (edge -15.48 against -15.43; placebo -92.597 against
-92.59) and the bracket held on every step. The pilot changed nothing in the design. Separately, a second AI-assisted review (GitHub Copilot,
of the proposed next steps) recommended a ledger-free exploit finder run where forgiveness is hidden from the trainer; the FINDER arm and
hypotheses ZIC1-ZIC2 were added for that reason (smoke-tested at 18:23), raising the run count to 64.

## Design (`run_zb.py`, `envs_zb.py`, `rewards_zb.py`)
- **Environment.** The Study T edge-market sleeve (`S_DK1`: DK1, `p_edge` 0.60, kappa 4, the Study T market draw) with one account and a
  margin mode (`IsolatedEnv`): cross margin writes the position with the zero floor; isolated margin limits the loss to the own cost plus
  `M_ISO` = 100 EUR per MWh (never more than the balance). The ledger books every loss. The agent observes the balance, the signal and the
  share of the episode left; the action is `MultiDiscrete([9, 2])` (the sleeve's leverage grid, the margin mode). Training episodes start at an
  equity drawn log-uniformly between 0.05 and 2 allocations (no zero starts: the exploit tested here lies at funded equity).
- **Learner.** PPO as in Studies V, X and Z: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Arms.** REP (reported reward), LEDGER (booked reward; reads the ledger), RUINP5 and RUINP25 (Study V's ruin penalty at 5 and 25 percent
  of the allocation on the balance), LIQ5 (5 percent of the allocation per liquidation of an isolated position; needs the margin rule), SPR and
  TIR (reward channel, sign-reversed clone, own cost; never the ledger, the meaning of the margin mode or the margin rule), and FINDER, a
  ledger-free exploit finder whose reward is the twin bracket's width `r + r_twin + 2C`, the credit paid on the pair for any forgiving
  accounting (it reads what SPR reads). The finder searches for forgiveness; what it finds is a witness, a lower bound on the credit available,
  not a certificate that none exists elsewhere.
- **Seeds.** 220-227 per arm: 8 arms x 8 seeds = 64 runs. A missing or failed run is reported and not replaced.
- **Measures per run (greedy policy).** Probes at equity in {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2} allocations, both signals, share of the
  episode left 0.05 and 0.5. The exploit share `X` = share of the 16 funded probes (equity 0.25 to 2 allocations) at maximal leverage (16), in
  either mode; `X_iso` = the same with isolated margin; `X_low` = the 16 low-equity probes (exploratory). `E` = booked P&L in the edge market on
  the test split, percent of K0 per one-week episode, mean of 100 episodes (start equities log-uniform between 0.05 and 2 allocations). Per test
  week: account ruins, isolated liquidations, forgiven credit, share of isolated steps; placebo reported, booked and forgiven returns.
  Ledger-free audit as in Studies X and Z (reported, SPR, TIR and the bracket's lower end over the 100 test weeks, both markets), with the
  bracket checked on every step and episode; the bracket width per week is the audit's reported return minus its lower end.

## Hypotheses and decision rules (`aggregate_zb.py`)
"Present", "removes the exploit" and "keeps the edge" are Study Z's rules: present = `X(arm) > X(LEDGER)` (one-sided Mann-Whitney, `p < 0.05`)
and `mean X(arm) >= mean X(LEDGER) + 0.10`; removes = `X(arm) < X(REP)` (one-sided Mann-Whitney, `p < 0.05`) and `mean X(arm) <= mean
X(LEDGER) + 0.10`; keeps the edge = mean `E` of the arm over LEDGER's at least 0.80, one-sided 95% bootstrap lower bound (10,000 resamples of
runs within each arm, generator seed 20261105) at least 0.60.
- **ZIA1.** The exploit is present: REP present.
- **ZIA2, ZIA3.** The ruin penalty does not remove it at strength 5 or 25: RUINP5 present, RUINP25 present (two decisions).
- **ZIA4.** The liquidation penalty, which needs the margin rule, removes it: LIQ5 removes.
- **ZIA5.** SPR removes it. **ZIA6.** SPR keeps the edge.
- **ZIA7.** TIR removes it. **ZIA8.** TIR keeps the edge.
- **ZIB1.** For all 56 runs, both markets, the twin bracket contains the booked P&L of every audited step and the booked return of every audited
  episode.
- **ZIB2.** For all 56 runs, both markets, the TIR estimate of the 100-week mean booked return is within 0.5 points of the allocation per week
  (TIR is exact where the account can pay its own cost, part 1).
- **ZIB3.** Mechanism of ZIA2 and ZIA3: the RUINP runs with `X >= 0.25` incur at most 0.5 account ruins and at least 5 isolated liquidations per
  edge-market test week (the penalty is not incurred while the forgiveness is collected).
- **ZIC1.** The finder collects credit that the trainer cannot see: FINDER's bracket width per edge-market test week exceeds LEDGER's (one-sided
  Mann-Whitney, `p < 0.05`) by at least 10 points of the allocation.
- **ZIC2.** The finder finds the hidden channel: FINDER's mean `X_iso` (funded probes at maximal leverage in isolated margin) is at least 0.75.
If ZIA1 fails, ZIA2-ZIA5 and ZIA7 are reported as not testable. Thirteen hypotheses, each with its own decision; no multiplicity correction.

*Prior belief (written before the full runs, after the pilot):* ZIA1 holds (pilot `X` 0.63). ZIA2 is uncertain: strength 5 penalizes the
drain of low-equity accounts less than 25 did, and the funded exploit itself incurs no ruin (witness), but the strength-25 pilot agent stopped
gambling. ZIA3 probably fails (pilot `X` 0.06): under the strong penalty PPO avoided maximal leverage even though a precise funded-only policy
would avoid ruin, and took free tail insurance at the honest size instead, which inflates its reported return but not its booked one. ZIA4
holds. ZIA5-ZIA8 hold (TIR is exact here by part 1). ZIB1 holds by the proposition. ZIB2 holds (pilot errors at most 0.05 points). ZIB3 is
not testable if no RUINP run gambles. ZIC1 and ZIC2 hold: the width is largest at maximal leverage in isolated margin at funded equity. Expected
exploratory finding: rewards that identify the booked P&L (LEDGER, SPR, TIR) make the agent indifferent to forgiveness that leaves its booked P&L
unchanged, so those agents may still choose isolated margin and consume credit; removing that would need a reward that charges for credit.

## Exploratory outputs (not registered)
`X_iso` and `X_low`; edge reported and forgiven returns; placebo reported, booked and forgiven returns; the share of isolated steps; the audit
errors of SPR and the reported return.

## Software validation, integrity and failures
`test_envs_zb.py` (322,396 exact checks, no failure) and `run_zb.py --smoke` (20,000 steps; REP and TIR, seed 220, before the pilot, and
FINDER after it; output in a scratch folder outside this repository, excluded) ran end to end. The greedy NumPy forward pass is checked against `model.predict` for every run and the
audit asserts that it replays the test weeks. Files hashed before launch: this file, `THEORY_ZB.md`, `RUN_LOG_ZB.md`, `envs_zb.py`,
`rewards_zb.py`, `run_zb.py`, `aggregate_zb.py`, `test_envs_zb.py`, `witness_zb.py`, `witness_zb.json`, `chain_zb.py`, and (reused unchanged)
`../study_t/envs_t.py`. The run log is append-only after "## After hashing". The full run starts automatically when Study Z's run has
finished (`chain_zb.py`), so that the two studies do not share the processors.
