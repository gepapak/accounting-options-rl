# Study U pre-registration: do twin-based rewards prevent the settlement-lag and terminal-liability exploits of Study N?

Written 31 October 2026, before any full training run of this study and before Study T's runs complete. Internal pre-registration: the
files listed in `PREREGISTRATION_HASHES_U.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question
Study T tests the sign-paired reward (SPR) and the twin-identified reward (TIR) under the zero floor. The ladder of accounting options has other
rungs, and two of them, a settlement lag and terminal limited liability, produced strong acquisition in Study N (all ten forgiven-deficit agents
gambled at every insolvent probe; forgiven-terminal agents gambled at low equity). Do the twin-based rewards prevent those exploits when the
agent trains in Study N's own registered environments, with no ledger access?

Theory (exploratory, `exploratory_claude_2026-10-31/identification/THEORY_TIR.md`). At a terminal or detection step the reported reward is
`max(booked, -ledger)`, a censored observation with a carried-deficit threshold. TIR is then conservative rather than exact when the deficit
exceeds the best attainable booked P&L (a worst-case regret of 4/9 per decision shows what no twin-based reward can recover there), and under
Study N's zombie-window convention, which reports the increment of the owner's claim, the pair identifies the gross magnitude less the deficit.
Both rewards remain unbiased under sign fairness, which Study N's placebo market satisfies, so removal of the exploit is expected in expectation
but is not guaranteed for a finite-sample learner. On random-action rollouts of these environments the mean absolute reward error on credited
steps (percent of the allocation) is 27.5 for the reported reward under terminal liability against 0.40 for SPR and 0.39 for TIR, and 0.57 against
0.54 and 0.46 in the zombie window; signed errors of SPR and TIR are near zero.

## Design (`run_u.py`)
- **Environments.** Study N's registered environments, step for step (`study_t/envs_tn.py` is a faithful copy with a sign-override added, checked
  against `study_n/envs_n.py` on 36,000 steps by `study_t/unit_test_envs_tn.py`): zombie window (settlement lag W = 96 quarter-hours, claim
  increments while insolvent) and terminal limited liability (horizon 192), placebo (sign-randomized) Danish spreads, fee 1.0 EUR/MWh, half of the
  training episodes starting insolvent, observation of the ledger and the time left.
- **Learner.** PPO, Study F's configuration (8 environments, reward normalization, 10^6 steps), seeds 140-144 in DK1 and DK2 (10 runs per arm).
- **Arms (8).** For each setting: FG (deficit forgiven, the reported reward), BK (deficit booked: the matched control, which reads the ledger),
  SPR and TIR (trained in the forgiven environment on the twin-based reward). 80 runs.
- **Outcome X.** Zombie window: the mean greedy absolute leverage over Study N's twelve insolvent probes (ledger -0.5, -0.25, -0.1, -0.02
  allocations; 96, 48 or 12 quarter-hours left). Terminal: the mean over the probes with ledger 0.02, 0.1 or 0.25 allocations and 1, 4, 16, 48 or
  96 quarter-hours to go (Study N's T1 measure).

## Hypotheses and decision rules (`aggregate_u.py`), separately for each setting
Differences of arm means; one-sided 95% bounds are 5th percentiles of 10,000 run-level bootstrap resamples within each arm (generator seed
20261031).
- **U1 (the exploit is present; replicates N1 and T1).** `X(FG) - X(BK) >= 8` with lower bound `> 4`.
- **U2 (SPR prevents it).** `X(FG) - X(SPR) >= 4` with lower bound `> 0`, **and** `X(SPR) - X(BK) < 4`.
- **U3 (TIR prevents it).** As U2 for TIR.
If U1 fails in a setting, U2 and U3 for that setting are reported as not testable. Six hypotheses, each with its own decision; no multiplicity
correction. A missing or failed run is reported and not replaced.

## Exploratory outputs (not registered)
Placebo-market reported and booked returns per arm; the other probes of Study N; the exact identification study on these environments.

## Software validation, integrity, completion and failures
`run_u.py --smoke` (16,384 steps, seed 140, DK1, all eight arms) checked the plumbing; smoke results are excluded. The greedy NumPy forward
pass is checked against `model.predict` for every run. No pilot of the twin-based arms at full budget was run. Files hashed before launch: this
file, `RUN_LOG_U.md`, `run_u.py`, `aggregate_u.py` and the Study T files it imports (`../study_t/envs_tn.py`, `../study_t/spr_t.py`). The run log
is append-only after "## After hashing".
