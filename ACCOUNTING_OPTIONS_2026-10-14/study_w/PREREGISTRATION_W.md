# Study W pre-registration: choosing a cheap penalty from the reward channel alone

Written 2 November 2026, before any run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_W.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question and what precedes it
Study V found that a lump-sum ruin penalty is a competitive cheap remedy at one of three strengths (5 percent of the allocation: 91 percent
and 87 percent of the ledger-trained edge in two starting-equity regimes), that strengths 1 and 25 are not, and that the twin-based audit
(TIR) estimated every trained agent's booked return to within 0.39 point. A retrospective, unblinded look at the stored Study V episodes
(`exploratory_codex_2026-11-01/review_theory/`, with a re-computation in this repository's `study_w/retrospective_v.py`) showed that the strength
with the highest TIR-audited return is the strength with the highest booked return in both regimes. That look reused the data that had
produced the finding. Study W is the prospective test: in two settings that were not used for the ruin-penalty grid, can a trainer that sees
only the reward channel (plus a clone of the simulator for the audit) choose the working strength?

## Design
- **Settings.** The two further edge-market settings of Study T, with the same market draw, the same fixed test windows and start equities:
  `T_DK2` (DK2, p_edge 0.60, kappa 6) and `T_DK1M` (DK1, p_edge 0.58, kappa 4).
- **Learner and training.** As Study V (`run_v.py` unchanged, called with explicit settings, arms and seeds): PPO, 8 environments, n_steps 1024,
  batch 256, gamma 0.99, 64x64 network, reward normalization, 10^6 steps.
- **Arms.** The ruin penalty RUIN1, RUIN5, RUIN25 (percent of K0 per transition from positive reported equity to zero), as in Study V.
- **Seeds.** 180-187 in each setting and arm: 2 settings x 3 arms x 8 seeds = 48 runs.
- **References.** Study T's PLAIN and ORACLE runs in the same settings (`../study_t/runs_t.csv`, seeds 130-139, hashed before Study T ran).
  They share the test windows with Study W's runs, so differences reflect the policies. The ledger-trained arm is the reference for edge
  retention.
- **Measures per run.** `X` = share of the 16 reachable low-equity probes at maximal leverage (greedy policy); `E` = booked P&L in the edge
  market on the test split (percent of K0 per one-week episode, mean of 100 episodes); `T` = the TIR-audited estimate of the same return, made
  from the reward channel alone with twin queries at evaluation time (`audit_edge.tir` in the result files); `R` = the reported return in the
  edge market (`audit_edge.reported`). All as in Study V.

## Hypotheses and decision rules (`aggregate_w.py`)
Let G be the grid {RUIN1, RUIN5, RUIN25}. In a setting, the **audit-selected** strength is the member of G with the highest mean `T` over its 8
runs; the **booked-best** strength is the member with the highest mean `E`; the **report-selected** strength has the highest mean `R`. "Both"
for an arm means mean `X` <= mean `X`(ORACLE of Study T in the setting) + 0.10 and mean `E` >= 0.80 x mean `E`(ORACLE of Study T).
- **W1** (per setting): the audit-selected strength's mean `E` is within 1.0 point of the booked-best strength's mean `E`. (The 1.0-point tolerance
  is the audit-accuracy bound of Study V, VD1.)
- **W2** (per setting): the audit-selected strength does "both".
- **W3** (pooled over the 48 agents): the largest |`T` - `E`| of the 100-episode mean is at most 1.0 point, in the edge market and in the
  placebo market (the placebo-market estimate is `audit_placebo.tir` against `audit_placebo.booked`).
- **W4** (pooled over the two settings): in at least one setting, the report-selected strength's mean `E` is at least 1.0 point below the
  booked-best strength's mean `E` (the reported return is not a reliable selector). The prediction that motivates the audit.
Six decisions in all (W1 and W2 in each of two settings, W3, W4). No multiplicity correction. A missing or failed run is reported and
not replaced; if fewer than 6 of 8 runs of an arm are available in a setting, the setting's W1 and W2 are reported as not testable.

*Prior belief (written before the runs):* W3 holds (VD1 held in 128 of 128 agents). W1 is probable but depends on how close the two
best strengths are. W2 is uncertain: strength 5 worked in the two regimes of Study V, but DK2 and the weaker-edge setting have different
costs and edges. W4 is probably not supported: on Study V's full test sample the reported return also picked strength 5 in both regimes (it picked strength 1 on the
odd-numbered half of Part A, by 0.1 point), so the audit's value may lie in the level it recovers (the reported return of strength 1 overstated its
booked return by 6.8 and 15.2 points) rather than in the choice. This is registered so that the paper cannot claim a selection advantage that the
data do not show.

## Exploratory outputs (not registered)
The booked order of the three strengths; the X of each; the error of the SPR estimate; per-setting differences between report-selected,
audit-selected and booked-best strengths; the same selection on the odd and even test episodes.

## Software validation, integrity and failures
Reused unchanged from Study V: `run_v.py`, `rewards_v.py` and (from Study T) `envs_t.py`, `spr_t.py`. `aggregate_w.py` is new and was checked on
the stored Study V results (`python aggregate_w.py --results ../study_v/results_v --demo` prints the same selection tables for Study V's
settings as a software check; it writes nothing). Files hashed before launch: this file, `RUN_LOG_W.md`, `aggregate_w.py`, `run_v.py`,
`rewards_v.py`, `../study_t/envs_t.py`, `../study_t/spr_t.py`. The run log is append-only after "## After hashing".
