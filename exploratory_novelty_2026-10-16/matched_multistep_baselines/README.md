# Exploratory matched multi-step baselines

**A simple delayed open-loop policy also certifies harmful exposure.** The fixed
adaptive controller has a positive lower bound, but this case does not establish
that adaptive feedback is necessary. All earlier results and plans remain intact.

The comparison uses the previously selected DK1 empirical IID TRAIN law, cash
10,000 EUR, fee 1 EUR/MWh, horizon 96 and gamma .99. Before any new draws,
`EXPLORATORY_PLAN.md` and `matched_baselines.py` fixed every policy, seed, cap,
sample budget and inference rule. This is a disclosed follow-up on a selected
state, not new market validation or a comparison of policy-search algorithms.

Every distinct policy receives the same 8192 SEARCH paths for cap selection and
524288 fresh CONFIRMATION paths. Global 96-period row/sign paths are shared
across policies; individual whole trajectories are IID. One-sided bounds use
delta .01/9, giving simultaneous 99% coverage within this nine-policy family.
They do not combine with prior families into a paper-wide confidence guarantee.

| Fixed policy | Active periods | Search-selected cap, EUR | Capped residual mean, EUR | Lower bound, EUR | Certificate |
|---|---:|---:|---:|---:|---|
| Frozen adaptive | 96 | 32768 | 13.102237 | 8.202260 | Positive |
| Constant leverage 16 | 96 | 32768 | 6.246069 | -0.078660 | Abstain |
| Constant leverage 4 | 96 | 2048 | -55.478159 | -55.662331 | Abstain |
| Constant leverage 1 | 96 | 2048 | -14.104244 | -14.175555 | Abstain |
| Constant leverage .25 | 96 | 2048 | -3.533477 | -3.602541 | Abstain |
| Leverage 16, final 1 period | 1 | 2048 | -0.410975 | -0.699000 | Abstain |
| Leverage 16, final 4 periods | 4 | 8192 | -0.311599 | -1.278182 | Abstain |
| Leverage 16, final 16 periods | 16 | 32768 | 0.840011 | -1.877173 | Abstain |
| Leverage 16, final 48 periods | 48 | 32768 | 6.735658 | 2.427796 | Positive |

Open-loop leverage 16 for all 96 periods aliases constant leverage 16; it is
represented once rather than counted as a tenth test. A nonpositive lower bound
is abstention, not evidence that the policy has nonpositive expected reward.
No inferential comparison of controller means was planned or made.

For the final-48-period policy, the exact expected discounted booked return is
-78.799612 EUR. For the adaptive controller, the mandatory first trade gives an
exact booked-return upper bound of -3.335532 EUR. Thus both positive charged-return
certificates also establish booked harm under the stated fair-sign law. Sample
raw, residual, booked, cost and forgiveness means are descriptive and retained
in full, including the substantial finite-sample gross-return variation.

Independent direct integration reproduces every isolated-action value at the
deadline: leverage .25, 1, 4 and 16 give -0.021974392, -0.087713240,
-0.347968881 and -0.014332682 EUR. Flat gives zero. Earlier isolated actions
cannot improve this expectation with nonnegative costs and the same terminal
forgiveness. The positive open-loop policy therefore demonstrates multi-step
capability beyond isolated actions, while removing adaptive necessity as an
interpretation of this particular empirical case.

All nine confirmation budgets completed. The run took 7.359 process CPU seconds
and 7.541 wall seconds on one numerical thread. Counts are 4,497,408 SEARCH and
287,834,112 CONFIRMATION active transitions, plus 1098 vectorized source-validation
transitions: 292,332,618 total. Outer confirmation time slots are 452,984,832;
flat prefixes require no active simulator call. Separately, 36 paired arm/path
checks used 3456 frozen environment steps, with maximum discrepancy
9.095e-13 EUR. Maximum accounting-identity error was 5.457e-12 EUR.

`results.json` is the full source of truth, `policy_summary.csv` contains every
policy and descriptive endpoint, and `cap_search_scores.csv` retains all 36 cap
scores, including unselected caps. `run.log` records the original execution.
Hashes bind the old selected-controller output, old kernel, DK1 panel, frozen
environment files, this analysis script and its plan. No Study R input, learner
training, DP solve or old artifact modification was used. `verify_results.py`
checks these hashes and independently recomputes bound arithmetic, direct
isolated-action integration, booked expectations and transition counts without
new simulation; `verification.json` records its output.

From the code repository, the original command was:

```text
python exploratory_novelty_2026-10-16/matched_multistep_baselines/matched_baselines.py
```

The analysis refuses to overwrite completed or partial outputs. A reproduction
can place copies of the fixed script and plan in a new sibling folder beneath
`exploratory_novelty_2026-10-16` and run that copy. To check the shipped results
without simulation:

```text
python exploratory_novelty_2026-10-16/matched_multistep_baselines/verify_results.py
```
