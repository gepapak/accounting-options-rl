# Repository description

## Repository name
`accounting-options-rl`

## GitHub "About" text (≤ 350 characters)
Every accounting rule is an option: RL traders learn to exploit loss forgiveness, evaluation reports it as profit, and a sign-paired reward prevents it without the ledger. Theory, 1,164 internally pre-registered agents on Nordic imbalance spreads, engine audit and settlement stress tests.

## Suggested topics
`reinforcement-learning` `reward-hacking` `limited-liability` `option-pricing` `reward-misspecification` `counterfactual-reward` `simulation` `electricity-markets` `energy-trading` `imbalance-settlement` `collateral` `backtesting` `evaluation-methodology` `pre-registration` `reproducibility`

## Abstract

Simulators that train reinforcement-learning (RL) traders decide what happens to losses beyond equity, and that decision is an option. Five common accounting rules, from full liability to a zero cash floor, form an ordered ladder of options; for fixed positions the floor is worth between one and two times Merton's call, and reported equity cannot identify booked performance. In over a thousand internally pre-registered agents trading Nordic imbalance spreads, RL traders learned to exercise these options: where no policy can profit, floor-trained agents reported +38% to +148% while booking -15% to -57%, and an audited energy-trading engine reported +270% for a controller that booked -535%. A pre-training diagnostic ordered when PPO and A2C acquire the exploit, but simple cost-aware scores ranked conditions equally and a prospective transfer failed. To prevent it without the ledger, we introduce a sign-paired reward that subtracts the same step's reward with the market's sign reversed; uniquely, it recovers the booked objective in fair markets while keeping any legitimate edge. In a pre-registered test with a real edge, PPO trained on the reported reward gambled and booked -6.5% per week; with the sign-paired reward it did not gamble, booking +22.6% and matching ledger training, whereas a KL penalty toward a safe policy kept the gamble or lost over half the edge; in fair markets it matched ledger training for PPO and A2C. After the 2025 Nordic reform, a historical collateral component would not have deterred a modeled three-week gamble in 36.8% of zone-months, up from 8.6%.

## Key facts
| | |
|---|---|
| Markets | Danish (DK1, DK2), Finnish (FI) and southern Norwegian (NO2) imbalance spreads for the learning studies; all twelve Nordic zones (eSett, 2021–2026) and Danish regimes (Energinet, 2019–2026) for the settlement analysis |
| Learners | PPO, A2C and DQN (Stable-Baselines3 2.7.0); frozen MAPPO controllers in the engine audit |
| Studies | 1,164 distinct completed main-study agents: B–K 594, L 20, N 40, O 80, P 90, Q 100, R 60, S 180. M reuses L; C's snapshots and O rows reused in P are not new agents. Study A: separate 280 |
| Registered hypotheses | Internally hash-stamped: B–K 24/46 supported; N 6/9; O 4/4; P 3/4; Q 6/6; R 0/1; S 8/8. Unadjusted decision counts, not independent replications |
| Prevention (Study S) | Sign-paired reward (r − r_twin)/2 − own cost. Edge market (PPO, DK1, 10 seeds per arm): reported reward gambled at 85% of low-equity probes and booked −6.5% of capital per week; ledger training 6% and +18.1%; SPR 0% and +22.6%; KL to a safe policy at β 0.003/0.03/0.3: 73%/17%/0% and −2.0%/+7.8%/+8.4%. Study O environment (20 pairs per learner): exploit shape L(0.02) − L(2) for reported reward / SPR / ledger, PPO 12.2/2.7/1.4, A2C 7.8/0.7/3.0. DQN not tested (exploratory failure) |
| Learnability | The pre-training index ordered acquisition by forgiveness, budget and learner (O 4/4, P 3/4, Q 6/6); simple cost-aware scores tie its arm ranking and Study R's prospective cost transfer failed. A shape reanalysis (`exploratory_claude_2026-10-16/reanalysis_shape/`) confirms the orderings |
| Frozen-controller audits | Exploratory statistical audits (`exploratory_novelty_2026-10-15/`, `exploratory_novelty_2026-10-16/`): silent on the guarded engine (0/40 cases); with the documented joint fault, cash-inflow certificates in 10/20 controllers at zero charge and 0/20 at full charge; learned components added nothing |
| Settlement scope | Historical volume-term proxy versus a modeled buffer under an external owner risk-cost hurdle; excludes other collateral and monitoring. No actual total-collateral adequacy, causal reform or conduct claim |
| Audits | 18 public RL trading environments (plus FinRL and TensorTrade); 21 public electricity-RL environments |
| License | Code: MIT. Docs and results: CC BY 4.0. Third-party data: see `THIRD_PARTY_LICENSES.md` |

## Status
Accompanies a manuscript in preparation for submission to *Engineering Applications of Artificial Intelligence*. Every registered study has its protocol, hashes, append-only run log, results and verdicts in its folder; `python tools/verify_studies.py` re-checks the registered hashes and headline numbers. Exploratory work, including every failure, is kept in the `exploratory_*` folders: Study S's development is in `exploratory_claude_2026-10-16/nfl_training/` (`PLAN.md`, `NFL_THEORY.md`).
