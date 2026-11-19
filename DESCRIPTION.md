# Repository description

## Repository name
`accounting-options-rl`

## GitHub "About" text (≤ 350 characters)
Simulators that forgive losses teach RL traders to gamble. Accounting rules as options with closed forms, 2,295 registered agents in an electricity sleeve and three public simulators (incl. backtesting.py), a black-box reading of a simulator's accounting from its reward, and twin rewards that remove the exploit.

## Suggested topics
`reinforcement-learning` `reward-hacking` `metamorphic-testing` `limited-liability` `option-pricing` `reward-misspecification` `counterfactual-reward` `simulation` `backtesting` `electricity-markets` `energy-trading` `imbalance-settlement` `evaluation-methodology` `pre-registration` `reproducibility`

## Abstract

Reinforcement-learning (RL) traders are trained and evaluated in simulators. A simulator's accounting decides who bears a loss beyond the agent's capital, and many forgive it: they floor equity at zero, write off the overshoot, or leave the bankruptcy step unrewarded. This hands the agent a free option. RL agents learn to exercise it by gambling near ruin, while evaluation reports a profit. Five accounting rules form an ordered ladder of such options, and approximate closed forms give their value. A finer time step removes a capped loss's subsidy but raises a zero floor's to twice that of terminal limited liability. In 2,295 registered agents on Nordic electricity spreads and three public trading simulators, the exploit was learned where training reached a forgiven floor. In the open-source backtester backtesting.py, PPO learned to bet maximal leverage near ruin in 12 of 16 registered seeds on a market without edge, against 1 of 16 in backtrader, which books the loss. No log whose market moves stay short of a floor can reveal it, however long; one query on a clone past the floor can. Along the market input, the reward is the booked profit plus a convex option payoff, so its shape locates hidden floors. Rewards built from a sign-reversed clone step removed the exploit in every registered test and mostly kept the honest edge, including where ruin penalties failed because losses were floored per account. In an electricity-trading engine, shortcuts ranked first a controller reporting +437% while booking -2,980%.

## Key facts
| | |
|---|---|
| Markets | Danish (DK1, DK2), Finnish (FI) and southern Norwegian (NO2) imbalance spreads for the learning studies; all twelve Nordic zones (eSett, 2021–2026) for the settlement analysis and the closed-form tests; daily EURUSD, Bitcoin and 2021 crypto moves in the public simulators |
| Learners | PPO, A2C and DQN (Stable-Baselines3 2.7.0); frozen MAPPO controllers in the engine audit |
| Studies | 2,295 distinct agents in registered studies: B–K 594, L 20, N 40, O 80, P 90, Q 100, R 60, S 180, T 170, U 80, V 144, W 48, X 32, Y 80, Z 136, RS 80, EXT 70, FT 81, RE 50, XE 56, GC 40, DR 64. M reuses L; C's snapshots and O rows reused in P are not new agents. Study A: separate 280; the exploratory backtesting.py control: 16 |
| Registered hypotheses | Internally hash-stamped: B–K 24/46 supported; N 6/9; O 4/4; P 3/4; Q 6/6; R 0/1; S 8/8; T 16/16; U 6/6; V 14/17; W 5/6; X 6/6; Y 12/13; Z 22/26; RS 6/8; EXT 16/17; FT 20/23; RE 3/3; GC 6/7; DR 1/4; XE 4/7. Unadjusted decision counts, not independent replications |
| Public simulators | Gym-Trading-Env (Study EXT): PPO at maximal leverage at 89% of probes, bust in 61% of test windows (3% and 2% on the public wealth). gym-mtsim (Studies G, GC, DR): the floor learned in the package's crypto market (low-equity exposure 0.28 against 0.08 with the floor deleted), not in its forex market. backtesting.py against backtrader (Study XE): learned in 12 of 16 seeds against 1 of 16; same engine with its two floor lines replaced (exploratory): 6 of 8 against 0 of 8 |
| Closed forms (exploratory) | What a capped loss, terminal limited liability and a zero floor forgive, from leverage, volatility, horizon, time step and fee; a finer time step removes the capped loss's subsidy and raises the zero floor's to twice the terminal one. Unmodified backtesting.py: 18 of 18 settings within 6% of the prediction; two pre-stated criteria failed and are reported |
| Reward diagram | The reward along the market input is the payoff diagram of the accounting: kinks locate floors and caps, tails give the exposure, the error falls with the range by an exact minimax law. On real clones of the engine (exploratory, 26 runs): within 0.0001 points in ten volume-payoff runs; fails its rule on the capped payoff; 8 of 12 rules met |
| Remedies | Sign-twin rewards (SPR, TIR) removed the exploit in every registered test and kept 0.95 to 1.16 of the ledger-trained edge with simulated signs; KL and occupancy regularization and quadratic utility did not do both; a ruin penalty did in one account but not with losses floored per account or per position (Study Z) |
| Engine audit | Shortcuts from an electricity-trading engine's earlier protocols ranked first a controller reporting +162% to +795% while booking −1,822% to −3,655%; reconciliation flagged 53 of 54 affected runs and none of 246 others |
| Audits | 18 public RL trading environments; 21 public electricity-RL environments (20 without a capital account) |
| License | Code: MIT. Docs and results: CC BY 4.0. Third-party data: see `THIRD_PARTY_LICENSES.md` |

## Status
Accompanies a manuscript submitted to *Engineering Applications of Artificial Intelligence*. Every registered study has its protocol, hashes, append-only run log, results and verdicts in its folder. `python tools/verify_studies.py` re-checks every registered hash, re-runs the exact checks and the unit tests of the twin rewards, re-runs the registered aggregators of Studies RE, GC, DR and XE on their stored results in a scratch copy, and re-checks headline numbers against the stored results; it does not re-train agents. Dated errata accompany the hashed protocols of Studies T, W, X, Z and EXT, and Study RS's README discloses its registration defect. Exploratory work, including every failure, is kept in the top-level `exploratory_*` folders and in the pilot folders of Studies GC, DR and XE.
