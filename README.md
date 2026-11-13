# Range, not samples: reading hidden loss forgiveness from the reward

**Reading hidden loss forgiveness from the reward of reinforcement-learning trading simulators, with an electricity-trading engine.** Code, data pipelines, pre-registrations and aggregated results for the paper of that title (short name: "every accounting rule is an option").

## The question
An accounting rule decides what an agent's owner receives once losses exceed equity, and that claim is an option. Trading simulators often floor an account's cash at zero, end an episode at bankruptcy, or let losses accumulate until a final settlement; real settlement systems book losses with a lag and hold collateral. Limited liability rewards risk, most strongly near zero equity ("gambling for resurrection"). This repository asks which option each rule writes, whether reinforcement-learning (RL) agents *learn* to exercise it, whether standard evaluation then reports the behavior as profit, whether acquisition can be anticipated, how to prevent it when the simulator's ledger is not exposed, and how large the option is in real Nordic imbalance settlement.

## Main results
- **Five accounting rules form a ladder of options.** Full liability ≤ capped final loss ≤ settlement lag ≤ terminal limited liability (Merton's call) ≤ the zero floor. In a market without edge, for fixed positions, zero costs and independent symmetric moves, the floor's subsidy is between one and two times the terminal call, reaching two in the diffusion limit (adaptive policies can exceed the factor two); reported equity is the booked ledger reflected at zero (for a constant position from zero equity with exchangeable increments it has the law of the running maximum), and reported equity alone cannot identify booked performance. Charging every injection restores the booked objective with continued trading.
- **Origin.** In an audit of a validated electricity-market engine, shortcuts from its earlier protocols ranked a learned controller first, reporting **+162% to +795%** in all six runs while its booked P&L was **−1,822% to −3,655%** (volume payoff). A missing liquidity cap and missing solvency rules were jointly necessary; reconciling reported against booked returns flagged 53 of 54 affected runs and none of 246 others. **Correction (6 November 2026):** the audit's legacy protocol L7 (+270% reported, −535% booked) was described as using a capped percentage payoff, but the cap was off in every run, because evaluation restores the checkpoint's saved configuration. With the cap on, MAPPO reports −38.5% and −45.0% against −7.9% and −13.7% for the rule, and the floor binds in 2 of 20 runs. See Section 3 of the manuscript and the archived research record in `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-05/engine_twin/`.
- **Agents learn to exercise the options** (placebo markets with sign-randomized imbalance spreads, where no policy can profit):
  - floor-trained agents raised leverage as equity fell and reported **+38% to +148%** while booking **−15% to −57%** (Study B; replicated with growing false profit in Study C);
  - with equity as the only input, **all ten floor-trained PPO agents chose maximal leverage at the low-equity probe**, 11.1 units of 16 above full-liability agents (Cohen's d = 2.6; Study F); A2C and DQN also gambled (Study I); Finland reproduced the pattern and calm southern Norway did not (Study J);
  - charging injections in full cut low-equity leverage by 9.6 units [4.5, 13.8] with matched continued trading (Study L);
  - under a **settlement lag, all ten forgiven-deficit agents took maximal leverage at every insolvent probe** (Study N, 6 of 9 hypotheses).
- **Anticipating acquisition (Studies O, P, Q; R failed).** A pre-training advantage-to-noise index, with a threshold fixed before Study O, classified 5 of 6 confident treatments (O 4/4); the threshold fell with training budget as N^−0.43 [−0.63, −0.22] (P 3/4; four times more training found an exploit missed at one million steps); the ordering transferred to A2C and DQN (Q 6/6). Simple cost-aware scores rank the arms equally, and Study R's prospective cost-transfer prediction failed (R 0/1). A reanalysis with the exploit's shape L(0.02) − L(2) confirms the orderings (`_archive/exploratory_2026-11-13/exploratory_claude_2026-10-16/reanalysis_shape/`).
- **Preventing the exploit without the ledger (Study S, 180 agents, 8/8 registered hypotheses supported).** The sign-paired reward clones the simulator before each step, reverses the exogenous sign in the clone, and trains on (r - r_twin)/2 - own cost. It recovers expected booked reward under conditional sign fairness and preserves booked reward when neither branch receives credit. Uniqueness is within a constant-coefficient linear class; with an edge and credits it can attenuate legitimate returns. In the developed DK1 edge testbed, new-seed SPR PPO booked +22.6% of capital per week (ledger training +18.1%) and avoided maximal leverage at all saved low-equity probes. None of three tested KL strengths met both registered prevention criteria. In the zero-floor fair-market test, SPR reduced exploit shape for PPO and A2C and met the registered five-unit noninferiority criterion against ledger training. DQN is not claimed (exploratory failure). The exploratory theory note is retained unchanged in `_archive/exploratory_2026-11-13/exploratory_claude_2026-10-16/nfl_training/`; see `REVIEW_ERRATA_2026-10-29.md` there and manuscript Section 8 (Propositions 4-7) and its supplement for the corrected statements and proofs.
- **Recovering the ledger from the reward channel (Studies T and U, 250 agents, 22/22 registered hypotheses supported).** Under the zero floor, and under the other rules of the ladder at their crediting steps (except the zombie window of a settlement lag), the reported reward is the booked profit censored from below by a hidden threshold, `r = max(booked, tau)`; a simulator that floors several cash writes in sequence is outside this form. A sign twin (the same step with the exogenous sign reversed) identifies the booked profit exactly where the favourable branch is uncensored (`r != r'`), brackets it where both branches are censored, and cannot do better where a carried deficit hides it (worst-case regret 4/9 per decision, verified by a linear program). The twin-identified reward (TIR) is exact where the twin identifies the profit and uses the sharp lower end of the identified set for the gross magnitude elsewhere, with pathwise error never above SPR's; SPR's half-difference is a valid but generally non-sharp lower bound (g = 10, C = 1, tau = 0 gives the pair (9, 0): the pair identifies g = 10, SPR uses 4.5). Exact rational checks: 200,000 random cases in `_archive/exploratory_2026-11-13/exploratory_claude_2026-10-31/identification/`, a 2,070-case audit in `_archive/exploratory_2026-11-13/exploratory_codex_2026-11-01/review_theory/` and an identified-set search in `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-02/`. On a new DK1 market draw (Study T Part A) reported-reward PPO chose maximal leverage at 60% of low-equity probes and booked +2.0% of capital per week against +20.5% for ledger training, SPR 0% and +19.4%, TIR 2.5% and +21.3%; none of five ORPO strengths (occupancy-measure regularization, implemented from the authors' code semantics) did both (strong strengths reproduce the reference policy's +7.7%, weak ones regain the exploit). The result held in DK2 and in a weaker-edge DK1 setting. In Study N's settlement-lag and terminal-liability environments (Study U) both rewards brought acquisition down to the ledger control's level. TIR's exactness did not give a measurable learned advantage over SPR. The estimators are packaged and tested in `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/`.
- **Cheaper remedies, a reward without a clone, and a ledger-free audit (Study V, 144 agents, 14/17 registered hypotheses supported; VA3, VA5 and VC6 not supported).** A lump-sum ruin penalty (`RUIN`) and the quadratic-utility reward `r - lambda r^2` (`MV`, registered as "mean-variance") were compared with the twin rewards, together with the pooled-magnitude reward (PMR), which needs no clone: on favourable steps (r + cost)/|q| is the per-unit move when the favourable branch is uncensored, and PMR rewards sigma·|q|·(running mean of these) − cost (Assumption M: the move law does not depend on equity, action or sign). Reported-reward PPO gambled in 15 of 16 agents (Parts A and C); PMR removed the exploit in 31 of 32 agents (one Part A agent gambled, so VA3's edge bound, 0.48 against 0.60, failed) and booked 18.0%, 22.0%, 35.1% and 15.1% against 20.9%, 19.8%, 32.7% and 11.6% for ledger training; no quadratic-utility weight kept the edge (+8.6%, +4.5%, +1.1%); **a ruin penalty of 5% of the allocation removed the exploit and kept 91% (Part A) and 87% (Part C, half of the training episodes starting at zero equity) of the ledger edge, so VA5 and VC6 were not supported** (strength 1 left the gamble, strength 25 stopped trading from zero equity). Summing the TIR estimate over each agent's test episodes recovered the booked return of all 128 audited agents within 0.39 points; the reported return was off by a median of 68 points for the 28 agents that gambled. See `ACCOUNTING_OPTIONS_2026-10-14/study_v/` and `_archive/exploratory_2026-11-13/exploratory_claude_2026-10-31/study_v_development/` (pilots, theory note).
- **Choosing the penalty's strength from the reward channel (Study W, 48 agents, 5/6 registered decisions supported; W4 not supported).** In two further edge-market settings (DK2 and a weaker-edge DK1 sleeve) the ruin-penalty strength with the highest TIR-audited return was the strength with the highest booked return (strength 5 in DK2, strength 1 in the weaker-edge sleeve), and it did both (102% and 96% of the ledger-trained edge, no probe at maximal leverage). Study W's W2 judges a strength by arm means; under the common criterion of Studies S to Y, which adds a one-sided Mann-Whitney test, strength 1 in DK2 does not remove the exploit (p = 0.076), while the two selected strengths meet both (`_archive/exploratory_2026-11-13/exploratory_claude_2026-11-03/w_common_criterion.py`, `study_w/ERRATA_W.md`). The TIR audit error was below 0.08 points for all 48 agents. The reported return would have chosen the same strength (W4 not supported, as registered beforehand), so in these settings the audit supplies the level of the return, not a better choice. See `ACCOUNTING_OPTIONS_2026-10-14/study_w/`.
- **Do the twin rewards survive a violated single-floor assumption? (Study X, 32 agents, 6/6 registered decisions supported).** The public engine writes its cash in several floored steps, so the single-floor assumption behind SPR and TIR fails there; the exact cost-shift test (`twin_reward.cost_shift_test`, which decides whether any single-floor world reproduces the twin pairs at shifted costs) flags it. In a two-write variant of the edge-market sleeve (the opening fee written first with its own floor; half of the training episodes from zero equity), reported-reward PPO gambled in 8 of 8 agents and booked -13.4% of the allocation per week against +18.5% for ledger training; SPR and TIR gambled in 0 of 16 and booked +20.7% and +18.2% (1.12 and 0.99 of the ledger edge). The TIR audit error was at most 0.0065 points for the 24 agents that did not gamble and off by a median of 1.4 points (edge market) for the 8 that did, where the reported return was off by a median of 27. Graceful degradation in one sleeve, one learner and one write order, not evidence that the assumption is unnecessary. See `ACCOUNTING_OPTIONS_2026-10-14/study_x/`.
- **A second learner (Study Y, 80 agents, 12/13 registered decisions supported; YC1 not supported).** A2C (rollout 64, as in Studies I, Q and S) in the single-floor and the two-write sleeve, with half of the training episodes from zero equity: reported-reward A2C gambled in 5 of 8 agents in each sleeve and booked +5.1% and +2.6% of the allocation per week against +21.1% for ledger training; SPR and TIR gambled in 0 of 32 and kept 1.00 to 1.04 of the ledger edge; the TIR audit of the 48 ledger, SPR and TIR agents was off by at most 0.0025 points. Without zero starts no reported-reward A2C agent gambled (YC1, anticipated in the registration), so A2C acquired the exploit only when training visited insolvent accounts. See `ACCOUNTING_OPTIONS_2026-10-14/study_y/`.
- **A ledger-free bracket that needs no single floor (exploratory, 3 November 2026).** If the accounting only forgives (credits >= 0), the twin pair brackets each step's booked P&L in `[2*SPR - r, r]` (`twin_reward.twin_bracket`), a sharp interval whose width is the credit paid on the pair. On the stored audits of Studies V, W, X and Y (304 agents, 60,800 test episodes, single and two-write floors, PPO and A2C) no booked return lay outside its bracket, and on the public engine's own cash-writing methods it covered the booked value in all six hand cases, of which TIR's point estimate is wrong in four (`_archive/exploratory_2026-11-13/exploratory_claude_2026-11-03/twin_bracket.py`). Its premises are credits that are non-negative at every step on both branches and an own cost known and equal on both: the settlement lag's zombie window and a charge on one branch only break it, and the width, which covers both branches, was 1.35 to 2.42 times the forgiveness realized on the path (`_archive/exploratory_2026-11-13/exploratory_claude_2026-11-05/bracket_limits.py`).
- **When the trainer cannot see where losses are floored (Study Z, 136 agents, 22/26 registered decisions supported).** Part A: two accounts with their own floors trade the same spread (both balances visible); Part B: one balance and a margin mode whose isolated margin limits a position's loss to its own cost plus 100 EUR/MWh (the floors are hidden from the trainer). Reported-reward PPO gambled in both (booked -18.3% and -19.0% of the allocation per week against +33.8% and +20.6% for ledger training). **The ruin penalty measured on the portfolio or the balance did not remove the exploit at strength 5 or 25**; measured per account it did (79% of the edge), and a liquidation penalty, which needs the margin rule, did at 36% of the edge. SPR and TIR, from the portfolio's reward channel, removed it and kept 1.15 and 1.07 (Part A) and 0.99 and 1.02 (Part B) of the ledger-trained edge. A ledger-free finder, an agent rewarded by the twin bracket's width, found the hidden isolated-margin channel in all eight seeds. The bracket held on every audited step of all 136 agents. Not supported: ZA10 (pooled control, X 0.15 against 0.10), ZB3 (park share 0.47 against 0.50), ZIB2 (TIR audit 1.45 points, from the finder agents; 0.29 for the other 56, see `study_zb/ERRATA_ZB.md`) and ZIB3. See `ACCOUNTING_OPTIONS_2026-10-14/study_z/` and `study_zb/`.
- **Real signs (Study RS, 80 agents, 6/8 registered decisions supported; RS3 and RS6 not).** A stress test of the twin rewards with the real DK2 spread, sign and magnitude, and a signal from the sign eight panel rows earlier (exactly two hours on 99.8% of test rows); the zero floor, the sign-reversed payoff and equal costs stay constructed. Reported-reward PPO gambled at 83% of the low-equity probes and booked -100.6% of the allocation per week while reporting +9.2%, against +8.5% booked for ledger training. SPR and TIR chose maximal leverage at no probe. **TIR kept 0.997 of the ledger-trained edge (bootstrap lower bound 0.755) and met the registered rule; SPR kept 0.687 (0.370) and did not;** a ruin penalty of 5 met the probe tolerance but booked -9.5%. This is a threshold contrast, not a shown superiority: TIR minus SPR is +2.64 points, exploratory 95% interval -0.87 to +6.29, Welch p = 0.18. The market, the lag and the zero starts were chosen after pilots scored on the same test panel, the 100 test windows overlap and hold about 31 separate weeks, and the run log's registered digest covers only its first 147 bytes through a hashing bug (`study_rs/README.md`). See `ACCOUNTING_OPTIONS_2026-10-14/study_rs/` and `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-05/rs_reanalysis.py`.
- **The twin on the engine's own paths (exploratory, 6 November 2026; no agent trained).** A read-only tracer of every cash write re-evaluated the engine audit's frozen controllers: the 54 runs in which the floor bound and 39 in which it did not, each reproducing the audit's return within 1e-12 points, under rules hashed after three pilot runs. The twin is a one-step replay of the cash writes, in the engine's order, with the gross reversed. **The bracket contained the booked P&L on all 3,655,923 steps, and the reported return minus the summed TIR estimate flagged the same 53 of 54 runs as the ledger's reconciliation and none of the other 39.** Neither point estimate was safe: with the volume payoff TIR erred by a median of 43 points and SPR by 618, with the percentage payoff TIR by 608 and SPR by 11. Two of seven rules failed narrowly. A direct calculation was exact with the formula in force and off by medians of 1,925 and 838 points with the other protocol generation's. The engine exposes its position and gross P&L, so this tests the bracket on realistic accounting and does not show that a twin is needed there. See `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-05/engine_twin/`.
- **The reward diagram (theory with exact checks, 6 November 2026).** Freeze a step on a clone and move the market input along a line, `input(t) = reference + t * (input - reference)`: `t = 1` is the real step, `t = -1` the sign twin, `t = 0` no move. For an accounting built from floored cash writes the reward is the booked P&L plus a convex non-negative credit, so it is the payoff diagram of the option book the accounting hands the agent: a kink is a floor, a flat tail reads the balance or the margin, opposite tail slopes are accounts of opposite exposure, a concave kink is a cap, a jump is a reward replaced on a branch. A zero width at scale `T` certifies that nothing is forgiven on `[-T, T]`. The endpoint estimator at range `L` has exact minimax error `(E + A + 2 eta) / L`, so the resource is the range and not the number of queries. Three witnesses show what the whole line cannot separate. `ACCOUNTING_OPTIONS_2026-10-14/twin_reward/reward_diagram.py` is the reader, and `test_reward_diagram.py` runs 88,702 checks in rational arithmetic. See also `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-06/reward_diagram/` (what it reads in every study sleeve) and `sign_cost/` (a one-sided certificate that needs no clone).
- **The reading on real clones of the engine (exploratory, 7 November 2026; no agent trained).** 26 evaluations of the engine audit's frozen controllers, every query being the engine's full, unchanged step on a deep copy, under rules hashed after four pilot runs: 170,326 read steps and 674,523 clones. **With the volume payoff (10 runs) the reading recovered the booked return within 0.0001 points at 2.5 to 5.1 clones per step, where audits from the same clones erred by 9 to 390 points (TIR) and 31 to 1,410 (SPR).** With the uncapped percentage payoff it ties SPR (13 to 18 points summed over read steps; TIR 826 to 1,228). A concave kink at `|t| <= 2` appears on 98.5 to 99.4% of the moved steps with the cap on and at most 0.6% with it off, so the reward alone separates the two protocols that the audit confused. **Eight of twelve rules were met.** Not met: S1 (two liquidation steps differ from the replay), S2 (the bracket misses on 5,278 steps, by at most 485 DKK, all where the own cost differs between the branches), S6 (capped payoff with the floor: 4.8 and 7.3 points against a limit of 1, SPR 0.2 and 0.3) and S9 (the flat level equals minus the balance within 1 DKK on only 87%). The reading needs the own cost declared: read from `t = 0` it errs by 188 to 882 points, as TIR and SPR would. Direct calculation stays exact, so this does not show that a reading is needed on this engine. See `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-06/engine_scan/`.
- **An unmodified public simulator (Study EXT, 70 agents, 16/17 registered decisions supported; EA9 not).** The official Gym-Trading-Env at a pinned commit, byte-identical and checked at import, with a linear reward through its custom-reward function, positions up to +-10 and a placebo market of real daily Bitcoin moves with fair signs. On a step that takes the valuation to zero or below the engine ends the episode and does not call the reward function. **Reported-reward PPO chose maximal leverage at 89% of probes and went bust in 61% of test windows, reporting +50.4% per episode while booking -9.8%, against 3% and 2% on the public wealth.** SPR, TIR and TSR removed it; ruin penalties of 5 and 25 did not; one of 100 cut it to 18%, above the registered limit (three of eight seeds kept gambling). The finder chose maximal leverage in 8 of 8 seeds, and its negative control in the default position set had width 1e-13. The tail reading was exact on all 8,000 states and all 3,505 jumps were located and sized from the reward. The simulator exposes its wealth, so this shows that the exploit and the remedies carry over, not that a twin is needed. With the default logarithmic reward (exploratory, eight seeds, `study_ext/results_log/`) agents stayed nearly flat (X at most 0.011, no bust), so here the exploit needs the linear reward as well as the leverage. See `ACCOUNTING_OPTIONS_2026-10-14/study_ext/`.
- **A correct copy of the reward's definition certifies the floor (exploratory, 7 November 2026; no agent trained).** On unmodified gym-mtsim 2.0.0 (EURUSD, 100 held positions, 11 probes each at +-1, 2, 4, 8, 16 times a 1% move, seed 20261107) the reward is defined as the change in equity, and a reimplementation of that definition matched the returned reward on all 1,100 probes while the zero-balance write in `tick()` hid 11,587 to 11,846 USD of the order's loss at the -16% probe. The slope of the reward around a zero move recovers the exposure (relative error 1e-14) and the economic profit to 1.6e-10 USD; a control subclass that deletes the two floor lines passes 100 of 100 probes. See `_archive/exploratory_2026-11-13/exploratory_line_2026-11-07/` (`PROTOCOL.md`, `run_formula_gap.py`, `results.json`).
- **Logs do not buy range (proposition with a witness, 7 November 2026).** Two floored programs that agree on `[-lambda, lambda]` write identical transition logs of any length whose moves lie inside it, so no estimator from such logs separates them, and one query outside does. Witness on the same gym-mtsim positions: 5,000 logged moves inside 0.4% are identical with and without the floor lines, an OLS line through them recovers the exposure (relative error 5e-16) and, extrapolated to the -16% probe, misses the public reward by 11,827 USD while matching the unfloored control to 7e-12 USD. See `_archive/exploratory_2026-11-13/exploratory_line_2026-11-07/` (`PROTOCOL_RANGE.md`, `run_range_logs.py`, `results_range.json`).
- **A two-action learner on the same simulator (exploratory, 8 November 2026).** Flat or the margin maximum, equity 10,000 USD, one bar, 1,000 Student-t(3) draws and 20 seeds. The line through the published reward at a zero move matches the floor-deleted step within 1.5e-9 USD. At 0.15% per bar both learners stay flat. At 1.2% the published mean exceeds that line by 504 to 750 USD on 20 of 20 seeds. A pre-registered count of 18 commits to the maximum was not met (16 of 20). See `_archive/exploratory_2026-11-13/exploratory_mtsim_reach_2026-11-08/` (`PROTOCOL_LEARN.md`).
- **Reach before training (Study RE, 50 PPO runs, 3/3 registered decisions supported).** Scaling the Study F sleeve's spreads and fee by `c` in {1, 0.3, 0.1, 0.03, 0.01} moves only how often training can reach the floor. At `c = 0.01` the predicted reach is 0, no run hit the floor, and all five floored/unfloored pairs have identical probes and identical parameter digests. At full scale the exploit gap averages 13.2 units of leverage. See `ACCOUNTING_OPTIONS_2026-10-14/study_re/`.
- **Accountings that a sign query cannot read (Study FT, 81 agents, 20/23 registered decisions supported; FB4, FB6 and FB7 not).** The edge-market sleeve with an unrealized bucket (Part A), a carried deficit with half of the training funded (Part B) and a carried deficit only (Part C). Reported-reward PPO gambled in all three (X 0.98, 0.33, 0.75; from a deficit -30.6%, -1.3%, -27.3% per week against about +20.6%). **SPR and TIR removed the exploit everywhere, but what they kept differs: 0.80 and 0.84 of the edge with the bucket, 0.01 and 0.002 with a deficit and no funded training (the study's main prediction), and 0.55 and 0.58 with half funded, where our prediction from one pilot seed failed.** TSR matched the ledger arm exactly on each of its three seeds, as it must by construction. From a deficit the twin audits err by about 30 points per week on honest agents and the reading by at most 1.1e-12. See `ACCOUNTING_OPTIONS_2026-10-14/study_ft/`.
- **Model selection.** In an independent certification cohort, the winners selected on reported return (+108.8% and +104.3%) were certified worst in expected booked return within their ten-agent sets, with one joint 95% guarantee (Study M).
- **Historical settlement-component stress test (exploratory).** A proxy for the historical eSett volume term was below a modeled three-week deterrence buffer in **36.8%** of post-March-2025 FI/SE/NO zone-months versus **8.6%** before. It excludes the invoice component, minimum collateral and monitoring, and applying the historical formula to Finland after November 2024 and to Denmark is counterfactual; it is not an assessment of total collateral adequacy or conduct. The fee-free floor/terminal ratio was 1.62–1.99 across 24 zone-years.
- **Boundaries.** With realistic trading costs (Study H), inside gym-mtsim (Study G) and with equity-scaled positions in our sleeve (Study E), agents did not learn to gamble, or did so only weakly. That scope is not general: in Gym-Trading-Env, whose positions scale with wealth, leaving the bankruptcy step's loss unrewarded sufficed (Study EXT). Reported returns still exceed booked ones wherever a floor binds.
- **Prevalence.** Two of the 18 most-starred public RL trading environments forgive losses: one by default with orders of fixed size, the other only when users enable shorts or leverage (Study D); 20 of 21 public electricity-RL environments have no capital account.
- **Tools.** Reconciliation of reported against booked returns, a placebo-market test, simultaneous cost-based certificates, the reader of the reward diagram (`ACCOUNTING_OPTIONS_2026-10-14/twin_reward/reward_diagram.py`), the sign-paired and twin-identified rewards, the exact cost-shift test of their single-floor assumption (`twin_reward/`; a falsification test, not a certificate) and an evaluation checklist (`CHECKLIST.md`). Dated errata for statements in hashed protocols: `study_t/ERRATA_T.md`, `study_w/ERRATA_W.md`, `study_x/ERRATA_X.md`, `study_zb/ERRATA_ZB.md` (the protocols are unchanged); Study RS's registration defect is described in `study_rs/README.md`. Analyses added after the fourth AI-assisted review (audit cohorts, Study W under both criteria, Holm sensitivity, validation of the exact test) and Study RS reanalysis are archived in `_archive/exploratory_2026-11-13/`.

All registered hypotheses are reported: B–K supported 24/46, N 6/9, O 4/4, P 3/4, Q 6/6, R 0/1, S 8/8, T 16/16, U 6/6, V 14/17, W 5/6, X 6/6, Y 12/13, Z 22/26 (11/13 in each part), RS 6/8, EXT 16/17 and FT 20/23. These are shared-outcome decision counts, not independent replications or family-wise guarantees; a Holm adjustment of every registered significance test changes no decision within any of Studies S to Z, RS and EXT and four within Study FT; across all 69 tests of these studies 17 of the 65 supported decisions lose significance (largest adjusted p 0.0825), though none does across the 49 of Studies S to Z and RS alone (`_archive/exploratory_2026-11-13/exploratory_claude_2026-11-03/holm_sensitivity.py`). The completed main-study cohort has **2,085 distinct agents** (B–K 594; L 20; N 40; O 80; P 90; Q 100; R 60; S 180; T 170; U 80; V 144; W 48; X 32; Y 80; Z 136; RS 80; EXT 70; FT 81), excluding the separate **280-agent Study A**. C's five snapshots per training path do not add agents, M reuses L's checkpoints, and P's 170-row CSV includes O's 80 reused runs.

## Studies and where to find them
| Study | Question | Folder | Pre-registration | Aggregated results |
|---|---|---|---|---|
| Engine audit | Which evaluation shortcuts decide the verdict on frozen controllers? | `engine/`, `audit/`, `results/` | `docs/PREREGISTRATION.md` | `results/` |
| A | When do simulator shortcuts reverse "learning beats the rule" (three environment designs)? | `GENERALIZATION_STUDY_2026-09-29/` | `PREREGISTRATION_A.md` | `results/` there |
| B | Do agents learn to gamble under a zero floor? | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_B.md` | `results/` |
| C | Does the gambling grow with training (fresh seeds)? | `LIABILITY_STUDY_2026-09-29/scaling/` | `PREREGISTRATION_C.md` | `scaling/results/` |
| D | How common are the enabling conditions in public trading environments? | `LIABILITY_STUDY_2026-09-29/env_audit/` | `PREREGISTRATION_D.md` | `coding_d.csv` |
| E | Equity-scaled positions and the log reward | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_E.md` | `results_e/` |
| F | Well-posed learning (PPO, equity as the only input) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_F.md` | `results_f/`, pilot `results_pilot_f/` |
| G | A public simulator (gym-mtsim) | `LIABILITY_STUDY_2026-09-29/mtsim_study/` | `PREREGISTRATION_G.md` | `results_g/` |
| H | Realistic trading costs | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_H.md` | `results_h/` |
| I | Other learners (A2C, DQN) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_I.md` | `results_i/`, pilot `results_pilot_i/` |
| J | Other markets (Finland, southern Norway) | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_J.md` | `results_j/` |
| K | Patience: does the gambling region widen with the discount factor? | `LIABILITY_STUDY_2026-09-29/` | `PREREGISTRATION_K.md` | `results_k/` |
| L | Reward correction with matched continued trading | `revision_experiments/2026-10-10/` | `continuation_protocol.txt`, `prospective_manifest.json` | `continuation_experiment/` |
| M | Independent model selection and certification on the L checkpoints | `revision_experiments/2026-10-10-selection/` | `frontier_certification/prospective_manifest.json` | `frontier_certification/`, `VALIDATION.json` |
| N | Settlement lag (zombie window) and terminal limited liability | `ACCOUNTING_OPTIONS_2026-10-14/study_n/` | `PREREGISTRATION_N.md`, `PREREGISTRATION_HASHES_N.txt` | `runs_n.csv`, `VERDICTS_N.md` |
| O | Does a pre-training learnability index predict whether PPO learns the exploit (partial liability)? | `ACCOUNTING_OPTIONS_2026-10-14/study_o/` | `PREREGISTRATION_O.md`, `PREREGISTRATION_HASHES_O.txt` | `runs_o.csv`, `VERDICTS_O.md` |
| P | Does the training budget shift the learnability threshold as 1/√N? | `ACCOUNTING_OPTIONS_2026-10-14/study_p/` | `PREREGISTRATION_P.md`, `PREREGISTRATION_HASHES_P.txt` | `runs_p.csv`, `VERDICTS_P.md` |
| Q | Does the learnability index transfer to A2C and DQN? | `ACCOUNTING_OPTIONS_2026-10-14/study_q/` | `PREREGISTRATION_Q.md`, `PREREGISTRATION_HASHES_Q.txt` | `runs_q.csv`, `VERDICTS_Q.md` (6/6 supported) |
| R | Does the frozen exact one-step score transfer across fees at the same charge share? | `ACCOUNTING_OPTIONS_2026-10-14/study_r/` | `PREREGISTRATION_R.md`, `PREREGISTRATION_HASHES_R.txt` | `runs_r.csv`, `analysis_r.json`, `VERDICTS_R.md` (R1 not supported) |
| S | Does a sign-paired reward meet the registered prevention criteria, compared with three tested KL strengths? | `ACCOUNTING_OPTIONS_2026-10-14/study_s/` | `PREREGISTRATION_S.md`, `PREREGISTRATION_HASHES_S.txt` | `runs_s_A.csv`, `runs_s_B.csv`, `VERDICTS_S.md` (8/8 supported) |
| T | Do sign-paired and twin-identified rewards replicate on a new market draw and in two further settings, compared with ledger training and five ORPO strengths? | `ACCOUNTING_OPTIONS_2026-10-14/study_t/` | `PREREGISTRATION_T.md`, `PREREGISTRATION_HASHES_T.txt` | `runs_t.csv`, `VERDICTS_T.md` (16/16 supported) |
| U | Do SPR and TIR prevent the settlement-lag and terminal-liability exploits of Study N? | `ACCOUNTING_OPTIONS_2026-10-14/study_u/` | `PREREGISTRATION_U.md`, `PREREGISTRATION_HASHES_U.txt` | `runs_u.csv`, `VERDICTS_U.md` (6/6 supported) |
| V | Do a ruin penalty, a quadratic-utility ("mean-variance") reward and a clone-free pooled-magnitude reward (PMR) match the twin rewards, and can twin estimates audit a trained agent without the ledger? | `ACCOUNTING_OPTIONS_2026-10-14/study_v/` | `PREREGISTRATION_V.md`, `PREREGISTRATION_HASHES_V.txt` | `runs_v.csv`, `VERDICTS_V.md` (14/17 supported; VA3, VA5, VC6 not) |
| W | Can the TIR audit choose the ruin-penalty strength, in two settings outside Study V's grid? | `ACCOUNTING_OPTIONS_2026-10-14/study_w/` | `PREREGISTRATION_W.md`, `PREREGISTRATION_HASHES_W.txt` | `runs_w.csv`, `VERDICTS_W.md` (5/6 supported; W4 not) |
| X | Do SPR and TIR still prevent the exploit when the reward is a two-write (not single-floor) censor? | `ACCOUNTING_OPTIONS_2026-10-14/study_x/` | `PREREGISTRATION_X.md`, `PREREGISTRATION_HASHES_X.txt` | `runs_x.csv`, `VERDICTS_X.md` (6/6 supported) |
| Y | Do SPR and TIR transfer to a second learner (A2C), in the single-floor and two-write sleeves? | `ACCOUNTING_OPTIONS_2026-10-14/study_y/` | `PREREGISTRATION_Y.md`, `PREREGISTRATION_HASHES_Y.txt` | `runs_y.csv`, `VERDICTS_Y.md` (12/13 supported; YC1 not) |
| Z, Part A | Does a ruin penalty measured on the portfolio still work when each account has its own floor, and do SPR and TIR? | `ACCOUNTING_OPTIONS_2026-10-14/study_z/` | `PREREGISTRATION_Z.md`, `PREREGISTRATION_HASHES_Z.txt`, `THEORY_Z.md` | `runs_z.csv`, `VERDICTS_Z.md` (11/13 supported; ZA10, ZB3 not) |
| Z, Part B | The same when the floors are hidden from the trainer (isolated margin), and can a ledger-free finder find them? | `ACCOUNTING_OPTIONS_2026-10-14/study_zb/` | `PREREGISTRATION_ZB.md`, `PREREGISTRATION_HASHES_ZB.txt`, `THEORY_ZB.md`, `ERRATA_ZB.md` | `runs_zb.csv`, `VERDICTS_ZB.md` (11/13 supported; ZIB2, ZIB3 not) |
| RS | Do the twin rewards hold with real signs? A stress test on one zone and a reused test panel | `ACCOUNTING_OPTIONS_2026-10-14/study_rs/` | `PREREGISTRATION_RS.md`, `PREREGISTRATION_HASHES_RS.txt` (disclosed run-log digest defect) | `runs_rs.csv`, `VERDICTS_RS.md` (6/8 supported; RS3, RS6 not), `independent_verification/` |
| EXT | Does the exploit, and do the remedies and the reading, carry over to an unmodified public simulator (Gym-Trading-Env)? | `ACCOUNTING_OPTIONS_2026-10-14/study_ext/` | `PREREGISTRATION_EXT.md`, `PREREGISTRATION_HASHES_EXT.txt`, `ERRATA_EXT.md` | `runs_ext.csv`, `VERDICTS_EXT.md` (16/17 supported; EA9 not) |
| FT | What do agents learn where a sign query cannot see the gross (unrealized bucket, carried deficit)? | `ACCOUNTING_OPTIONS_2026-10-14/study_ft/` | `PREREGISTRATION_FT.md`, `PREREGISTRATION_HASHES_FT.txt` | `runs_ft.csv`, `VERDICTS_FT.md` (20/23 supported) |
| Settlement | The ladder and collateral in twelve Nordic zones (exploratory) | `ACCOUNTING_OPTIONS_2026-10-14/` | not pre-registered | `ladder12.csv`, `coverage12_*.csv`, `frontier12_*.csv`, `tails12.csv` |

Studies L and M are called "revision experiment L" and "revision audit M" in their frozen records. They were frozen before their runs but registered after the scorecard of Studies B–K, and are reported separately from its 46 hypotheses. Study N was registered on 14 October 2026; its `RUN_LOG_N.md` is append-only, and the registered version is the prefix before the line `## After hashing`.

Each study folder of A–K also holds `PREREGISTRATION_HASHES.txt` (SHA-256 of every registered file, with UTC timestamps) and `RUN_LOG.md` (launches, interruptions, relaunches and changes of job order). Theory solvers: `dp_theory.py` (Denmark), `dp_frictions.py` (Study H), `dp_j.py` (Study J), `dp_full.py` (full training distribution, all zones; supersedes the 3,000-sample solves, see `RUN_LOG.md`), `theory_k.py` and `reflection_barrier.py` (closed-form barrier), and `study_n/dp_n.py`, `dp_n2.py` (Study N). Best-moment theorem: `make_best_moment_figure.py` (writes `results_k/best_moment.csv` and the figure; `best_moment.py` is its earlier 1,000-draw version) and `best_moment_zones.py` (twelve zones). Other theory checks: `action_gap.py`, `false_profit_bound.py`, `partial_liability.py`, tables by `make_theory_tables.py`. Placebo-market test: `placebo_market.py`. Exploratory analyses are in `review_rigor.py` and `explore_i.py`.

## Repository layout
```
README.md, DESCRIPTION.md        overview; short description and repository metadata
CITATION.cff                     citation metadata
CHECKLIST.md                     evaluation checklist for RL trading agents, their simulators and settlement rules
LICENSE, LICENSE-docs.md         MIT (code); CC BY 4.0 (docs, results, derived tables)
THIRD_PARTY_LICENSES.md          sources, licenses and attribution for all third-party material
requirements*.txt                standalone analyses; engine stack; Studies A-Z, RS, EXT and FT; Study G (gym-mtsim)
tools/verify_studies.py          registered hashes; completed counts and headlines; theory/settlement checks
tools/verify_accounting_revision.py   exact rational-arithmetic checks of the reflection identity and reward correction
tools/verify_reflected_control_revision.py   numerical check of the reflected-control candidate (partial charges)
tools/verify_release.py          recomputes every headline number of the engine audit
tools/ledger_check.py            standalone reported-vs-booked-P&L reconciliation
GENERALIZATION_STUDY_2026-09-29/ Study A: environments, runner, aggregation, Energinet data and panels, results
LIABILITY_STUDY_2026-09-29/      Studies B-K: environments, runners, aggregation, theory solvers, results
revision_experiments/            Study L (2026-10-10/, with all 20 checkpoints) and Study M (2026-10-10-selection/)
ACCOUNTING_OPTIONS_2026-10-14/   option-ladder theory checks, historical settlement-component analysis, Studies N-Z, RS, EXT and FT
_archive/exploratory_2026-11-13/  archived development logs, exploratory pilots, and AI-assisted reviews
                                 (preserved outside this git repository for lightweight publication release)
engine/, audit/                  engine audit: simulator source (patched audit copy) and audit code
analysis/, code_audit/           cross-market liquidity test; code audit of 21 electricity-RL repositories
literature/                      pilot literature survey coding (no PDFs)
results/                         engine-audit results
data/                            Energinet data (CC BY 4.0), engine liquidity series, derived tables;
                                 fetch_esett.py downloads the eSett files of all twelve zones (not redistributed)
docs/                            engine-audit protocols, pre-registration and methods
```

## Reproducing
Python 3.10. CPU only.

**Verification without training (minutes).**
```
pip install -r requirements-studies.txt
python -m unittest discover -s tools -p test_verify_studies.py
python tools/verify_studies.py      # registered hashes, counts and completed headline checks
python tools/verify_accounting_revision.py
python tools/verify_reflected_control_revision.py
python tools/verify_release.py      # headline numbers of the engine audit
python tools/ledger_check.py results/ledger_examples/L3_marl_seed7_DK1.csv.gz   # a flagged run
```

The historical B-K manifest includes trailing explanatory notes on three entries.
The verifier checks these entries too. Its `aggregate_c.py` entry refers to the
byte-identical script now shipped at `LIABILITY_STUDY_2026-09-29/scaling/aggregate_c.py`;
the recorded manifest and digest are unchanged.

**Before committing a release.** Review intended additions and deletions first.
The `.gitattributes` exceptions preserve registered and exploratory records byte
for byte. After staging the intended files, run `git add --renormalize .` so
previously tracked files also acquire the current attributes; inspect the staged
diff again before committing. A checkout of the staged snapshot must pass the
verifier above, not just the working copy. Do not blanket-stage deletions that
have not been reviewed.

**Theory and aggregation from the shipped per-run results.** In `LIABILITY_STUDY_2026-09-29/`:

Run commands that regenerate registered results only in a disposable copy. Preserve released scripts, pre-registrations, verdicts and `runs_*.csv` byte for byte; the verifier and the new independent audit read them without rewriting them.
```
python dp_theory.py                 # optimal policies, Denmark (Section 3 of the paper)
python aggregate_b.py               # and aggregate_e.py, aggregate_f.py, aggregate_h.py, aggregate_i.py, aggregate_j.py, aggregate_k.py
python dp_full.py --gammas 0.95 0.98 0.99 0.995   # optimal gambling regions on the full training distribution (all zones)
python make_best_moment_figure.py   # Theorem 1, four zones: results_k/best_moment.csv and the figure
python placebo_market.py --demo     # the placebo-market test on a floor and a full-liability agent
```
Theory checks of Supplementary Section S11 (exploratory; the eSett files and the Study J panels are needed, see below):
```
python best_moment_zones.py         # Theorem 1 in all twelve Nordic zones
python action_gap.py                # action gap of the DP in every training condition; DP policy at gamma 0.99
python false_profit_bound.py        # the cap of Theorem 2 against fixed, threshold, closed-form and learned policies
python partial_liability.py         # the gambling region when a share of every forgiven loss is booked (Corollary 3)
python make_theory_tables.py        # outputs/tables/tab_zones.tex, tab_cap.tex, tab_actiongap.tex
```
Figure and table scripts (`make_*.py`) write to `outputs/` inside each study folder.

**Studies L and M** (see the `README.txt` in each folder):
```
cd revision_experiments/2026-10-10 && python aggregate_continuation.py        # Study L, all 20 agents
python revision_experiments/2026-10-10-selection/summarize_frontier.py          # Study M, from the repository root
python revision_experiments/2026-10-10-selection/verify_frontier.py
```

**Accounting options, Study N and the settlement analysis.** See `ACCOUNTING_OPTIONS_2026-10-14/README.md`. In short:
```
cd ACCOUNTING_OPTIONS_2026-10-14
python verify_theory.py                         # exact checks: ladder, factor two, doubling counterexample, non-identifiability
python study_n/portable.py aggregate_n.py       # Study N verdicts from the shipped per-run results
python study_o/aggregate_o.py                   # Study O verdicts (learnability index); study_o/dp_o.py and calibrate_o.py give the predictions
python study_p/aggregate_p.py                   # completed P verdicts (90 new runs + reused O80); disposable copy only
python study_s/aggregate_s.py                   # Study S verdicts (sign-paired reward); disposable copy only
python study_s/make_studyS_outputs.py           # the Study S figure
python study_t/aggregate_t.py --print           # Study T verdicts (twin-identified reward, ORPO); prints, writes nothing
python study_u/aggregate_u.py --print           # Study U verdicts (settlement lag, terminal liability); prints, writes nothing
python study_v/aggregate_v.py --print           # Study V verdicts (ruin penalty, mean-variance, PMR, audit); prints, writes nothing
python study_v/make_studyV_outputs.py           # the Study V figure and tables
python study_x/aggregate_x.py --print           # Study X verdicts (two-write floor); prints, writes nothing
python study_x/make_studyX_outputs.py --tables DIR --figures DIR   # the Study X figure and tables (default: the sibling manuscript repository)
python study_y/aggregate_y.py --print           # Study Y verdicts (A2C); prints, writes nothing
python study_y/make_studyY_outputs.py --tables DIR --figures DIR   # the Study Y figure and tables (default: the sibling manuscript repository)
python study_z/test_envs_z.py; python study_zb/test_envs_zb.py      # exact checks of the split-account and isolated-margin proposition
python study_z/aggregate_z.py --print; python study_zb/aggregate_zb.py --print   # Study Z per-arm tables (Parts A and B); print, write nothing (the verdicts are in VERDICTS_Z.md and VERDICTS_ZB.md; without --print the scripts rewrite them: disposable copy only)
python study_zb/make_tab_hidden.py --out DIR                        # the main-text table of Study Z (default: the sibling manuscript repository's sections/)
python study_rs/portable_rs.py test_envs_rs.py; python study_rs/portable_rs.py aggregate_rs.py --print   # Study RS: exact checks; registered analysis, printed (writes nothing)
python study_rs/make_studyRS_outputs.py --tables DIR --figures DIR   # the Study RS figure and tables (default: the sibling manuscript repository)
python ../exploratory/exploratory_claude_2026-11-05/rs_reanalysis.py --print; python ../exploratory/exploratory_claude_2026-11-05/bracket_limits.py --print   # exploratory; print, write nothing
python study_ext/aggregate_ext.py --print; python study_ft/aggregate_ft.py --print   # Studies EXT and FT per-arm tables; print, write nothing (the verdicts are in VERDICTS_EXT.md and VERDICTS_FT.md; without --print the scripts rewrite them: disposable copy only)
python study_ext/fetch_data.py                  # the example price file of Gym-Trading-Env, by hash; only needed to re-run Study EXT
python twin_reward/test_reward_diagram.py       # the reward diagram: 88,702 exact checks in rational arithmetic
python study_w/aggregate_w.py --results ../study_v/results_v --demo   # Study W's analysis on Study V's runs (software check; writes nothing)
python twin_reward/test_twin_reward.py          # tests of the TIR/SPR estimators and the Gymnasium wrapper
python make_tables.py                           # the paper's settlement tables from the shipped outputs
```

**Independent exploratory audit (from the archive folder, seconds).**
```powershell
# Run from repository parent / Paper1:
python _archive/exploratory_2026-11-13/exploratory_codex_2026-10-15/statistical_audit.py --seed-fit-bootstrap 199
```
This independently checks the pre-Q cohort, O/P point fits, baselines and seed sensitivity. The completed-Q diagnostic is `_archive/exploratory_2026-11-13/exploratory_codex_2026-10-15/diagnose_completed_q.py`; its separate output covers the 924-agent cohort through Q. Completed R adds 60 agents, bringing the main-study count to 984. Q2 paired-seed exploratory intervals remain positive; exact common-arm permutations give p = 0.025 for each learner under that reference distribution. Training visitation and critic gaps were not serialized and cannot be reconstructed from greedy probe summaries. See `_archive/exploratory_2026-11-13/exploratory_codex_2026-10-15/STATISTICAL_CHANGES.md`.

The DP-free fixed-reference rollout analysis has completed: see `_archive/exploratory_2026-11-13/exploratory_codex_2026-10-15/model_free/EXPLORATORY_RESULTS.md` and its reproduction README. Across 26 existing conditions, the primary surrogate's DP-index rank correlation was 0.950 overall but 0.552 among legacy conditions. It matched O's 58/70 run classifications, with wide Monte Carlo uncertainty; three of seven O arms had intervals crossing the point threshold, and the transferred threshold missed both 4M arms. This demonstrates an affordable fixed-policy diagnostic, not an interchangeable estimate of the optimal gap or a confirmed scalable predictor. All references and failures are reported; that method did not lead to a registered study. The later exact one-step method and separately registered Study R are described below; they do not retrospectively rescue this failed design.

### Exploratory record: exact witnesses and Study R

`_archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/estimator_audit/` contains an explicitly exploratory
exact empirical one-step audit. It integrates every TRAIN magnitude/cap row and
both fair signs, so it needs neither a DP nor Monte Carlo. Its numerical analysis
took about 0.40 seconds. This is exact conditional on the finite empirical model,
not a guarantee about future markets or arbitrary black-box simulators. The
seven-probe predictor and the executable dense-grid witness are distinct.

The geometric calibration cutoff is 0.089669%, within a very wide separating
interval. Historical acquisition transfer remains mixed: O gives 6/7 arm gaps but
46/70 raw-run endpoints, and alpha .9 is misclassified. All failures and source
revisions remain visible in its `EXPLORATORY_RESULTS.md`.

`_archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/state_audit/` extends the feasible-policy question
to Study N's ledger/clock states. Positive one-action witnesses occur at all
24 insolvent lag probes, versus none in the 24 matching booked controls. Terminal
immediate/deadline policies give 69/108 and 78/108 witnesses; their booked controls
give none. These are empirical-model incentive witnesses, not transferred O
thresholds or proofs of learned optimal actions. A nonpositive witness is
inconclusive about multi-step exploitation.

**Study R is complete: all 60 PPO agents finished and the automatic registered
aggregation returned R1 NOT SUPPORTED.** New seeds 60–64, DK1/DK2, fees .40/1.50
EUR/MWh and alpha 0/.8/1 were fixed before training. Each run completed 1,007,616
actual steps for the requested 1M budget. The frozen exact-score convention
predicted acquisition at alpha .8 under the lower fee and non-acquisition under
the higher fee; both retain a positive one-action incentive. All four required
primary gates are reported:

| R1 gate | Full-cohort result | Met? |
|---|---|---|
| Low-fee treatment-control gap >=4 | 4.500 | Yes |
| High-fee treatment-control gap <4 | 4.475 | No |
| Mean paired-seed interaction >=4 | .025 | No |
| Exact one-sided seed-cluster sign p<=.05 | .8125; 2/5 positive | No |

The descriptive paired-seed bootstrap interval for the interaction is
[-4.475,5.125]; it does not replace the sign test. Full-forgiveness anchors chose
16 in 10/10 runs at both fees. At alpha .8 the mean leverages were 7.600/5.600
(4/10 and 3/10 maximal); matched booked controls averaged 3.100/1.125 (1/10 and
0/10 maximal). The raw fee difference therefore does not establish the registered
treatment-control interaction. The operational gap thresholds do not prove
population equivalence or absence of gambling.

The frozen protocol, predictions, full CSV, analysis and verdict remain in
`ACCOUNTING_OPTIONS_2026-10-14/study_r/`; the launch logs are retained. Its 60
agents are included in the completed 984-agent main cohort. This failed
acquisition prediction supplies no superiority claim over simple fee-aware
scores and does not alter the separately confirmed accounting witnesses.
Preserve all registered records and outputs byte for byte; rerun result writers
only in a disposable copy.

**Retraining (hours to days on a multi-core CPU).** Each runner is resume-safe and skips finished jobs.
```
cd GENERALIZATION_STUDY_2026-09-29 && python run_study.py --phase rules && python run_study.py --phase rl --workers 6
cd LIABILITY_STUDY_2026-09-29
python run_b.py --phase rules && python run_b.py --phase rl --workers 6   # Study B (the study relaunched with run_b_order.py; see RUN_LOG.md)
python scaling/run_c.py --workers 7                                       # Study C
python run_e.py --workers 7 ; python run_f.py --workers 7 ; python run_h.py --workers 7 ; python run_i.py --workers 7
python run_k.py --workers 7                                               # Study K (discount factor)
cd ../ACCOUNTING_OPTIONS_2026-10-14 && python study_n/portable.py run_n.py --workers 7   # Study N (about 75 minutes with 7 workers)
```
Study B took about 8 hours, Study I about 2 hours and Study K about 3 hours with 7 workers on an 8-core CPU. Study L: `python revision_experiments/2026-10-10/run_portable.py --workers 4` (preserve the released jobs first; see its README).

**eSett data (Study J and the settlement analysis).** eSett data are not redistributed:
```
python data/fetch_esett.py                                   # from the repository root: twelve zones, 4 March 2025 to 27 September 2026; records content hashes
python ACCOUNTING_OPTIONS_2026-10-14/fetch_esett_pre.py      # the same zones, November 2021 to March 2025
cd LIABILITY_STUDY_2026-09-29
python get_esett_j.py && python build_panel_j.py && python dp_j.py && python run_j.py --workers 7
python ../tools/verify_studies.py                            # reports revised-source panel hashes explicitly
```
eSett revises recent months. `data/esett/CONTENT_SHA256.json` records the hash of each file as first downloaded; a later download of a revised month is reported as differing.

**Study G (gym-mtsim)** needs its own environment, because gym-mtsim's bundled data load only with pandas 2.0.x:
```
cd LIABILITY_STUDY_2026-09-29/mtsim_study
python -m venv .venv && .venv\Scripts\pip install -r ../../requirements-mtsim.txt
.venv\Scripts\python run_g.py --workers 7 && .venv\Scripts\python aggregate_g.py
```

**Engine audit (hours).** The engine source is in `engine/`; its datasets and trained checkpoints (about 57 GB) are not included. `pip install -r requirements-engine.txt`, then see `docs/RUNNING_THE_CAMPAIGN.md`. `audit/setup_audit_engine.py` copies the engine and applies the three-block patch in `audit/engine_patch/`, which bypasses the train/eval contract-hash check only when `SIM_AUDIT_ALLOW_PROTOCOL_MISMATCH=1` and applies config overrides from `SIM_AUDIT_CFG_OVERRIDES`.

## Pre-registration
All pre-registrations are internal: written and hashed before the corresponding test-period evaluation, not deposited with an independent registry. Their records disclose smoke tests on training data, design changes made before registration (for Study N: the fee and claim reporting, after a failed first pilot), a change of job submission order during Study B, a configuration-selection rule for Study I written after its pilot had been seen, aggregation scripts written after the runs for Study C, and, in the engine audit, timestamp corrections and one post-hoc correction of an aggregation script. `tools/verify_studies.py` checks every registered file against its recorded hash. The settlement analysis is exploratory.

Study RS was registered the same way, with one defect: its hasher matched the marker quoted in the run log's introduction, so the registered digest of `RUN_LOG_RS.md` covers its first 147 bytes and not the intended 3,555. The other 16 registered files match, and `tools/verify_studies.py` checks and reports this. It ran next to this repository and was copied in byte-exact, so its manifest and scripts name the original layout: `tools/verify_studies.py` maps the manifest and `study_rs/portable_rs.py` runs the scripts.

## Exploratory record: cost-aware comparators

The exploratory comparison in `_archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/cost_baselines/`
uses thirteen fixed scores, the original ten historical labels and common
balanced-accuracy calibration. It reports all calibration-optimal cutoff sets,
arm-level results separately by learner/budget, the bundled H cost intervention,
six-setting and revised-FI sensitivities, and all failures. Dose-aware simple
scores tie the DP/exact indices' ranking in the two-class O/P/Q cohorts. H already
separates under nominal fee. The completed studies do not establish superior
acquisition prediction by the diagnostic.

```powershell
# Run from repository parent / Paper1:
python _archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/cost_baselines/compare_completed_cost_baselines.py
python _archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/matched_moments/exact_tail_counterexample.py
```

The second command exactly verifies a bounded pair with identical fee, dose,
equity, variance and gross/net absolute-payoff means but opposite one-action
accounting-incentive signs. Full booking removes the positive proxy reward.
This explains a concrete information requirement for the feasible-policy audit;
it is not a result about learner acquisition. No new agents are trained. Neither
script uses Study R outcomes or changes its frozen records. Manuscript table generation is
`tools/generate_cost_baseline_tables.py` in the companion manuscript repository.

## Exploratory record: adaptive witness synthesis

The exploratory audit now extends the isolated-action diagnostic to executable
ledger/clock feedback policies. `_archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/adaptive_witness/`
contains the assumptions, proofs and exact rational counterexamples. A negative
immediate prefix can reach a profitable continuation: at equity .7, cost .4 and
fair gross +/-1, a two-step adaptive policy earns charged .0985 while booking
-.598; isolated action and constant two-step trading both lose. A second example
needs a low-equity gate before insolvency. The terminal-account clock compiler
places each fixed local controller at the deadline and reuses its certificate
across outer horizons by deterministic discount scaling. It does not optimize
all policies or transfer to time-varying markets or information gained while flat.

The accounting identity gives `J_alpha = E[(1-alpha)*F-C]` and
`J_book = -E[C]` under predictable no-edge gross payoffs. Bounded executed costs
give the residual a known lower endpoint even with integrable unbounded prices;
independent upper-capped confirmation can certify a positive expectation or
abstain. Full booking analytically removes positive expected monetary reward
under these assumptions. Search, martingale controls, clipping and concentration
are established tools; the proposed contribution is the accounting-specific
construction and certificate. Residual variance can increase, and failure to
find a witness is not safety or a learner-acquisition prediction.

The fixed-budget empirical families in `adaptive_witness_empirical/` preserve all
failures. The full-clock family confirms 38/70 selected state-policy pairs, but
structured, random, barrier and forced-prefix-myopic methods each confirm 8/14
states; none adds a state beyond the exact isolated-action reference's 9/14.
A separately planned selected-case follow-up treats those outputs as development
data. It freezes one DK1 96-step controller and a search-selected cap before
524,288 fresh confirmation paths. Its one-sided conditional 99% charged lower
bound is **11.09292566 EUR**, with exact expected booked return bounded by the
negative mandatory initial fee, **-3.33553183 EUR** (rounded). Every nonflat
isolated action at that state is negative (best: -.01433268 EUR). This establishes
one selected empirical-model multi-step capability, without revising the 38/70
benchmark or establishing superiority over random or constant multi-step search.

`adaptive_witness_heavytail/` validates the certificate on two constructed Pareto
laws. At alpha .8, infinite-variance shape 1.5 is confirmed with lower mean
.02335552 versus true .04509525; shape 2.5 is inconclusive at that charge, with
true mean -.07454469. The two-law error budget is .05 and covers each fixed
action's whole charge interval, separately from the empirical families. These
are generative mathematical cases, not independent markets or training results.

```powershell
# Run from repository parent / Paper1:
python _archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/adaptive_witness/exact_adaptive_benchmark.py
python _archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/adaptive_witness_empirical/selected_case_followup.py
python _archive/exploratory_2026-11-13/exploratory_novelty_2026-10-15/adaptive_witness_heavytail/bounded_cost_certificate.py
```

The empirical README also lists all three earlier benchmark commands, plans,
selection records, bounds and failures. These method analyses did not use R's
interim endpoints. **Completed R1 is not supported**, and its 60 agents are now
included in the 984-agent main-study count; the policy-specific certificates
answer a separate incentive question. Current build QA is reported in the
companion manuscript README; earlier page and abstract counts do not describe
the latest method revision.

## Not included
- Exploratory research logs, earlier development pilots, and scratch notebooks (11 exploratory folders, ~270 MB) have been moved to `Paper1/_archive/exploratory_2026-11-13/` to keep this publication repository compact and focused strictly on the core reproducible experiments and verification suite.
- Trained checkpoints and raw per-run logs of the other studies (size, or not saved: Studies S to Z and RS saved no policy weights). Per-run summaries are in each completed study's results folder (`jobs/`, `results_*/` and `runs_*.csv`), so the decisions can be recomputed exactly but those policies cannot be replayed without re-training, whose exact reproducibility has not been checked. The 20 Study L checkpoints and the Study R models are included.
- Engine datasets and checkpoints (about 57 GB) and raw engine logs (about 105 GB), and 110 of the 115 step traces of the engine twin check (about 290 MB; `_archive/exploratory_2026-11-13/exploratory_claude_2026-11-05/engine_twin/results/trace_sha256.txt` lists the digests of all 115).
- The 26 scans and traces of the engine's real-clone check except five (`_archive/exploratory_2026-11-13/exploratory_claude_2026-11-06/engine_scan/results/scan_sha256.txt` lists the digests of all).
- The example price file of Gym-Trading-Env (no license stated); `ACCOUNTING_OPTIONS_2026-10-14/study_ext/fetch_data.py` fetches it by hash.
- Raw eSett Open Data and the Study J panels derived from them (no explicit redistribution license); `data/fetch_esett.py` and `ACCOUNTING_OPTIONS_2026-10-14/fetch_esett_pre.py` re-download them.
- Third-party source code from the audits (copyright); the scan scripts regenerate the excerpts from pinned commits.
- The paper PDFs of the literature survey (copyright).
