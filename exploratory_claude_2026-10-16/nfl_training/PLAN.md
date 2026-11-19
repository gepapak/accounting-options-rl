# No-free-lunch (NFL) training: exploratory pilot

**Status: EXPLORATORY.** Written 2026-10-16 before any run. Nothing here is a registered study. If the pilot passes its kill
criteria, a confirmatory study is registered separately, with new seeds and hashes, before it runs.

## Idea
A correct trading simulator admits no free lunch: on a fair-game market (exogenous price changes with an independent random
sign), every policy's expected booked P&L plus its costs is exactly zero. Any positive expected reward there comes from the
simulator's accounting (a floor, a lag, a cap...), not from trading. NFL training adds placebo episodes to the training batch and
rewards them with the *negative* of (reported reward + own trading cost):

    J_NFL(pi) = E_real[r] - E_placebo[r + c]
              = E_real[booked] + (E_real[leak] - E_placebo[leak])        because E_placebo[booked + c] = 0 for every pi

so the accounting leak cancels while the legitimate edge, which the placebo market does not have, passes through. The trainer does not
need to know which accounting rule leaks; it needs (i) a placebo copy of the exogenous price series (sign-randomized) and (ii) the
agent's own trading cost, which every simulator logs.

Implementation: a mixed vector environment (4 real, 4 placebo); in placebo environments the reward is -(r + c). The policy cannot tell
the two apart from its observation (equity, signal), whose distributions overlap.

## Testbed (new, built on the Study B/O sleeve)
- Real Danish spread magnitudes |s_t| (DK1/DK2, Study A panels), independent fair sign eps_t.
- **Edge market:** a signal z_t with P(z_t = eps_t) = 0.6, observed before acting. **Placebo market:** z_t independent of eps_t.
- Costs: fee 0.10 EUR/MWh * |q| + kappa * q^2 with kappa = 8 EUR/MWh^2, so the legitimate optimum is leverage 4 (q = 1 MWh) in the
  direction of z at every equity (edge 0.2 * E|s| ~ 16-18 EUR/MWh). Leverage 16 loses money on the ledger.
- Accounting: zero floor (ZF) as in Studies B/O: reported equity max(e + pnl, 0), trading continues.
- Observation: (reported equity / K0 clipped to [0, 5], z). Actions: the nine signed leverages of Study B.
- PPO as in Studies F/O: 8 envs, n_steps 1024, batch 256, gamma 0.99, [64, 64], reward normalization, 1M steps.

## Pilot arms (DK1, seeds 80, 81, 82)
- PLAIN: reward = reported (ZF) change, 8 real envs.
- NFL: 4 real envs (reported reward) + 4 placebo envs (reward -(r + c)).
- ORACLE: reward = booked pnl (every loss charged, the alpha = 1 control), 8 real envs. Needs the ledger; upper benchmark.

## Outcomes
- Probe leverage (greedy) at equity {0, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2} x z in {-1, +1}.
- Test-split booked and reported return (% of K0) of the greedy policy, edge market, 40 episodes with training-like start equity.

## Kill criteria (fixed now)
1. Testbed shows the exploit: PLAIN has |L(0.02)| = 16 or |L(0)| = 16 in >= 2/3 runs.
2. Testbed has a learnable edge: ORACLE's mean test booked return > 0 in >= 2/3 runs.
3. NFL removes the exploit: |L(0.02)| < 16 and |L(0)| < 16 in >= 2/3 NFL runs.
4. NFL keeps the edge: mean NFL test booked return >= 50% of ORACLE's (seed means).

If 1 or 2 fails, the testbed is redesigned once and the failure is recorded.

## Pilot v1 result (19:45, 16 Oct 2026): criterion 1 FAILED, testbed redesigned once
- PLAIN learned the legitimate policy (leverage 4 following z) at every probe, including equity 0; PLAIN and ORACLE chose
  identical leverages. With kappa = 8 the floor makes leverage 16 attractive only at equity ~0, which the edge market rarely visits.
- NFL (episode-level) kept the legitimate direction but halved the edge (booked +7.1% vs ORACLE +14.1% of K0 per episode, seed means;
  leverage 1 or 0 instead of 4). Cause: occupancy mismatch (Proposition 3). Placebo episodes have no drift, hit the floor far more
  often (ORACLE's policy: 12% of K0 forgiven per placebo episode vs ~0.5% in the edge market), so honest leverage is over-penalized.
- Also: v1 placebo signs were one fixed draw for the training period (spurious realized edge). Fixed in envs_nfl2.py (fresh per episode).

## Pilot v2 design (fixed before running)
- Testbed: P_EDGE 0.6, KAPPA 4 (envs_nfl2.py). One-step economics (DK1): booked per step lev 1 +3.6, lev 4 +11.4, lev 16 +4.4 EUR;
  zero-floor reported advantage of lev 16 over lev 4 at equity 0 / 0.02 / 0.05 / 0.1 / 0.25 K0: +100 / +36 / +19 / +9 / +0.4 EUR.
- New arm SPR (sign-paired reward): the simulator is cloned before each step and the clone is stepped with the opposite sign of the
  same exogenous price change; the training reward is (r - r_twin) / 2 - own cost. In expectation this is the booked P&L at every state in
  a fair market (no occupancy mismatch); with an edge it is conservative by (1 - 2p) * forgiven / 2. Needs clone + sign override + own
  cost, not the ledger or the rule.
- Arms: PLAIN, ORACLE, NFL (episode-level, fresh placebo signs), SPR. ZF, DK1, seeds 80-82.
- Kill criteria v2: (1) PLAIN has |L| = 16 at some probe with equity in {0, 0.02, 0.05, 0.1} in >= 2/3 runs; (2) ORACLE booked > 0 in
  >= 2/3; (3) a method passes only if it has no |L| = 16 at those probes in >= 2/3 runs AND (4) its mean edge-market booked return is
  >= 80% of ORACLE's (raised from 50%: half the edge is not a useful mitigation). If 3 or 4 fails, NFL is killed and reported as a failure.
If all pass: next steps are (a) other leak types (terminal limited liability, settlement lag, short-loss cap) with the same trainer,
(b) established black-box mitigations as baselines (KL/occupancy regularization to a safe reference, early stopping),
(c) A2C/DQN, then (d) a registered confirmatory study.

## Pilot v2 result (20:23, 16 Oct 2026), ZF, DK1, seeds 80-82
| Arm | edge booked % K0 (seed mean) | max16 at low-equity probes | placebo forgiven % K0 |
|---|---|---|---|
| PLAIN | 9.30 (18.4, -6.5, 16.0) | 0.42 / 0.50 / 0.58 | 35.0 |
| ORACLE | 19.17 | 0 / 0 / 0 | 4.2 |
| SPR | 20.82 (109% of ORACLE) | 0 / 0 / 0 | 4.3 |
| NFL (episode) | 14.95 (78% of ORACLE) | 0 / 0 / 0 | 3.4 |
Criteria: (1) PASS 3/3, (2) PASS 3/3. SPR: (3) PASS 3/3, (4) PASS. NFL: (3) PASS, (4) FAIL (78% < 80%), as the occupancy mismatch
predicts; NFL is kept only as an ablation. Two SPR seeds ended with ORACLE's exact greedy policy: SPR's training reward equals the
booked P&L whenever no forgiveness occurs (Proposition 4), so the training paths coincide; SPR runs took longer (twin steps) and s81 differs.

## Stage 2 design (fixed before running)
Rules ZF (done above), LL, TERM, LAG; arms PLAIN, ORACLE, SPR; DK1; seeds 80-82. Baseline KL to the safe reference (follow z at
leverage 1, p 0.7) with beta in {0.003, 0.03, 0.3} on ZF and TERM. Per rule, an exploit is present if PLAIN's placebo forgiven exceeds
1.5x ORACLE's (seed means) or PLAIN has max16_low > 0 in >= 2/3 seeds. Where present, a method passes if (a) max16_low = 0 in >= 2/3
seeds, (b) placebo forgiven <= 1.5x ORACLE's + 1 pp, and (c) edge booked >= 80% of ORACLE's. KL passes only if one beta meets (a)-(c).
Next if stage 2 passes: DK2, A2C, more seeds, then a registered confirmatory study with new seeds.

## Stage 2 result (23:31, 16 Oct 2026), DK1, seeds 80-82, 45 runs, 0 failures
| Rule | Exploit present? (PLAIN vs ORACLE placebo forgiven; PLAIN seeds with max16_low > 0) | SPR booked / ORACLE | SPR max16_low > 0 | Verdict |
|---|---|---|---|---|
| ZF | yes: 35.0 vs 4.2; 3/3 | 20.8 / 19.2 (109%) | 0/3 | SPR PASS |
| LAG | marginal: 1.94 vs 1.11 (1.75x, rule met); 1/3; PLAIN booked 22.7 >= ORACLE 20.6 | 22.9 / 20.6 (111%) | 0/3 | SPR PASS (weak test) |
| LL | no: 0.42 vs 0.52; 0/3 | 22.6 / 19.8 | 1/3 (0.04) | no exploit to remove; SPR costs nothing |
| TERM | no: 2.34 vs 2.94; 1/3 | 22.6 / 20.3 | 0/3 | no exploit to remove; SPR costs nothing |
KL to the safe reference, ZF (where the exploit is present): beta 0.003 keeps the exploit (max16_low 1.0, 1.0, 0.04; forgiven 69%;
booked 3.4%); beta 0.03 and 0.3 collapse to the reference policy (leverage 1 following z): booked 8.0% = 42% of ORACLE (fails c);
beta 0.03 still has max16_low > 0 in 3/3. No beta passes. On TERM, beta >= 0.03 again costs the edge (8.0 vs 20.3).
Reading: with an edge present, PPO learned the exploit robustly only under the zero floor (top of the ladder); limited liability
closes the account and TERM/LAG options sit in states the edge market rarely visits from these starts. Generality of SPR across
rules is therefore NOT yet shown. SPR never cost edge where no exploit existed (booked >= ORACLE under all four rules).

## Stage 3 design (fixed before running)
(a) Generality: TERM and LAG with insolvent training starts as in Study N: with probability 0.3 an episode starts at equity uniform in
[-0.5, 0) K0 (LAG clock starts at reset), otherwise log-uniform [0.05, 2]. Evaluation uses the same start distribution. Arms PLAIN,
ORACLE, SPR; DK1; seeds 80-82. Same per-rule criteria as stage 2.
(b) Replication of the ZF result: DK2 (PPO) and A2C (n_steps 64, gae_lambda 0.95, normalized advantages, as Study Q) on DK1;
arms PLAIN, ORACLE, SPR; seeds 80-82. Pass: per replication, the stage-2 criteria (a)-(c).

## Stage 3 partial findings (01:10, 17 Oct 2026)
- A2C (DK1, ZF): PLAIN learned no exploit (max16_low 0 in 3/3, PLAIN = SPR, close to ORACLE), so there is nothing to remove; uninformative.
- DK2 (PPO, ZF): DESIGN FLAW, my error. With KAPPA 4 the honest optimum in DK2 is leverage 16 (booked per step following z: lev 4 +12.6,
  lev 16 +13.7 EUR; the liquidity cap binds in 48% of DK2 quarters), so max16 probes do not separate exploit from honest trading in
  DK2 (ORACLE itself has max16_low 0.5). The pre-written probe criterion is invalid for DK2; only the booked edge is interpretable there.
- TERM with insolvent starts: so far PLAIN ~ ORACLE (no learned exploit).
Conclusion so far: in the edge testbed the exploit is learned robustly only by PPO under ZF in DK1.

## Stage 4 design (fixed before running): SPR in the paper's registered fair-market environment
Study O/Q environment exactly (study_o/envs_o.PartialEnv with alpha = 0, placebo NE market, equity-only input, 8 envs, reward
normalization, gamma 0.99, 1M steps; PPO as Study O, A2C-n64 and DQN-default as Study Q), reward replaced by SPR (clone, twin step
with the sign of this quarter's spread reversed, (r - r_twin)/2 - own fee). Seeds 85-89, DK1 and DK2: 30 runs.
Comparators are the registered results (same environment and budget, other seeds): alpha 0 (reported reward) and alpha 1 (every loss
charged = the ledger) from Study O (PPO, 80 runs) and Study Q (A2C, DQN).
By Proposition 2 SPR is exact here (fair market). Prediction: SPR's runs at leverage 16 at equity 0.02 are close to the alpha-1 control.
Pass per learner: SPR runs at 16 <= control runs at 16 + 2 (of 10), and <= 3/10 in absolute terms, while alpha 0 had >= 9/10.

## Stage 3 result (02:31, 17 Oct 2026)
- (a) TERM and LAG with 30% insolvent starts: no learned exploit. PLAIN ~ ORACLE (TERM booked 21.9 vs 23.2, placebo forgiven 18.4 vs 19.5;
  LAG booked 15.5 vs 17.8, forgiven 12.9 vs 12.7; max16_low <= 0.06). SPR = ORACLE within noise (TERM 23.2, LAG 17.9). Generality of
  SPR across ladder rules therefore remains untested: in this edge testbed PPO does not acquire the TERM/LAG options.
- (b) DK2: invalid by design (lev 16 honest-optimal); booked edge PLAIN 17.6 (one seed -8.5), ORACLE 21.5, SPR (2 of 3 seeds) 14.3;
  not interpretable as a test.
- (c) A2C: no learned exploit; SPR = PLAIN.
Bottom line of stages 1-3: SPR removes the exploit and keeps the edge in the one configuration where PPO learns the exploit with an edge
present (ZF, DK1), where KL at every beta fails; it never costs edge elsewhere. The decisive multi-learner test is stage 4.

## Stage 4 result (04:39, 17 Oct 2026): 30/30 runs, pre-written rule scored as written
Runs at leverage 16 at equity 0.02 (of 10): PPO alpha0 10, alpha1 0, SPR 3 -> FAIL (3 > 0 + 2); A2C alpha0 10, alpha1 1, SPR 3 -> PASS;
DQN alpha0 9, alpha1 5, SPR 8 -> FAIL. Exploit shape, mean L(0.02) - L(2): PPO alpha0 +10.6, SPR +1.8, registered alpha1 +0.4.
Caveat found when reading profiles: in this fair market every leverage has the same expected booked P&L within the fee (0.4 EUR vs
+-300 EUR noise per step), so ledger-trained policies settle on arbitrary constant leverages; SPR's leverage-16 runs are mostly flat-high
(e.g. PPO DK2 s86 16 at every equity); 2/10 PPO SPR runs are exploit-shaped (DK1 s87, s88).

## Post-hoc diagnostic (EXPLORATORY, added after seeing stage 4): same-seed ledger controls
The registered alpha-1 controls used other seeds (30-34, 50-54). SPR's reward differs from the ledger reward only at forgiveness events,
so same-seed training should nearly coincide. Run alpha = 1 (reward = booked pnl) with seeds 85-89, DK1/DK2, PPO/A2C/DQN (30 runs). This
does not change the stage-4 verdict above; it only separates seed variance from a residual SPR effect.

## Post-hoc diagnostic result (06:47, 17 Oct 2026): same-seed ledger controls, 30/30
At leverage 16 at equity 0.02 (of 10) and exploit shape mean L(0.02) - L(2):
| Learner | ZF reward (registered) | SPR | LEDGER same seeds | shape ZF / SPR / LEDGER | SPR vs LEDGER (Wilcoxon) | ZF > SPR (Mann-Whitney) |
|---|---|---|---|---|---|---|
| PPO | 10 | 3 | 5 | +10.6 / +1.8 / +3.1 | p 0.84 | p 0.016 |
| A2C | 10 | 3 | 1 | +8.7 / -1.1 / +0.3 | p 0.62 | p 0.026 |
| DQN | 9 | 8 | 4 | +5.4 / +4.6 / +0.7 | p 0.38 | p 0.42 |
Reading: for PPO and A2C, SPR is indistinguishable from same-seed ledger training and removes the exploit shape; for DQN it does not.
The registered PPO ledger control (seeds 30-34) had 0/10 at 16; the same configuration with seeds 85-89 has 5/10: the leverage-16 count
of fair-market controls is seed-dependent (indifference among actions whose expected booked P&L differs only by the fee), so near-threshold
counts in Studies O and R are within this noise; the exploit shape is the more robust outcome. All of this is exploratory and post hoc;
a registered confirmatory study is needed before any paper claim.

## Correction (07:05, 17 Oct 2026): max16_low included unreachable probes
analyze_nfl.py's max16_low counted probes at negative equity, which the zero floor never reaches (reported equity >= 0); under ZF
those are extrapolations. Restricted to reachable probes (equity 0-0.1 K0, both z, rem 0.05 and 0.5): PLAIN s80/s81/s82 have
maximal leverage at 6/16/16 of their nonnegative low-equity probes (s80 only near the episode end, rem 0.05), ORACLE 0, SPR 0,
KL0.003 32/32/0. The pilot conclusions are unchanged; Study S uses reachable probes only.
