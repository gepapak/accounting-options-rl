# Study T pre-registration: can the booked reward be recovered from a censored reward channel with sign twins, and how does that compare with occupancy-measure regularization?

Written 31 October 2026, before any full training run of this study. Internal pre-registration: the files listed in
`PREREGISTRATION_HASHES_T.txt` are hashed (SHA-256, UTC time) before launch. No public registry is claimed. Every run and every
hypothesis is reported, whatever the result.

## Question and what precedes it
Studies B-S show that RL traders learn to gamble when a simulator's accounting forgives losses, and Study S found that a sign-paired
reward (SPR, `(r - r_twin)/2 - own cost`) removed the exploit in an edge market without reading the ledger. An independent review
(`Paper1/CODEX_REVIEW_2.md`) then showed that SPR can attenuate a legitimate edge when credits and an edge coexist, and that Study S compared
SPR only with a per-step action-KL penalty, not with the occupancy-measure regularization of Laidlaw, Singhal and Dragan (ICLR 2025).

This study tests (i) whether Study S's edge-market result **replicates on a new market draw and new seeds**, (ii) a new estimator that uses
the censoring structure of accounting credits, the **twin-identified reward (TIR)**, (iii) a faithful **occupancy-measure (ORPO) baseline**
with a grid of five regularization strengths, and (iv) whether the results hold in **two further market settings**.

**Theory (exploratory, `exploratory_claude_2026-10-31/identification/THEORY_TIR.md`).** Under every rule of the accounting ladder the
reported step reward is a left-censored observation of the booked P&L, `r = max(b, tau)` with a hidden threshold `tau`. With a sign twin
`(r, r')`: `r != r'` exactly when the favourable branch is uncensored, and then the gross magnitude is `g = max(r, r') + c` and the booked P&L
is recovered exactly; `r = r'` means both branches are censored and only `g in [0, r + c]` is known. SPR is the sharp lower end of that
interval. TIR uses `max(r, r') + c` where the pair identifies `g` and the conservative value `-c` otherwise; its pathwise error never exceeds
SPR's and it is unbiased under sign fairness. On random-action rollouts the mean absolute error on credited steps (percent of the allocation per
quarter) is 0.96 for the reported reward, 0.48 for SPR and 0.008 for TIR under the zero floor. These are exact-arithmetic and numerical
checks, not tests of learning. A consequence worth stating before the data: under the zero floor TIR is almost the booked reward, so its edge
can be expected to be close to ORACLE's; the question the study answers is whether a learner trained on it also avoids the exploit.

**Development (pilot seeds 90-92, `exploratory_claude_2026-10-31/study_t_development/`; not part of this study).** The pilots (a) checked that
the ORPO baseline runs and behaves sensibly, (b) chose the additional settings, and (c) fixed the ORPO coefficient grid. Findings that
shaped the design: ORPO at coefficients 10, 1 and 0.1 returned a policy at the reference policy's level (booked about +6% to +7% of K0 per
episode in `S_DK1`, against about +16% for a twin-maximum reward in the same pilot), and at 0.01 one of two seeds acquired the exploit; the
registered grid is therefore {1, 0.1, 0.05, 0.02, 0.01} to cover the transition. A first weaker-edge setting (DK1, p_edge 0.55) was dropped
because its ORACLE edge was only +1% to +2% of K0 per episode, too small to test edge retention; it was replaced by DK1 with p_edge 0.58
(`T_DK1M`); a 4-run pilot of `T_DK1M` (PLAIN and ORACLE, seeds 90-91) showed the exploit and a positive edge there (booked edge of PLAIN -6.4%
and -0.1% of K0 per episode, ORACLE +10.2% and +13.8%). A first pilot launch was killed by a console-close event and rerun.

## Design (`run_t.py`)
- **Markets.** The Study S edge-market sleeve (`envs_t.py`): real Danish spread magnitudes, an independent fair sign, a signal `z` observed
  before acting with `P(z = sign) = p_edge`, the Study B sleeve (fixed sizing from K0, liquidity cap, nine signed leverage levels), costs
  `0.10|q| + kappa q^2` EUR, the zero floor, training equity log-uniform in [0.05, 2] allocations, one-week episodes. **New market draws**
  (Study S used seeds 7101-7104; Study T uses 8101-8104). Settings: `S_DK1` (DK1, p_edge 0.60, kappa 4; Study S's setting), `T_DK2` (DK2, 0.60,
  6) and `T_DK1M` (DK1, 0.58, 4; a moderately weaker edge).
- **Learner.** PPO: 8 environments, n_steps 1024, batch 256, gamma 0.99, 64x64, reward normalization, 10^6 steps.
- **Arms.** PLAIN (reported reward), ORACLE (booked reward; needs the ledger), SPR, TIR and, in Part A only, ORPO with the sqrt-chi^2
  divergence at scale-independent coefficients c in {1, 0.1, 0.05, 0.02, 0.01} (`OM1`, `OM0.1`, `OM0.05`, `OM0.02`, `OM0.01`). ORPO follows
  the main-text formulation of Laidlaw et al. and the semantics of the authors' public code (`orpo_t.py` documents every choice and difference):
  a discriminator trained each iteration on fresh rollouts of the current and the reference policy, penalty
  `-c * sigma_ref * chi2 / sqrt(trimmed mean of chi2)`. The reference policy is hand-specified and safe: leverage 1 in the direction of the
  signal with probability 0.7, otherwise uniform over the other eight levels. It is the same for every setting and does not read the ledger.
- **Part A** (`S_DK1`, seeds 120-129): PLAIN, ORACLE, SPR, TIR, OM1, OM0.1, OM0.05, OM0.02, OM0.01: 90 runs.
- **Part B** (`T_DK2` and `T_DK1M`, seeds 130-139): PLAIN, ORACLE, SPR, TIR: 80 runs.
- **Measures per run.** `X` = share of the 16 reachable low-equity probes (equity in {0, 0.02, 0.05, 0.1} allocations, both signals, share of
  the episode remaining 0.05 and 0.5) at maximal leverage (greedy policy); `E` = booked P&L in the edge market on the test split, percent of K0
  per one-week episode, mean of 100 episodes. The 100 test windows and start equities are the same for every run (fixed evaluation seeds), so
  differences in `E` between runs reflect the policies, not the sample of test episodes.

## Hypotheses and decision rules (`aggregate_t.py`)
In Part A each rule is evaluated for the setting `S_DK1`; in Part B separately for `T_DK2` and `T_DK1M`. "Edge ratio" is the mean `E` of an
arm over the mean `E` of ORACLE, with a one-sided 95% bootstrap lower bound (10,000 resamples of runs within each arm, generator seed
20261031).
- **x1 (the exploit is present).** `X(PLAIN) > X(ORACLE)`, one-sided Mann-Whitney `p < 0.05`.
- **x2 (SPR removes it; replicates SA2).** `X(SPR) < X(PLAIN)` with `p < 0.05` and `mean X(SPR) <= mean X(ORACLE) + 0.10`.
- **x3 (SPR keeps the edge; replicates SA3).** Edge ratio of SPR `>= 0.80` with lower bound `>= 0.60`; requires `mean E(ORACLE) > 0`.
- **x4 (TIR removes it).** As x2 for TIR.
- **x5 (TIR keeps the edge).** Edge ratio of TIR `>= 0.90` with lower bound `>= 0.75`.
- **TA6 (occupancy regularization, Part A only).** No tested ORPO coefficient has both `mean X <= mean X(ORACLE) + 0.10` and
  `mean E >= 0.80 * mean E(ORACLE)`. Reported whatever the outcome: if one does, that is a result. Not testable if `mean E(ORACLE) <= 0`.
If x1 fails in a setting, x2-x5 for that setting are reported as not testable (the exploit was not acquired). The Part A rules are named
TA1-TA6 and the Part B rules TB-`<setting>`-1 to TB-`<setting>`-5. Sixteen hypotheses, each with its own decision; no multiplicity
correction. A missing or failed run is reported and not replaced.

## Exploratory outputs (not registered)
Placebo-market reported and forgiven returns per arm; the exact identification study on random-action rollouts (identification directory);
discriminator diagnostics of the ORPO arms; the ORPO arms' booked edge relative to the reference policy's.

## Software validation, integrity, completion and failures
`run_t.py --smoke` (20,000 steps, one seed per arm and setting) checked that every arm runs end to end; smoke results are excluded. The greedy
NumPy forward pass is checked against `model.predict` for every run. Files hashed before launch: this file, `RUN_LOG_T.md`, `envs_t.py`,
`spr_t.py`, `orpo_t.py`, `run_t.py`, `aggregate_t.py`. The run log is append-only after "## After hashing".
