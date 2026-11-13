# Source and scope addendum to NOVELTY_REVIEW_V1

31 October 2026. This addendum takes precedence over the statements specified below. No study source, result, verdict or manuscript was changed.

## Correction: existing chronological magnitudes do not establish M

The proposal-3 sentence in `NOVELTY_REVIEW_V1.md`, “Current V/T/U magnitudes deliberately satisfy M,” is withdrawn. The current implementation randomizes signs, but retains chronological magnitude and cap arrays. Directly inspected `study_t/envs_t.py`: `market_arrays` returns `abs(m.spread)` and `m.cap_mwh`; reset selects contiguous windows; observations contain reported equity, a sign signal and remaining episode time; `step` uses `abs_s[t]` and clips the action by `cap[t]`. Reported equity depends on previous magnitudes, and executed position/cost can convey current cap information. Random signs do not establish independence of magnitude from the observation, action or state/history. This is a failure to establish the premise, **not a measured numerical violation or a discovered implementation bug**.

The corrected proposal is to test the magnitude-invariance and conditional-bias premises on the existing chronology first, with an observation/history-conditioned diagnostic and transparent exploratory status. A deliberately controlled two-regime toy can then illustrate the failure and a conditional estimator's behavior. It should not be described as the first test outside a previously proved M-valid sleeve. The theory-review report independently addresses the stronger mathematical defects; the novelty review does not certify the current proposition.

## Additional contemporary mitigation prior: MCVL

Evgenii Opryshko, Umangi Jain and Igor Gilitschenski, **Modification-Considering Value Learning for Reward Hacking Mitigation in RL**, Reinforcement Learning Conference 2026 / Reinforcement Learning Journal 7, 2026. [Official conference schedule](https://rl-conference.cc/2026/paper_schedule.html), [author page](https://ktolnos.github.io/mcvl/), [primary full text, arXiv:2606.28955v1](https://arxiv.org/html/2606.28955v1). Pagination and a journal DOI were not verified and are not invented.

Read Algorithm 1, Section 3, Proposition 3.2, Sections 4.3–4.4/5 and Appendix D. MCVL filters off-policy DDQN/TD3 transitions by forecasting equally trained include/exclude branches and scoring them with a frozen reward/value evaluator. It needs non-hacking seed data, resets after rejection, and its theorem assumes evaluator accuracy. Main scoring uses true dynamics; a learned-model ablation is limited to Box Moving. It also compares occupancy regularization and discusses suppression of legitimate improvement.

**Verdict:** keeping legitimate improvement while reference regularization fails is an established mitigation theme. MCVL does not duplicate the accounting sign-pair identity. **Reviewer recommendation:** discuss this current alternative; do not claim it is an immediately faithful PPO baseline or retrofit it into a laptop-scale comparison without a protocol defining safe seed data, scoring access and compute. None was run here.

## Wang et al. 2025 metadata and fuller source check

The complete publication is Han Wang, Yang Xu, Wenbin Lu and Rui Song, **Off-Policy Evaluation under Nonignorable Missing Data**, ICML 2025, PMLR **267:65020–65058**. [Official publication](https://proceedings.mlr.press/v267/wang25dt.html), [primary full text](https://arxiv.org/html/2507.06961v1).

The OpenReview route in the first memo later presented a browser challenge; the arXiv text was read instead. Sections 4.1–4.3 and Assumption A.2 were checked: monotone dropout, inverse-probability weighting, a shadow variable and positive observation probability. This supports the existing table's stated distinction. It is not an estimator for arbitrary unknown hard-floor rewards without identification assumptions.

Two remaining metadata checks were completed after V1. The [publisher version of Armstrong, Brigo and Tse](https://link.springer.com/article/10.1007/s10479-023-05295-5) confirms **Annals of Operations Research 336:861–898 (2024)**, with online publication 4 April 2023. The [author-hosted ACM version of Karimi Madahi et al.](https://ugentai4e.github.io/papers/2024/karimi2024eenergy.pdf), first-page reference block, confirms DOI **[10.1145/3632775.3661948](https://doi.org/10.1145/3632775.3661948)**, ACM e-Energy 2024, 11 pages. V1's statement that this latter DOI was not recovered is now superseded.

## Existing-data ruin selection

Root independently reports a completed descriptive TIR-only selector on existing V results: it selects RUIN5 in both A/C, including odd-to-even episode splitting, and matches booked-return ordering. This is root's computation, not independently recomputed in this novelty task. The result would directly undermine “only booked returns can identify the working strength,” while adding no new reward-learning principle. Reuse of the already inspected test episodes remains exploratory; the observation should not be sold as a fresh confirmatory selection guarantee.

## Final novelty judgment unchanged

The possible contribution is a specialized reward-identification and engineering-verification method, rather than a new generic RL principle. Correct theory and a concise, useful native-engine demonstration would add more submission value than another sweep on the same construction. The first action is to report/correct the meaning-changing theory claims under the user's review rules; the most valuable subsequent empirical addition is the verified ledger-withheld engine audit in proposal 2, conditional on a valid observation contract.
