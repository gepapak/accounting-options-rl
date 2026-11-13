# Public second-domain feasibility: source semantics first

Focused primary-source research during this review; no package, environment,
model, or training execution. No source/manuscript edits. The search verified
concrete accounting/reward implementations rather than treating any paper
mentioning bankruptcy or censored demand as a direct second-domain test.

**Conclusion:** a genuine public monetary-floor simulator exists: gym-mtsim.
It is already in Study G, however, and is not an off-the-shelf direct test of
the current twin theorem in its full generality. **No suitable off-the-shelf
direct test in a genuinely new nontrading domain was verified.** This is the
outcome of a bounded search, not proof that none exists.

## Verified candidates and boundaries

| Candidate | Verified original reward/accounting | Direct fit to C and sign-twin theorem? | Practical verdict |
|---|---|---|---|
| gym-mtsim, AminHP, public source | `tick` marks orders, performs margin stop-outs, then replaces negative balance by zero; `MtEnv` rewards the change in equity. | Plausible in a restricted single-instrument, same-profit-currency, known-fee case. No native sign-override interface; arbitrary currency conversions, portfolios and action-dependent liquidation are not already proved to give `sigma*g-C` with the same C in both branches. | Best existing public accounting candidate, but already Study G; not a new domain discovery or a ready prevention demonstration. |
| Gym-Trading-Env, Clement Perroud, public source | Default reward is log portfolio-valuation ratio. At valuation <=0 the episode terminates and leaves the step reward at zero by skipping the reward callback. | No. In general this is a log objective with a terminal reward omission, not `max(sigma*g-C,tau)` with a common lower threshold. | Genuine public reward-channel issue, already disclosed by the manuscript's environment audit. Needs a new model/estimator or an explicitly changed reward; cannot claim TIR applies unchanged. |
| FinRL standard stock environment | Reward is scaled change of total asset value. | No monetary forgiveness established by that reward formula. It exposes the account components and primarily uses long-only affordable positions at defaults. | A useful comparator, not a verified naturally censored booked-reward domain. Known transaction/terminal issues do not establish a new application of this theorem. |
| OR-Gym inventory/lost sales | Sales are `min(available_inventory,demand)`, but reward is period profit after replenishment, holding and unmet-demand costs. | No. Demand is only partially satisfied/observed; the simulator is not clipping the true period monetary loss from below. The primitive shock is nonnegative demand, not an equal-cost signed linear payoff. | Do not call censored-demand inventory a direct validation of the accounting-credit method. |

Primary sources for the table:

* gym-mtsim [simulator source, `tick`, `close_order`, `_update_order_profit`,
  `_get_unit_ratio`](https://raw.githubusercontent.com/AminHP/gym-mtsim/main/gym_mtsim/simulator/mt_simulator.py)
  and [environment source, `reset`, `_apply_action`, `_calculate_reward`](https://raw.githubusercontent.com/AminHP/gym-mtsim/main/gym_mtsim/envs/mt_env.py).
  The former also shows that profit currency conversion can depend on exit
  price, so linearity cannot be assumed for all instruments. The latter
  deep-copies the baseline on reset and accepts a callable fee; cloning during
  a live step and preserving the same fee realization still need an adapter.
* Gym-Trading-Env [pinned source at the manuscript audit's commit
  a86fc03d29759374fae2db7df4d878485808a730](https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/a86fc03d29759374fae2db7df4d878485808a730/src/gym_trading_env/environments.py),
  `basic_reward_function` and `TradingEnv.step`.
* FinRL [standard environment source](https://raw.githubusercontent.com/AI4Finance-Foundation/FinRL/master/finrl/meta/env_stock_trading/env_stocktrading.py)
  and the project contest's [reward and execution documentation](https://finrl-contest.readthedocs.io/en/latest/envs/stock/finrlmeta.html).
* OR-Gym [inventory environment source](https://raw.githubusercontent.com/hubbs5/or-gym/master/or_gym/envs/supply_chain/inventory_management.py),
  `_STEP`, specifically sales, profit, and the return of profit as reward.

The gym-mtsim current main source was retrieved and inspected. Its older audit
pin is recorded locally in `LIABILITY_STUDY_2026-09-29/env_audit/pins.json`, but
retrieving that pinned raw URL through the browser returned a cache miss; I do
not claim independently to have reverified its remote bytes at that pin in this
research pass. Any new execution must pin and check the actual source it uses.

## Why gym-mtsim is promising but not a quick successful second study

The public simulator already supplies an account, leverage, margin, stop-outs,
fees and an actual zero-balance write. One can investigate a restricted EURUSD
account denominated in USD with a single position and a fixed known fee:
before a floor or closeout, P&L is potentially linear in a signed price
increment. A proposed clone could reflect only the next increment about the
current price while retaining the same action, fees, fills and random state.
This is a **feasibility inference**, not an implemented or proved adapter.

The restrictions and required checks matter. Both reflected prices must remain
valid positive prices; the two branches must have the same executed exposure
and own cost; reward must be the same booked step plus a nonnegative credit
under one threshold; conversion rates and unrelated positions cannot silently
change. Multi-order partial liquidation or a global change of a price table
does not automatically satisfy those requirements. Conditional sign fairness
is an additional market construction/assumption, not a consequence of cloning.

Study G already trained 20 PPO agents in this package. Its preserved
`mtsim_study/results_g/VERDICTS_G.md` gives 0/3 supported hypotheses, and the
current main text `sections/boundary_costs.tex:6` states that all agents learned
not to open positions. Repeating that study with another reward could produce
three flat policies and no test of prevention. Do not omit this negative prior
when proposing it. The existing `mtsim_study/run_g.py:44-64` evaluates final
equity separately in ORIG and the full-liability patch; the unused wrapper
`ledger` field must not be mistaken for its booked evaluation endpoint.

Lending or insurance limited liability also needs an economic target check.
An owner's intentionally limited-liability payoff is a legitimate nonlinear
objective; removing its symmetric option value changes the intended task.
I did not verify a public lender/insurance environment whose *mistakenly*
floored booked loss and access assumptions make this a faithful direct test.
Finding an insurance simulator or a positive-part payoff alone would not do so.

## Ranked next step

1. **Connect the existing method to the manuscript's real engine, beginning
   with a source-semantics gate and frozen-policy audit.** This is the most
   valuable bridge because the paper already names it as missing
   (`discussion.tex:29`). Before any new training, prove whether the returned
   reward at the chosen event fits C with a common sign-invariant own cost.
   Delayed obligations, multiple cash components, fills and the order of the
   cash floor make that a substantive check. Keep the engine's ledger hidden
   from the estimator; use it only as external evaluation ground truth. Test
   original dynamics/cost parity, then reconstruction error on the original
   strict protocol and the existing documented fault profile. The strict
   profile's floor does not bind, so zero error there alone is not evidence of
   useful censoring recovery. If the semantic gate fails, report that boundary;
   do not rewrite the engine to manufacture a fit and call it native transfer.
   No clone theorem is yet verified for this engine in this focused note.
2. **Restricted gym-mtsim semantics/audit**, if public replication is the
   priority. Start with deterministic hand states and original-source parity,
   then frozen policies. Only consider learned prevention after showing the
   original reward actually causes acquisition in that stated task. Expected
   engineering value is meaningful; risk of a prevention null is high given
   Study G. This is another financial simulator, not cross-sector validation.
3. **Do not add a newly named censored toy as a claimed second domain.** A
   small state-dependent-magnitude falsification test can still expose PMR's
   failure and test a conditional predictor. Label it a mechanistic toy; it
   adds no evidence that an unchanged estimator transfers to a natural public
   accounting system. In particular do not relabel censored inventory demand
   as censored booked profit.

A successful native audit would strengthen application credibility, not by
itself establish a strong new learning algorithm. A full learning-prevention
claim additionally needs acquisition under reported reward, a booked target,
strong cheap baselines, repeated learner seeds and retained legitimate edge.
No such new result was produced or assumed here.
