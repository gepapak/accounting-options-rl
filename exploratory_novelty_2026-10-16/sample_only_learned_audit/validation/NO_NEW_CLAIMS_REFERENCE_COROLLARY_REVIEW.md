# Conditional no-new-claims reference corollary: independent review

16 October 2026. Source-only mathematical review. No endpoint was inspected, no experiment/source/plan was changed, and no new estimator, learner or experiment is proposed. The current auxiliary plan and its `new_contract_source_snapshot_run_001` copy have identical SHA-256 `D6CF689A9EF323A7EF1C8FB87E6CF6CEA5A2D5ED2AD00CCC40FFDE4A1072E79B`.

**Verdict: valid as a conditional corollary for the fixed auxiliary financial component. It does not prove that a native flat/no-new-claims policy exists or that the experiment measures flat-deployment advantage.** The frozen plan correctly excludes those stronger interpretations.

## Statement and proof

Fix the simulator profile, initial-state/context law, event attribution and nonnegative discount weights. Apply the frozen auxiliary rule to each policy's own actual trajectory: exclude direct gross and injection of inherited maturity events, retain every forward financial cost and cost-event injection, and retain inherited-state effects.

Let `r` be an admissible reference whose financial decisions open no audit-created claims, such that **every retained gross flow is zero** and **every retained injection is attached to a financial cost-only event**. Suppose its pre-event cash satisfies `e_j>=0`, costs satisfy `c_j>=0`, and its native floor gives `i_j=(c_j-e_j)+`. For every charge `0<=alpha<=1`,

`-c_j+(1-alpha)i_j <= -alpha*c_j <= 0`.

Therefore the reference's actual auxiliary charged component satisfies

`Y_aux,alpha(r)=sum_j w_j[-c_j+(1-alpha)i_j] <= 0`

pathwise, and hence `J_aux,alpha(r)<=0` when the expected component is well defined. This conclusion requires neither a fair-sign law nor an expectation oracle for the reference: its retained gross is identically zero. The audited policy still requires the assumptions supporting its own lower confidence limit.

If a valid simultaneous event gives `J_aux,alpha(pi)>=L>0`, then on that **same event**

`J_aux,alpha(pi)-J_aux,alpha(r) >= L > 0`.

The deterministic reference inequality consumes no extra statistical error and needs no separate reference-mean confidence interval. It is a difference of expected values, not a pathwise ranking of `pi` against `r`. The reference may have different cash histories, future costs, battery actions and physical transfers; forcing it to share the audited policy's realized state sequence would generally not be an executable counterfactual.

## Assumptions needed for a native reference claim

1. **Executable nonanticipative policy.** At every history reachable under the reference, an allowed action/mapping must prevent all new claim openings, including automatic rolls and mandatory orders. Zero action, zero position delta, hold, liquidation and stopping the sleeve cannot be treated as equivalent without source proof. Changing guards, engine configuration or bypassing its mapping would change the policy class/profile.
2. **Same target and initial law.** Retain the same inherited queues, cash intervention, context mixture and component-attribution rule. No-old-claim initialization is a different intervention. Establish that all the reference's inherited settlements remain excluded under the frozen six-step attribution boundary, and that there are no retained liquidation/interest/gross events. “No new claims” implies zero retained gross only under this event taxonomy and maturity-classification condition.
3. **Nonnegative cash at each reference cost event.** This must follow from native source semantics/invariants for the reference, not merely from observed audited-policy traces. Interleaved physical losses or transfers are harmless to this proof only if the required pre-event cash condition remains true.
4. **Actual separate native events.** Costs are nonnegative, use `i=(c-e)+`, and include all corresponding cost injections. Every excluded old maturity injection remains excluded. A standalone recapitalization event with zero associated cost is not automatically covered. Preserve event ordering and a well-defined finite-horizon expectation.

The reviewed adapter records the native max-floor cash identity but `_record_event` does not itself assert nonnegative pre-event cash. The frozen auxiliary analyzer checks its supplied audited trajectories with `cash>=-1e-6`; this is not a universal proof for an unexecuted reference. This review therefore establishes the conditional corollary, not existence/admissibility of a native reference actor.

The pre-cash condition is essential: `e=-1`, `c=.25`, `alpha=0` gives `i=1.25` and charged cost-event component `+1` despite zero retained gross. If a numerical implementation proves only `e_j>=-eta_j`, the corresponding upper bound is `(1-alpha)sum_j w_j eta_j`, rather than exactly zero. An advantage statement then subtracts that deterministic allowance from `L`; the ideal-arithmetic theorem should not silently borrow a floating-point tolerance as exact nonnegativity.

## Scope-preserving wording

> For any admissible policy that opens no new claims and whose retained auxiliary events are cost-only at nonnegative pre-event cash, the auxiliary financial component is nonpositive pathwise. A positive valid lower limit therefore also establishes an expected component advantage over every such reference, without separately estimating its mean; existence of a native policy satisfying these conditions is not established here.

The frozen auxiliary inference concerns `alpha=0` only. The reference sign inequality holds for all charges, but a positive uncharged lower limit does not certify the audited policy at other charges. A learned-control statistic can itself be positive on a reference path because its centered control is random; the pathwise claim concerns actual auxiliary charged return, not every corrected observation. Nothing here proves whole-fund advantage, complete lifecycle profitability, acquisition after intervention, a no-inheritance counterfactual or a new algorithm.
