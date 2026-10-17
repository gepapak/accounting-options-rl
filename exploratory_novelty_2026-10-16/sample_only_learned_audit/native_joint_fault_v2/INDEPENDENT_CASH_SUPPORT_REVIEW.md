# Independent source-only cash-support review

2026-10-16. Verdict: **PASS for the stated real-arithmetic cash endpoint and reachable source contract**, conditional on the exact guards and source bindings below. This is not authorization to run an incomplete pipeline or a native outcome/positivity finding. No adapter/native import, actor call, warm-up, simulator transition, model fit or endpoint inspection was performed in this review. Only source files were read and parsed.

Reviewed source SHA256:

- `joint_fault_adapter.py`: `fa7e5878e301bd73b2eb96b8867b38c7bea4cd7bca75784427c4a48f632c76d4`
- `joint_fault_support.py`: `7ca4c11df6167d27ae48e2ba50ddacc0a8d0c943691ff51282b7272f753d9f39`
- `SOURCE_SUPPORT_REVIEW_DRAFT.md`: `812055e26892af591cb19f85b38f2a2a0909efdbbfac12fdbce4d8d0b68e7466`

## Why K bounds the declared target

The new target retains every forward financial cost/injection and new maturity cash increment, while excluding direct inherited maturity G/I. The excluded shared-account change therefore includes physical cash and the actual net cash from old maturities. Keeping their indirect state effects does not remove their role as funding.

For nonnegative actual boundary cash and nonincreasing weights, summation by parts gives `sum w Delta(cash) >= -e0`. For each step, the positive part of the excluded net increment is bounded by the positive physical battery credit plus the positive inherited maturity credit. Hence

    Y_new,alpha0 >= -e0 - sum_t gamma^t (R_BESS,t + R_old,t) >= -K.

The code includes both terms. Its old-credit allowance now uses an upward-cushioned sum of absolute native per-asset payoff terms. This is conservative relative to the absolute shared-sign batch payoff and avoids cancellation sensitivity in the earlier `sum(volume)*unit` expression. The batch signed/unsigned aggregate remains descriptive metadata. All initial claims are checked to open before384, settle exactly6 steps later and mature at relative times0..5. New claims cannot mature before relative time6, so the exclusion mask is consistent with the native opener and aggregate maturity hook.

For a fixed nonnegative upper cap b, `min(Y_new,alpha0,b)` is supported on `[-K,min(b,Gbar)]` when the separately justified native raw upper envelope Gbar is also used. The lower endpoint itself does not need a gross/cost moment law, sign symmetry or cost envelope. Booked harm remains a separate fair-gross/positive-cost argument. This statement does not apply to alpha1 or to partial charge without another injection bound.

## Complete positive-credit and nonnegative-cash obligations

- **Battery:** `trading_engine.py:373–375,445–487` bounds moved energy by capacity*c_rate/6 in both target-SOC and intensity modes. `environment.py:3788` supplies the clipped fixed-calendar price. Nonnegative degradation only subtracts cash. The bound covers both discharge at positive prices and charging at negative prices. The battery helper does not mutate cash; `environment.py:4762` applies its return and floors the resulting balance. With nonnegative pre-cash, that complete net physical cash increment has positive part at most the positive returned battery flow. A separate additive allowance for its offsetting floor injection is therefore unnecessary for this **net-step** budget.
- **Old maturities:** the excluded positive net change is `[max(e+epsilon*g,0)-e]^+ <= |g|` for e>=0. The new triangle allowance covers it even though direct old gross and injections are removed from the target. It is predetermined from warm claims and the declared context, not sampled future signs or outcomes.
- **Distributions:** the full `environment._calculate_fund_nav -> FinancialEngine.calculate_fund_nav` route clips NAV nonnegative in both branches (`financial_engine.py:272,287`). With the asserted nonnegative ratios and distribution rate in[0,1], the excess withdrawal is nonnegative and at most available cash. It creates no positive cash credit and cannot overdraw the account. The native distribution operation is preserved, not repaired.
- **Refunds/conversions/recapitalization/resets:** margin conversion is unreachable after the marginFalse early return; emergency operational reallocation is disabled; legacy realized-MTM cash conversion and nonzero legacy transaction debits are unreachable in horizon mode. Reset allocation and the initial distress replacement occur before the audit account starts. No in-audit physical-capacity purchase, refund or other positive cash write was found in the reachable native source. Generation revenue changes accumulated fund revenue, not trading cash. These source exclusions are part of the contract; enabling one requires a revised budget.
- **Attribution:** financial cost/maturity operations floor their post-cash; battery flooring and non-overdrawing distribution preserve nonnegative cash between them. `joint_fault_adapter._record_event` rejects any negative financial PRE or POST value, preventing a financial injection from silently repairing an earlier physical overdraft. The checker also requires all saved step boundaries nonnegative and reconciles excluded cash against the physical allowance. Sample checks supplement this source proof and cannot replace it.

## Numerical and scientific boundaries

The earlier inherited-amplitude rounding concern was addressed before this review verdict by the conservative absolute-term sum. The proposed implementation still retains the original gross/cost-derived numerical allowance in the confidence calculation. Thus **the exact-arithmetic lower endpoint removes the mathematical need for a cost envelope; this floating-point implementation does not claim to eliminate all G/C access**. Upward cushions, trace tolerances and the final numerical allowance are engineering safeguards, not a formal floating-point proof. They must stay frozen before confirmation; no post-outcome tolerance adjustment is permitted.

The union bound uses the maximum of the eight predetermined context K values, not observed cash maxima. `check_trace_support` is a context-level checker because it needs calendar arrays; it must receive the episode's own context support, not the scalar-only union. The future verifier must independently recompute every chosen statistic/support and fail on any cash-write/queue/metadata violation. This review does not substitute for those remaining pipeline checks.

The source correctly calls the intervention L3-style after the original capped warm-up, preserves the fund-drawdown halt, and does not promise that a fault will produce positive evidence. Cash conservation, capped inference and fitted martingale controls are established ingredients. The cash-budget representation is an honest simpler accounting baseline, not a new AI algorithm or a proof of neural benefit. Preserve all original run001 null findings and distinguish component incentives from total-fund advantage, acquisition or future-market claims.

## Pre-statistical matched cash-control amendment

The subsequently proposed alpha0 `cash_ridge_control`, `cash_mlp_control` and `cash_local_control` are mathematically compatible with the same endpoint. Let the predictable coefficient be the globally discounted current new-financial RAW half-difference plus the half-difference of one independently fitted/frozen RAW-future predictor. The current term is zero for direct excluded old maturities; their coins can still enter the future predictor. Clipping each coefficient to the remaining absolute path budget before its fresh coin gives E[M]=0 and |M|<=B, irrespective of predictor accuracy. Consequently RAW-M has support `[-K-B,Gbar+B]`, and upper capping gives `[-K-B,min(b,Gbar+B)]` with mean no larger than the target mean. The analytic local comparator sets the future predictor to zero and otherwise uses the same TUNE/cap/scale/budget protocol. All common features must precede the coin, including the current-branch calculation; do not use a realized true next state in only one fitted-model branch.

This statement checks the mathematical construction, not the still-developing worker/critic implementation. Four nuisance regressors require independent private prediction verification; the local comparator requires no fifth fit. The final proposed family is340 (320 charged choices plus20 cost), with all additional cash representations alpha0-only and no extra native paths. These source-reviewed additions precede new statistical draws and address comparator fairness; they provide no expected or observed learned-control benefit.
