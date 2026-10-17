# Joint-fault v2 support and complete cash-write review — source draft

16 October 2026. This is a separate, unfrozen exploratory design after the
original run001 was completed. The two new adapter/support modules have only
been parsed, not imported into a native process. No v2 warm-up, actor query,
statistical draw or confidence limit has been produced. Preserve all run001
null findings. Positivity is not guaranteed by this known-fault construction.

## Intervention and remaining guards

Historical `audit/run_scarcity_tail_campaign.py:126–130` implements L3 through
participation fraction0, marginFalse and loss threshold1000. The proposed v2
uses explicit loss infinity, but changes the cap only AFTER the original
vulnerable .25-cap384-step warm-up. Thus this is an L3-style audit intervention,
not exact historical L3 redeployment. Preserve all20 checkpoint pairs, the8
fixed contexts,96 audit steps, original sizing base, physical/risk/meta state,
all fees, forecasts, native normalizers and queues. No outcome-based case,
context, cash, timing or policy selection is allowed.

Native `environment.py:1406–1420` leaves exposure scale1 when the configured
participation fraction is0. The genuine20D feasible-action geometry sets its
liquidity capacity to infinity at7954–7967. Zero therefore removes the hard
participation cap rather than enforcing zero volume. Tail, risk, forecast and
evidence constraints remain. `joint_fault_adapter.py` rebuilds actual native
observation buffers after cap removal and before the first actor invocation.
Every original warm parent/config snapshot remains unchanged.

The fund drawdown halt is retained: `environment.py:409` sets
`reward_calculator.trading_enabled=False`;3161/3219 then force flat exposure.
It does not independently convert assets or credit trading cash. The name
“joint fault” refers to the missing participation/margin/loss checks; it does
not mean every native safety mechanism is removed.

## Finite gross and cost envelopes without a participation bound

`environment.py:8582–8626` fixes the original sleeve sizing base F, clips the
allocation fraction to[0,1], and only subtracts collateral reservations.
Execution clips aggregate exposure to the configured finite risk cap3222 and
splits by weights of L1 norm1 at3251. The fixed MWh conversion is positive500
for these saved configurations4391–4394. Thus, with a declared floating point
cushion, every new contract obeys

    |q|_1 <= F * max_position_size * max(1,risk_cap_max) / conversion.

No current cash, price realization or participation volume enters that bound.
Retained initial capped claims are checked against it too. Native contracts
open only at the fixed decision cadence and mature after6 steps; the opener
refuses maturity outside the full480-row window4416–4456. Calendar support
covers all admissible new openings and every actual inherited claim.

Configured finite settlement clipping4017–4020 and fixed finite context entry
prices bound each maturity. Native roll fees4241–4306 are monotone in bounded
notional/volume for nonnegative configured fees/spread/impact/exponent, with
impact reference>=1 at1286–1297 and bounded tail multiplier1355–1380.
Outstanding bounded claims bound collateral funding2696–2729. Access fees
are fixed4344–4367. Hence finite gross and cost support survives cap removal,
although it can be very wide. The OLD adapter deliberately rejects cap0 at516;
the new subclass provides a separate fixed-sizing broad support. The existing
calendar routine already has a cap0-to-fixed-sizing branch.

## Complete reachable cash-write inventory

The account is `env.budget`, not NAV, shaped reward or accumulated generation
revenue. Source searches of native modules identify these writes:

| Source | Meaning | V2 reachability/classification |
|---|---|---|
| environment707/2164 | initial/reset allocation | outside the96-step audit; no in-window reset |
| native adapter413 | fixed initial cash stress | once before initial cash e0 is recorded |
| environment2725 | collateral funding with floor | included forward financial cost/injection |
| environment2735 | margin liquidation/conversion | unreachable: sole caller2783 is after marginFalse early return2761–2765 |
| environment3498 | legacy realized MTM | unreachable under horizon mode;3501 requires not horizon |
| environment3547 | legacy transaction debit | exactly0 under horizon branch3532–3538; actual roll charged separately |
| environment3601 | fallback floor after horizon opening | no credit: prior access/roll operations already leave nonnegative cash |
| environment4329/4360 | roll/access fee with floor | included forward financial cost/injection |
| environment4543 | aggregate maturity with floor | new maturities included; direct inherited maturity G/I excluded from v2 target |
| environment4762 | battery return plus floor | excluded physical account flow; positive credit bounded below |
| environment5028 | excess cash distribution | excluded nonpositive withdrawal, proved below |
| environment5065 | emergency operational recapitalization | unreachable: configFalse at4774; original adapter asserts itFalse317 |

`_execute_battery_ops` returns a flow without writing budget; only4762 applies
it. Generation revenue and battery operating costs affect accumulated fund
revenue4755–4771, not trading cash. Risk/meta/action routines may change sizing
and observations but do not add cash. Fund drawdown makes exposure flat and
does not enter the legacy realization branch. Physical assets are initialized
before warm-up; no audit physical-capacity purchase/reallocation is present.

The distribution concern is resolved by its FULL call chain, not an assumption
about its helper in isolation. `environment._calculate_fund_nav:1697/1724`
uses `FinancialEngine.calculate_fund_nav`, whose BOTH return branches at
`financial_engine.py:272/287` return max(fund_equity,0). With target cash
ratio>=0, distribution threshold>=0 and actual/evaluation distribution rate
in[0,1], an executed excess is positive and no larger than available budget.
The withdrawal cannot be negative or exceed budget. V2 support asserts these
configuration ranges and preserves native distribution code unchanged. An
earlier isolated-helper warning about negative NAV is superseded by this
complete source route; no new distribution repair is required.

Starting with nonnegative audit cash, all financial floor operations and the
battery floor preserve nonnegative cash; distributions cannot overdraw it.
Thus every financial PRE/POST cash and every step boundary is nonnegative.
The new adapter asserts exact nonnegative financial pre/post values, and the
new trace checker checks boundaries too. This prevents a financial floor from
repairing a prior excluded overdraft. A failed invariant stops the new run;
one must not silently attribute such a repair to financial loss forgiveness.

## Source cash budget for the new-contract alpha0 target

The new primary target excludes ONLY direct inherited maturity G/I, identified
by a nonzero maturity coin and relative step<6. It keeps all forward financial
costs/cost injections, all actual inherited state effects and all original
coin control terms. Every inherited contract has open<384 and settle=open+6,
hence settles before390; a new contract opens>=384 and cannot settle before390.
The target is a component under inherited state, not a no-prepaid redeployment,
complete contract lifecycle, feasible flat reference or learned-acquisition claim.

Positive excluded credits therefore have TWO possible sources:

1. Physical battery cash. `trading_engine.py:373–375` bounds moved energy by
   Emax=capacity*c_rate/6; all action modes cap requests199–327,445–487.
   Price is clipped to the fixed native interval at environment3788, and
   nonnegative degradation only reduces cash. For each known context row p,

       R_BESS = Emax * max(eta_discharge*p_+, (-p)_+/max(eta_charge,1e-6)).

   The actual4762 floor cannot increase cash by more than max(returned_delta,0)
   because its pre-cash is nonnegative. This also covers negative-price charging.

2. Direct inherited maturity cash. Before any fresh sign is used, the initial
   warm queue and fixed context determine the unsigned aggregate batch g0 at
   each old maturity. It is computed with the same native deterministic unit
   payoff and delivery index as the original adapter. All contracts in a due
   batch share one fresh epsilon. For nonnegative pre-cash,

       [max(e+epsilon*g0,0)-e]_+ <= |g0|.

   The implementation conservatively uses R_old=sum of absolute known
   per-contract/per-asset payoff terms, then rounds upward. R_old>=|g0| in exact
   arithmetic and avoids near-cancellation sensitivity from different floating
   accumulation orders. Native volume keys are validated. This is a deterministic
   amplitude bound, not a conditional-moment query or
   observed statistical maximum. It is added to the excluded inflow budget even
   though its direct G/I terms are removed from the target. Keeping inherited
   state effects alone would not justify omitting this funding source.

The pre-draw context bound is

    K = e0 + sum_t gamma^t (R_BESS,t + R_old,t).

Ordinary cash conservation and summation by parts then give Y_new,alpha0>=-K
pathwise. For a development-selected fixed upper cap b>=0, min(Y_new,b) lies
in[-K,b] and lower-bounds the desired expected component. No G/C support is
needed for THIS alpha0 lower endpoint; booked-harm reasoning still separately
uses fair predictable financial signs and nonnegative costs/mandatory fees.
The bound does not extend to alpha>0 without additional injection support.
This exact-arithmetic endpoint does not by itself certify the floating point
implementation. The proposed v2 confidence calculation retains the original
conservative numerical allowance derived from gross/cost support; its NUMERICAL
implementation therefore does not claim to eliminate all G/C access. The
separate cash endpoint can still substantially tighten the bounded statistic.

`joint_fault_support.py` records every calendar battery/old credit allowance,
initial cash, separate discounted components and K. The8-context union uses
the maximum SOURCE bound, never a sample maximum. Post-run checks require one
battery/distribution call per step, no margin conversion, nonpositive actual
distribution increments, bounded battery returns, no unexplained positive
excluded cash, bounded old credits, and the whole-path lower endpoint. Such
checks supplement the source proof; sample checks alone cannot establish it.

These are classical cash-conservation and bounded-statistic tools. The role is
an honest simpler baseline and known-fault engineering validation, not a new
generic AI theorem, a claim of estimator superiority, or a guaranteed result.
