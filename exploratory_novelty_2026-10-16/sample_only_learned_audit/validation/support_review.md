# Independent native support and component-hook review

16 October 2026. READ-ONLY review of Prototype5 source and the new
`../native_engine_adapter.py`; no policy inference, simulation, training or fresh
financial endpoint was run/read by this task. Line references below refer to the
current Prototype5/environment.py and must be bound by source hashes before the
final plan. The adapter is still under development, so this is a review and
derivation, not certification of a frozen final implementation.

## Main corrections and valid tightening

1. Do not use class constant META_CAP_MAX=.75 as a universal sizing bound.
   Instance initialization at1009-1012 reads cfg.meta_cap_max, and the authentic
   saved configs have.8. Allocation upper1 is a safe generic bound because
   `_get_investor_tradeable_capital_dkk` clips the fraction to[0,1]. A tighter.8
   invariant requires checking initial capital_allocation_fraction and ALL update
   paths. Native collateral reservation only reduces available capital.
2. Liquidity hard-cap tightening must preserve the scale1 fallback when the
   computed cap is0 (1386ff). The native volume helper normally forces a positive
   minimum, and causal early rows have a positive fallback too, but either prove
   those conditions for the frozen config or keep the sizing-only bound on the
   zero-cap branch. Raw volume arrays alone are not native effective capacities.
3. Roll costs occur only when a valid contract is opened, not on all H calendar
   rows. Use its row's volume, impact reference and tail multiplier, then sum
   discounted row envelopes. The former H*global-max-roll bound can be orders
   wider without adding validity.
4. Funding uses actual outstanding contracts after current maturity processing.
   Sum the allowed openings' individual volume envelopes on that calendar row
   instead of charging every row for a global maximum number of largest contracts.
5. The financial COMPONENT is not the full native cash ledger. Excluded battery
   and other transfers can replenish cash between financial losses. Thus the
   old raw floor-return lower bound minus initial cash does NOT automatically
   apply; derive raw support from the ordered event envelopes below.

## Conditional sizing and opening schedule

Let F be the ORIGINAL recorded initial trading sleeve, P the fixed MWh conversion
price (500 in these contracts), zmax the configured max_position_size, and amax a
proved upper bound on capital allocation. With the source's tradeable-capital
clipping, amax=1 is always the conservative choice; the saved meta interval may
permit.8 after full invariant validation. Include a conservative risk/weight
factor R>=1 if source arithmetic can enlarge absolute aggregate exposure.

    q_size = F*amax*zmax*R/P.

Keep F unchanged when setting initial CASH=1e-4*F (the current proposal, replacing
the earlier .1F proposal). Using the stressed cash as F
would silently change the policy's notional contract and the experiment.

For native row t, let V_t be `_market_liquidity_volume_mwh(t)`, not its raw input
array. Let Vpast_t be `_causal_market_liquidity_volume_mwh(t)`; with h=6 this uses
delivery t-7, and a positive configured early fallback. For cap fraction k,
K_t=k*min(V_t,Vpast_t). A safe per-opening volume envelope is

    Q_t=min(q_size,K_t) if K_t>0; otherwise Q_t=q_size.

The scale1 fallback matters mathematically even when unreachable under current
valid positive-volume metadata. Only upper-bound active quantities here; risk,
margin, cash, no-trade thresholds and learned gates may reduce them.

With horizon H96, fixed investment frequency6, settlement lag6, fresh initial
queue and no off-cadence contract creator, the allowed opening set is

    D={t: 0<t<H, t%6=0, t+6<H}={6,12,...,84}.

The source refuses time0/nondecision execution(3117ff), and
`_open_horizon_settlement_contract` returns without opening or charging if
settle_step>=len(prices)(4415ff). There is at most one new aggregate contract on
each allowed row, even when the target exposure has not changed: the hold branch
still opens a new settlement contract(3617-3633). Do not use a turnover-only bound
that assumes a held position incurs no new roll. In this exact horizon mode,
the legacy delta-trade fee is explicitly zero(3529ff); the new contract roll is
the financial execution cost, so there is no extra factor2 turnover charge here.

If a future config permits a different/adaptive cadence, construct D as a safe
superset of all reachable opening rows and permit its corresponding overlap.
Do not retain the six-step simplification without its configuration check.

## Native calendar roll-cost envelope

Let N_t=P*Q_t, Ref_t=max(native impact reference at t,1), s the nonnegative
half-spread bp, b the impact coefficient bp, p the impact exponent and tau_t an
upper bound for the source's causal tail multiplier at t. Source
`_estimate_horizon_roll_cost`(4241ff) is monotone in notional/volume at fixed
nonnegative parameters. At t in D with N_t>100, use

    Roll_t = BaseFee(N_t,Q_t)
             +N_t*s/10000
             +N_t*b*tau_t*(N_t/Ref_t)^p/10000.

Otherwise Roll_t=0. For the native Nord Pool fee proxy,
BaseFee(N,Q)=Q*max(transaction_fee_dkk_per_mwh,0)*max(friction_multiplier,0).
Legacy notional/fixed fees follow their declared source function instead.
Use the native helper's max/None fallback semantics rather than a nominal formula
that disagrees at zero values. The source clamps impact's reference to at least1
and initializes valid volume references positive(1286ff).

tau_t may safely be cfg.liquidity_tail_impact_max_multiplier. A tighter
deterministic tau_t is the native `_liquidity_tail_impact_multiplier(t,record=False)`:
it depends only on already observed delivery t-7 and fixed cached inputs, not
on the fresh stress coin or policy action. Bound it outward for floating point.
For a declared set of eligible contexts/starts, use that set's support union or
check bounds separately by its predetermined context. Do not use the extrema of
sampled confirmation episodes or discard a zero/missing-liquidity branch after
observing it.

## Outstanding collateral and funding

For each calendar t after maturity, safe outstanding opening rows are
O_t={u in D:u<=t<u+h}. Native contract notional is P times its MWh magnitude.
Let qopen_t=sum_{u in O_t}Q_u. Then

    Coll_t = qopen_t*max(P*margin_fraction,
                          stress_price*stress_loss_fraction),
    Fund_t = Coll_t*max(annual_funding_rate,0)*step_hours/(365.25*24).

This directly matches `_collateral_required_dkk`(2696ff) and
`_apply_collateral_cash_drag`(2714ff). With h=f=6, after settlement there is at
most one allowed active opening; a two-contract before-settlement envelope is
still safe if exact call ordering is not relied on. Actual margin liquidation,
partial/zero trades and reduced cash can only shrink this bound. If funding is
also called at another native point, union the relevant O_t sets or add both
calls; software validation must count actual calls per step.

The per-row market access fee Access_t is the native prorated fixed positive
fee(4344ff, market_fee_protocol.py), once per calendar step. It is charged even
at time0 and while flat. Therefore

    Cbar_t=1{t in D}Roll_t+Access_t+Fund_t,
    Cmax=sum_t gamma^t*Cbar_t.

A proved positive sum of unavoidable Access_t gives strict expected booked harm
analytically under the fair-financial-gross identity. Keeping the40 proposed cost
confidence events allocated remains conservative. This is absolute cost/harm;
it is not a trading advantage over always flat.

## Maturity gross envelope and queue closure

For an opening u, entry e_u is the native `_horizon_entry_price(u)`(4201ff).
With current-price mode it is the context price clipped to[-1000,1e9]. Settlement
is clipped to the CONFIGURED [L,U] bounds(4141ff), not a sampled maximum. Thus

    SpreadBar_u=max(abs(L-e_u),abs(U-e_u)),
    Gbar_(u+h)=Q_u*SpreadBar_u.

The contract's signed asset quantities have absolute sum at most Q_u, so this
also bounds its aggregated payoff. With deterministic lag6 and one opening per
row, at most the opening u=t-6 can mature on t. A generic delayed/batched engine
must sum all admissible due openings instead. The fresh fair sign multiplies
the whole actual maturity batch, leaving this magnitude envelope unchanged.
Using exact fixed context e_u is support information, not integration of the
unknown payoff law; its use and source access must be disclosed to all methods.

The native opener refuses every contract whose maturity is outside the finite
window. Starting with an empty queue, every allowed contract therefore matures
before H (or is removed by a native margin operation), and terminal queue0 must
be asserted on EVERY episode. `_mark_horizon_contracts_to_market_for_step`
(4539ff) explicitly assigns zero unrealized value in this protocol: remaining
financial_positions are target-notional records, not additional settlement
contracts. No extra artificial close is needed if those source conditions and
queue assertions hold. If any profile changes that contract, revisit closure.

## Charged, residual and control supports

At every financial event, cash before the event is nonnegative and native
injection F_j=max(C_j-E_j-G_j,0). Hence 0<=F_j<=C_j+abs(G_j), regardless of
unrelated cash transfers. Let Gmax=sum_t gamma^t*Gbar_t. Then safe whole-path
supports, without telescoping excluded native components, include

    B in [-Gmax-Cmax, Gmax],
    Y_alpha in [-Gmax-Cmax, Gmax],
    Z_alpha=(1-alpha)*F-C >= -Cmax.

The sharper charged upper bound follows pointwise:
Yalpha_j=(1-alpha)*(after-before)+alpha*(G-C)<=G^+, because before>=0,C>=0.
No full-component telescoping is required. For separate fair maturity events,
A_j=.5*((-E-m)^++(-E+m)^+)=.5*(m-E)^+<=m/2. Cost-only residual contributions
(1-alpha)*F-C are nonpositive. Consequently ordinary residual upper support is
(1-alpha)*Gmax, and SIGN-PAIRED has support[-Cmax,(1-alpha)*Gmax/2].
Cost-only events retain native order and are not assigned fake signs.
For path-budgeted learned control, sum_j abs(M_j)<=Bctrl gives
W=Zpair-sum M_j in[-Cmax-Bctrl,(1-alpha)*Gmax/2+Bctrl]. Upper-cap W AFTER
subtracting control and use its true lower endpoint. Tighten the capped upper
endpoint to min(selected_cap,known_support_upper), not the cap when it exceeds
the known support. All numerical quantities need outward rounding.

## Component hooks and remaining pre-freeze validation gates

The current hooks separately record roll/access/funding COST events and aggregate
financial MATURITY, and reconcile after=max(before+G-C,0). Only the temporary
native unit-payoff call inside actual maturity is sign-shocked. Forecast evidence
and observational queries outside settlement retain native semantics; the sign
is not in observations. The action-mapped-step assertion checks that settlement
occurs after current action mapping. One fresh sign per calendar maturity batch
is the declared law, not an independent coin per underlying asset.

Battery, physical revenue, margin conversion and distributions remain in native
state but outside the financial component. This is coherent for the declared
component target, not a claim that fund NAV equals financial G-C+F. Check all
financial cost paths are covered by these wrappers and that no other instrument
uses `_horizon_unit_payoff` while it is temporarily patched. Native margin
conversion can clear queues; record it separately and keep native order.

Before final freeze verify on the fixed outcome-blind software traces: actor
identity and observations, every event's cash reconciliation, no hidden sign
leak, exact queue closure, all per-row supports and cost-call counts, immutable
config snapshots, and source/data/checkpoint hashes. The current bound can use
safe allocationupper1; a source-proved.8 invariant and per-row summation improve
power without using financial outcomes. No numerical endpoint or draw was used
to derive this review.
