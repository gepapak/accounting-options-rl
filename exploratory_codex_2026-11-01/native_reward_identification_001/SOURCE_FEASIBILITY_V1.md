# Native reward identification: source gate and executable prototype

This is exploratory work after Review 3. All original files and registered
studies are read-only. No new registered study or PPO is proposed or run here.
The next hand gate requires root authorization after the frozen protocol is
reviewed. The prototype itself has only run its own pure checks.

## Target/source distinction

The target is the actual public repository's `engine/environment.py`, currently
SHA256 `85c9927867196c3137f7cb682b016cd84c9a7dc0a8533278344fff4831467880`.
This is different from the earlier native-prevention adapters' Prototype5
source SHA `61d446064bf6dacadaed5314487b72437d02e97860da0cb6e192a67aee2f0d63`.
Those earlier source/broker/ledger protocols are not reused as authorization or
claimed to verify this public-engine version.

The public investor reward is a shaped sum of base, return, optional, forecast
and risk terms (`environment.py:5418-5427`), including clipping around
`5354-5376`. It is not a reported monetary equity increment. Thus wrapping
that scalar with canonical TIR is not justified. `financial_engine.py` handles
valuation/legacy MTM helpers; the active fixed-horizon cash settlement and its
cash-floor writes are methods on the environment itself.

Multiple cash floors matter: access `4360`, fresh roll fee `4329`, matured
settlement `4543`, collateral funding `2725`, battery flow `4762`. In particular,
a roll-fee debit is floored BEFORE a previously opened contract settles. Even
if that gross settlement is sign-linear, the full step is not generally one
`max(sigma*g-C,tau)` observation of booked cash flow.

Example rooted in that ordering: starting cash0, known fresh fee10, then gross
settlement +1 or -1. Native end cash is (1,0), reported increments (1,0), but
booked flow is (-9,-11). Canonical TIR on the cash pair with C10 assigns +1 on
the first branch, whereas the ordered cash inverse recovers -9 in exact real
arithmetic. This is not a change to the original reward or dynamics; it is an
access-specific inverse using known cash/cost receipts.

## Implemented broader inverse

`floor_inverse.py` starts from public cash, known signed cash-prefix writes,
one unknown aggregate gross settlement z, and known signed subsequent writes.
The shadow branch reverses z while retaining own quantities and all other
declared writes. Composition of additions and zero floors after settlement
has form `cash_real=max(z+A,T)` and `cash_shadow=max(-z+A,T)` with computable
A,T. Inverting both maps gives a sharp REAL-VALUED interval for z; shifting by
the sum of the known signed writes gives the booked-flow interval. This is
elementary floor-affine inversion, not a new generic inverse algorithm.

An observed cash above its composed floor identifies z. If both are floored,
the interval remains non-singleton; no false exactness is returned. Funding
after settlement can create this ambiguity even when the favorable gross
settlement itself was positive. Known fee-floor subsidy is removed by keeping
the source write order, rather than treating nominal fees as though they were
all paid after gross settlement.

Binary64 mode is deliberately a conservative interval, NOT a sharp discrete
identified set: it reproduces the known pre-prefix in the same binary64
arithmetic, then allows 1/8 DKK rounding error per remaining cash write on an
independently checked operand/pre-floor-sum domain of [-2**50,2**50]. Max/floor
and translations are 1-Lipschitz, so the summed radius contains the source
cash result. The interval is inverted using that radius. It has no statistical
coverage claim and assumes the unknown source gross is exactly reversed by
the shadow.

## Minimal shadow intervention and its failures

At the delayed-settlement event, the own past fill identifies each contract's
entry price and executed quantity. A trusted shadow builder deep-copies the
state and replaces only each DUE delivery's private settlement-array value by
a reflected value. All methods are unchanged, including the original getter,
support clipping, arithmetic order, cash floors and queue removal. The inverse
receives only observed ending cash and own fee/funding receipts; original
gross PnL and private settlement prices are evaluator-only gold data.

The reflection must satisfy `RN(S_shadow-entry)=-RN(S_real-entry)` exactly,
not merely set the real-number expression `S_shadow=2*entry-S_real`. The pure
search verifies this identity and support bounds or rejects. Among 20 tested
price pairs 19 passed and `(entry10,settlement0.1)` was rejected. That rejection
is a numerical access boundary, not a claim of global reflection completeness.
Four adjacent floats in each direction are searched; failure means no tested
representative passed, not a universal impossibility theorem.

Contracts sharing a delivery key must share the same entry anchor; otherwise
one exogenous price cannot reflect about both entries. The pure gate rejects
that case. Under verified opposite unit payoffs, unchanged quantities and sum
order make binary64 products/aggregate sums opposite as well (finite domain;
signed zero is immaterial to the booked interval). The hand gate independently
checks the actual source returned gross antisymmetry rather than assuming it.

This hand gate does NOT prove that a full native step's sign flip leaves
liquidation, future obligations, other cash components or funding schedules
unchanged. It fixes those schedules in a source-kernel hand state and uses
known-own-cost interfaces. A full actor rollout requires a later contract gate
for those dependencies. It cannot use a signal-expectation theorem just from
one successful pair. Nor is access to cash receipts equivalent to access only
to the engine's shaped scalar reward; this extension deliberately states the
extra public account/cost access.

## Fixed pending native hand gate

Six prospective hand cases: free fee; fee-floor TIR counterexample; losing
real branch; funding hides both signs; pre-fee plus post-funding; two distinct
delivery keys. Each is run through three original-body branches: unqueried
baseline, shadow, real. Each branch opens one fresh six-step contract and pays
its fee, settles existing due claims, and charges collateral funding on the
fresh remaining claim, in that source order. These are actual original methods
on synthetic hand proxies, not whole Env constructor/reset/step calls.

The fixed known fee uses the existing legacy fixed-fee configuration with
zero impact/spread; no private volume or learned fee oracle is read. This is
a genuine supported cost subcase, not validation of the realistic impact law.
The funding cases choose a hand configuration with the requisite charge;
they establish source semantics, not realistic-rate reachability or utility.
No claims are opened to improve a policy; this is a kernel diagnostic.

Frozen budget: 54 top-level original calls, <=2,000 counted original helper
calls, 18 hand proxies; zero Env instances/resets/steps, market reads, model
calls and fits. All original data fields are hashed for original-vs-real
coupling and no shadow mutation. Every computed booked truth must lie inside
the ordered RN interval. The fee-floor case must exclude the naive TIR +1
value, while the fully censored case must remain ambiguous. Failures retain
partial outputs and do not authorize a new roster or a relaxed criterion.

Own-code pure checks already passed: 3,300 rational cases, 180 binary64
containment cases, 19 admitted price reflections, one reflected-price search
rejection and two explicit support/conflicting-anchor rejection tests. The
first pure check mistakenly demanded every decimal pair admit a reflection;
its failure and correction are documented separately. No original/native
function was called to produce these pure results.
