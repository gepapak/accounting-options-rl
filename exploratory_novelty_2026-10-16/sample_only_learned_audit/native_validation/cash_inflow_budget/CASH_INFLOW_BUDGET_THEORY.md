# Cash-inflow lower endpoint for the alpha=0 financial component

Source-only feasibility check, 2026-10-16. This is an ordinary cash-conservation/summation-by-parts argument, not a new generic mathematical method. It introduces no native call, learner, fit, numerical confidence limit or changed frozen artifact. It does not reinterpret the original forty-case experiment.

## Step-boundary version

Use actual money in one declared shared cash account, with initial cash e_0 and finitely many steps t=0,...,H-1. Suppose, pathwise:

1. e_t >= 0 at every step boundary, including the end;
2. e_{t+1}-e_t = X_fin,t + D_nonfin,t exactly: X_fin is the sum of **actual** financial cash increments, including their native floor injections, and D_nonfin contains every excluded increment of that same account, including its own injections;
3. w_0 >= ... >= w_{H-1} >= 0. Ordinary w_t=gamma^t, gamma in [0,1], has w_0=1;
4. w_0 e_0 + sum_t w_t (D_nonfin,t)^+ <= K for a justified deterministic K, uniformly over all allowed paths/contexts.

Then the alpha=0 charged financial component obeys the pathwise bound

    Y_fin,0 = sum_t w_t X_fin,t >= -K.

Indeed, exact summation by parts gives

    sum_t w_t (e_{t+1}-e_t)
      = w_{H-1}e_H - w_0e_0
        + sum_{t=1}^{H-1}(w_{t-1}-w_t)e_t
      >= -w_0e_0.

Subtract sum_t w_t D_nonfin,t and use D <= D^+. This is pathwise; adaptation, nonlinear feedback, endogenous fees and arbitrary dependence do not alter it. The weights can be random if these pathwise ordering conditions hold. Different timing weights for different attributed flows require an event-level formulation; do not silently apply one step's weight to a later cash update.

Here D_nonfin,t can be the **net** excluded increment in step t because boundary nonnegativity is assumed. A bound on a larger sum of individual positive excluded credits is also sufficient, but may be less sharp.

## Event-credit version when cash may dip between financial events

A useful weaker cash condition is possible with a stronger positive-credit budget. Index every ordered cash event j and use its actual chronological nonincreasing weight v_j (several events can share a step weight). Label each event financial, with increment X_j, or excluded, with increment D_j. Suppose initial cash is nonnegative and actual cash **after every financial event** is nonnegative. Excluded events may make actual cash negative between financial events.

Define a virtual resource account V starting at e_0. At financial events give it the same increment X_j; at excluded events give it only (D_j)^+. Equivalently,

    V_j = e_j + sum_{i<j, excluded} (D_i)^-.

It dominates actual cash. At an excluded event it cannot decrease; after a financial event it is nonnegative because actual post-event cash is nonnegative. Induction therefore gives V_j >= 0 at every event boundary. Apply the preceding identity to V:

    sum_financial v_j X_j >= -v_0 e_0 - sum_excluded v_j (D_j)^+.

Thus a deterministic bound on initial cash plus **all individual positive excluded credits** suffices without nonnegative actual cash at every intermediate excluded event. A native financial post-event floor provides the required financial post-state condition. Financial pre-event nonnegativity alone is insufficient if post-event overdrafts are permitted. The event-credit proof is an alternative to, not evidence for, unverified native step-boundary nonnegativity. This weaker algebra does not justify attributing a prior physical deficit to financial forgiveness: if pre-financial cash is negative, an injection can repair earlier excluded losses, so that semantic attribution needs a separate argument or the stronger native pre-event condition.

## Certificate and access comparison

For fixed b>=0 set T=min(Y_fin,0,b). Then -K <= T <= b and T<=Y_fin,0 pathwise. Established bounded fixed-sample lower inference on independent whole episodes applies after independent cap choice. No gross/cost conditional-moment oracle, deterministic gross/cost envelope, sign symmetry, or fitted continuation model is needed **for this positive charged-component endpoint**. If a learned centered control with absolute path budget B is added, min(Y_fin,0-M,b) instead has lower endpoint -K-B under its separate centering assumptions; a control is not needed for the basic result.

Finite first moments are not needed to bound the capped statistic. A finite-mean assumption is needed if the claim specifically means a finite expected uncapped financial return; with this lower bound alone its expectation is well-defined but may be +infinity. Positive capped mean supplies a lower bound for that expectation either way.

This does not establish booked harm. For that separate claim, one may use conditionally mean-zero integrable weighted gross flows, nonnegative integrable costs and a known positive unavoidable fee (or a separately valid positive expected-cost lower limit). Then expected booked financial profit equals minus expected executed cost and is strictly negative. Without the no-edge condition or a separate booked-return bound, positive charged financial return may reflect a real trading edge.

This access model is not “assumption free”: it substitutes a complete cash-account reconciliation and a uniform excluded-positive-inflow budget for the residual method's cost envelope. It can be useful if K is easier to justify or materially smaller than a conservative cost/gross envelope. It can be useless if physical revenues, margin conversions, refunds, reset credits or unrelated recapitalizations give a huge or unknown positive budget. The native financial component shares cash with such flows; initial sleeve cash alone is not a valid bound without proving that excluded positive credits are absent.

## Failure cases and limits

- **Partial charging with unchanged dynamics:** let e_0=0, G=0, C=L and I=L. Actual cash stays0 and Y_fin,0=0. For any alpha>0, Y_fin,alpha=-alpha L, unbounded below as L grows. A cash-inflow bound for alpha0 does not bound unbooked charge deductions. A separate bound on discounted financial injections would give -K-alpha*Ibar. If charging changes physical cash dynamics, derive a new complete account identity instead of reusing the unchanged-dynamics result.
- **Only terminal nonnegativity:** cash 0 -> -1 -> 0 with weights1,1/2 gives weighted total -1/2, below -e_0=0. Interior nonnegative boundary cash is required for the step proof.
- **Only financial pre/post cash, but a net-step budget:** within one step, excluded +1, financial -1, excluded -1 gives cash 0 -> 1 -> 0 -> -1. The financial event starts/ends nonnegative, net excluded flow is0, yet financial return is-1. The net-step bound fails. The event-credit budget counts the +1 and correctly permits the loss. If a native excluded floor repairs the final overdraft, its positive credit must enter D too.
- **Increasing weights:** cash 0 -> 1 -> 0, no excluded flows, weights1,2 gives financial return-1. Nonnegative cash alone is insufficient with increasing weights.
- **Missing credits:** an excluded transfer +2 followed by a financial cost -2 returns cash to0. Financial return is-2; calling K=0 because only initial cash was counted is incorrect.

Use executed ordered cash increments and the same financial attribution as the target, not a shaped reward, NAV including securities, model-predicted cash, or a returned actor reward with extra penalties. All investor transfers, margin releases, conversions, battery/physical-operation cash, nonfinancial floor injections and in-window cash resets that affect the account must be classified and budgeted. Carried queued claims and terminal closure must be treated consistently. Rounding allowances need separate source justification; this algebra is exact arithmetic, not a floating-point verification proof.

## Exact small checks and proposed scientific role

`exact_cash_inflow_budget_checks.py` uses only `fractions.Fraction` and toy finite enumerations. It checks the identity/lower bound on all small nonnegative step paths, the event-credit proof including negative intermediate actual cash, capped supports, both fee/maturity orders under fair signs, booked-mean identities and the failure cases above. Its exclusive JSON records source SHA and counts. These are consistency checks supporting the proof, not native positive results or exhaustive software verification.

Completed exact check: **PASS**, with 4,914 nonnegative-step paths, 9,828 cap checks, 800 admissible event-credit paths (116 with negative intermediate actual cash), 36 ordered fair-floor laws and nine partial-charge counterexamples. Only the new toy-check script was executed; no simulator or scientific endpoint was accessed.

The useful accounting-specific alternative is a **cash-funded alpha0 raw-component certificate** with explicit excluded-inflow access. It is simpler than the learned-control construction and should remain a baseline if implemented. It neither establishes a new general AI method nor inherits uniform partial-charge repair, acquisition prediction, whole-fund advantage or future-market validity. A separately designed known-fault fixture could test engineering utility, while retaining the original forty-case results and their abstentions unchanged; no such experiment is created or authorized by this note.
