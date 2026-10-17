# Sample-only financial-event certificate: exact assumptions and checks

16 October 2026. This note and `exact_sample_only_checks.py/json` are new exploratory mathematical artifacts. They train no model, run no Monte Carlo experiment, and read no prior study result. Root owns the separate learned-policy experiment. The exact neural predictor below is a fixed rational ReLU function used to check robustness to misspecification; it is not evidence about a trained critic.

## Target and event ordering

The target is **charged financial-component return**, not total fund performance or advantage over a flat policy. At a financial event let `G` be its gross cash flow, `C>=0` its executed financial cost, and `I>=0` its native financial floor injection. Event accounting gives financial reported increment `G-C+I`, charged increment `G-C+(1-alpha)I`, and booked financial increment `G-C`. Discount each at its actual event/step weight. With conditionally mean-zero gross cash flow, expected booked financial return is `-E C_discounted`.

Financial fees and matured gross flows are separate native events whenever the engine implements them separately. Apply the native floor after each event at the **actual pre-event cash**. Do not replace fee-before-maturity or ordered settlements by a clamp on their net sum. Deterministic financial cost events contribute their actual financial injections. Fair-sign maturity events contribute a paired local injection. If the actor reacts between events, each pairing must freeze the action already chosen for that event; it cannot freeze a future counterfactual action sequence.

Nonfinancial inflows/outflows and their native floor injections may occur between financial events. They change the actual cash and hence future financial injections, and the controller may observe them. Their direct cash flows and injections are **excluded** from this financial-component target. The exact suite includes such events and verifies the complete cash identity separately. An unrelated physical/battery-floor injection cannot be credited to the financial sleeve. A paired financial statistic does not by itself certify total fund return. If a flat comparator pays annual/idle fees, flat need not have zero financial value; a claimed advantage over flat requires a direct paired difference with its own bounds.

## Sign-paired financial events without a magnitude-moment oracle

Let the frozen actor choose the current financial action from its full allowed information. Let `H_t` include that history/action and any non-sign shock features made common to the two audit branches. A fresh sign must satisfy

\[
\Pr(\varepsilon_t=+1\mid H_t)=\Pr(\varepsilon_t=-1\mid H_t)=1/2.
\]

The hidden magnitude/context law can be unknown and history dependent. The common information may include a realized magnitude after the action but before its independent sign. At the actual native pre-event cash, compute the two financial injection values `I_t^+,I_t^-` under the same common information and correct financial map. Put `Iavg_t=(I_t^++I_t^-)/2>=0`. No expectation over magnitudes is evaluated. For a deterministic cost event, its point-mass version is simply the actual injection.

Write `A_pair=sum_t w_t Iavg_t`, `C=sum_t w_t C_t`, with finite horizon, nonnegative event weights and `0<=alpha<=1`. Require integrability of the actual finite-horizon financial gross flows/injections, and a known deterministic bound `0<=C<=Cmax`. Local sign averaging and conditional expectation give

\[
E[(1-\alpha)A_{pair}-C]=J_{financial,\alpha},\qquad
J_{financial,book}=-EC.
\]

This statement permits adaptive and recurrent frozen policies and nonfinancial cash events, because each local update uses its actual physical pre-event state. It assumes that each sign is fair after conditioning on everything the controller and branch predictor use. Merely observing balanced signs in a finite sample does not establish this condition. Executed quantities, fee branches and settlement timing must be paired correctly. This is an accounting specialization of conditional Monte Carlo/antithetic sampling, not a new generic estimator principle.

## Arbitrary frozen predictor and pathwise absolute-amplitude budget

Freeze any predictor `q_theta(H_t,branch)` independently of confirmation. It may be linear, neural, or based on an incorrect transition model. It need only return finite values. Evaluate **the same predictor** at both branch labels with identical common pre-sign features. Set

\[
\delta_t=\{q_\theta(H_t,+1)-q_\theta(H_t,-1)\}/2,\quad
a_t=\operatorname{sign}(\delta_t)\min(|\delta_t|,R_t),\quad
R_{t+1}=R_t-|a_t|,\quad R_0=B.
\]

Debit `abs(a_t)` **before** reading the current fair coin, and put `M=sum_t epsilon_t a_t`. The amplitude can depend on the actual previous path, but not the actual current coin except through its two fixed hypothetical branch labels. Then

\[
EM=0,\qquad \sum_t|a_t|\le B,\qquad |M|\le B.
\]

No accuracy assumption, value-function sup-norm bound or estimated conditional density is needed: the finite pathwise budget bounds the **control**, even if predictor outputs are arbitrarily large. A misspecified predictor can worsen variance and confirmation power; zero control is required as a comparator. Discounted-return predictors already produce amplitudes in global return units; do not silently apply a second discount factor. A predictor of a different reward target is still centered but may be inefficient.

An independently trained model does not repair incorrect centering. In particular, subtracting an average of modeled next-state predictions from a value evaluated at the **true actual** next state need not have mean zero. The exact counterexample has true next cash `2/0` equally likely and model predictions `2/2`: the mixed control has mean `-1`. Subtracting it produces apparent mean `+1` for a true zero-value financial component. Correctly using the common model's branch difference gives zero instead.

This is a known innovation-basis martingale control construction. Belomestny and Schoenmakers, *SIAM Journal on Control and Optimization* 62(1):650–679 (2024), Equation (5.6), already learns coefficients of known zero-mean innovation functions: [primary paper](https://www.wias-berlin.de/people/schoenma/Belomestny_SchoenmakersAccept.pdf), [DOI](https://doi.org/10.1137/22M1526010). The accounting-specific addition is the native financial-event target and its lower-bound support under the bounded control budget.

## Capping and independent confirmation

For a fixed charge define

\[
Z=(1-\alpha)A_{pair}-C-M,\qquad T_b=\min(Z,b),\quad b\ge0.
\]

Then `E Z=J_financial,alpha`, `E T_b<=J_financial,alpha`, and

\[
-Cmax-B\le T_b\le b.
\]

This needs no conditional-magnitude moment oracle or bound on the predictor itself. It does need bounded realized financial cost and valid sign symmetry. The cap restores a finite upper endpoint at the price of a one-sided lower-mean bias. Freeze the actor, predictor, budget, cap and event mapping before independent confirmation. A fixed-sample bounded empirical-Bernstein lower bound applies to IID **whole path** values of `T_b`, with range width `b+Cmax+B`; allocate error over every reported target. A SEARCH-chosen cap/budget is not reselected on CONFIRM. A sequential run requires an actual time-uniform confidence procedure, not repeated use of a fixed-n bound. Nothing here proves universal variance reduction, acquisition prediction or optimizer superiority.

Unknown finite mean cost cannot replace the deterministic negative endpoint. Truncating a negative cost downward is optimistic. If costs are unbounded, use a separately valid tail/envelope correction or abstain. A fitted cost model alone is insufficient. Positive financial value also does not imply strict booked loss when cost can be identically zero. For an arbitrary learned controller, establish a separate positive expected-cost lower bound on a jointly allocated confidence event, or use a known positive compulsory cost/probability. The old finite-ratio charged-to-booked division and global partial-charge repair are **not inherited** without their uniform ratio assumptions.

## Terminal sample-only Tanaka identity

For an additive **booked financial ledger** `N_next=n+G-C`, put `d(n)=(-n)^+` and

\[
T(n,G,C)=d(n+G-C)-d(n)+1_{\{n<0\}}G
=\begin{cases}
d(n+G-C),&n\ge0,\\
C+(n+G-C)^+,&n<0.
\end{cases}
\]

It is nonnegative. If `E[G|history,action]=0`, then `E[T|history,action]` is the true expected debt increment. **Fair sign symmetry and a conditional-moment oracle are unnecessary.** For terminal forgiveness at horizon `H`, use `A_T=gamma^(H-1)sum_t T_t`, with current-time discounted realized costs `C`. This gives the same terminal charged expectation relative to flat, subtracting the existing claim `gamma^(H-1)d(N_0)` when the start is insolvent. Upper capping and the cost-bound support argument apply again. The terminal potential has a constant final-transition discount factor; it must not receive per-step floor weights.

The exact suite tests asymmetric `G=+2` with probability `1/3` and `G=-1` with probability `2/3`, scaled by the adaptive action. This has zero conditional mean but a nonzero mean sign, so a naïve fair-sign predictor control would be invalid. At a native step floor the actual pre-event cash is nonnegative: the Tanaka correction reduces exactly to the ordinary injection. It is the existing realized injection/residual there, **not a separate new floor method**.

This terminal identity concerns the declared booked financial ledger and terminal claim. If unrelated flows enter that same ledger, their potential changes must be attributed explicitly; a total debt change cannot automatically be called financial-component loss forgiveness. Queue closure must follow the actual simulator rule: action zero means hold unless the simulator explicitly liquidates, and must not be relabeled liquidation by the auditor.

## Exact enumeration and interpretation

The floor suite enumerates both fee-before-maturity and maturity-before-fee orders, four policies (flat, constant, cash/context-adaptive and context-gated), two contextual magnitude mixtures, three steps, charges `0,4/5,1`, four frozen predictors (zero, linear, rational ReLU and deliberately wrong model), three control budgets and three upper caps. Context changes with the actual preceding sign; nonfinancial flows/floor events alter subsequent actual cash. Fractions verify path probabilities, the zero financial gross mean, financial/component cash identities, unbiased uncapped pairing, exact zero control mean, every absolute budget, every capped support and lower-mean inequality. Exact variances are retained; improvement is not presumed.

The terminal suite separately enumerates three initial ledgers, flat/constant/adaptive policies and the asymmetric mean-zero gross law. It checks the nonnegative Tanaka form, terminal relative-flat identity, cost support and capping. Counterexamples retain invalid actual/model mixing, non-equivalence of aggregate clamping, misattribution of a nonfinancial floor injection to a flat financial sleeve, and failure of sign centering under an asymmetric fair-mean gross law.

Run from this directory with `python exact_sample_only_checks.py`. The script refuses to overwrite its JSON. Reproduce to a new path with `python exact_sample_only_checks.py --output exact_sample_only_checks_reproduction.json`. The JSON binds this note and the script by SHA-256 and records exact rational results. These are finite mathematical examples checking implementation/theorem consistency, not simulations, independent markets, trained-model evidence, finite-sample power estimates or an exhaustive proof of a generic algorithm.
