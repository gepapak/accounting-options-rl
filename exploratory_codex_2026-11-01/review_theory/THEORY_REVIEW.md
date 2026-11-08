# Independent theory review, 1 November 2026

Scope: Part 1B of the user's review request. Read-only review of the manuscript,
the author exact-check source, the twin wrapper, and the T/V and N/U environment
source. No registered file, result, manuscript, or original source was changed.
No training, environment execution, market-data read, or model call was made.
No AGENTS.md was found by a workspace search. This report and its exact checker
are new exploratory review files only.

The central TIR identification and dominance argument is correct under a common
threshold and common cost, apart from the strict-boundary wording below. The
4/9 construction is also correct. However, SPR is not the sharp endpoint under
the stated censoring model, and the actual finite PMR pool does not have the
claimed general conditional unbiasedness. These are mathematical claim issues;
I did not find a result-affecting Study V/T/U implementation bug in this scoped
review.

## Findings that require changes before submission

### 1. SPR is not the sharp identified-set endpoint under C

Locations: `sections/identification.tex:18`, `sections/related_work.tex:41`,
`sections/introduction.tex:12`; repeated in the public wrapper's opening
docstring (`ACCOUNTING_OPTIONS_2026-10-14/twin_reward/twin_reward.py:7`).

With `g=10, C=1, tau=0`, the unordered twin pair is `(9,0)`. This pair identifies
`g=10` exactly. SPR instead uses `|9-0|/2=4.5`, which is outside the identified
set `{10}`, and consequently cannot be its sharp lower endpoint. The claim is
true on tied pairs, or when neither branch has a credit, but not in general.

Correct statement, for a feasible observed pair and known common C:

```
I_g(r,r',C) = {max(r,r')+C},       if r != r';
             [0,r+C],             if r = r'.
```

SPR's magnitude `|r-r'|/2` is a valid lower bound on g, generally not sharp.
TIR's magnitude is the sharp lower endpoint of this identified set; TMR uses
its sharp upper endpoint. On negative realized signs, the order of the booked
reward endpoints reverses, so 'lower endpoint' should explicitly refer to the
magnitude, not universally to booked reward.

Proof: for an unequal pair the larger reward is strictly above tau, hence equals
`g-C`. For an equal pair, any `g` in `[0,r+C]` with threshold `tau=r` reproduces
the observation, and no larger g can. This is already substantially the proof
in `supp_spr.tex:5`; the error is the additional SPR interpretation.

Recommendation: report and correct the interpretation; do not change SPR's
reward, registered verdicts, or frozen code.

### 2. PMR's finite sample mean is being conflated with a population mean

Locations: `identification.tex:40,43,56`, `supp_spr.tex:14`; the categorical
'unbiased given state and action' description also occurs in
`related_work.tex:43` and `discussion.tex:10`.

For every pooled favorable nonflat step, let

```
U_i = (r_i+C_i)/|q_i|
    = m_i + ((tau_i+C_i-|q_i|m_i)_+)/|q_i| = m_i + b_i,  b_i >= 0.
```

The valid pathwise statement is `mean(U_i) >= mean(m_i)` on those same sampled
steps. It is not `mean(U_i) >= E[m]`. For uncensored iid `m in {0,2}` with equal
probabilities, a single draw 0 gives an uncensored pool 0 below the population
mean 1. Even a censored draw can give a pool below the population mean:
`m in {0,4}`, `C=0,q=1,tau=1`, realized m=0 gives U=1 below mean 2.

For a representative population of favorable steps, strong exogeneity implies
`E[U | favorable,q != 0] = mu + E[b | favorable,q != 0] >= mu`.
No monotonicity restriction on how tau depends on m is needed for this result:
all favorable samples are retained, including the censored pseudo-magnitudes,
and U dominates m pointwise. If the method instead selected only identified or
uncensored steps, that selection would require a separate justification.

The finite pool need not even be unbiased in expectation under adaptive sample
counts. Stop after the first draw if it is 0, otherwise take two iid `{0,2}`
draws. The expected stopped sample mean is 3/4, although each draw has mean 1.
The actual pool size depends on the learned policy and favorable outcomes; a
fixed-N iid mean theorem cannot silently be applied to that adaptive pool.

The proposition already says 'with the exact mean' for its edge-unbiasedness
part. That qualifier must be retained in the access table, discussion, method
description and novelty claim. Replacing the exact mean by a finite running
estimate is a plug-in approximation, not an exactly state-unbiased reward.

### 3. Conditional PMR guarantees need a history/predictability condition

Locations: `identification.tex:34,40,43`, `supp_spr.tex:14`.

Let F_t include the pre-step history, pool, observation, chosen executed action,
and all sign-invariant inputs determining q and C. A sufficient explicit model
is `E[m_t | F_t,sigma_t] = mu`, with `mu` constant. Let the pool estimate
`mu_hat_t` be F_t-measurable. Then the exact conditional bias is

```
E[PMR_t - x_t | F_t] = |q_t| d_t (mu_hat_t-mu),
d_t = E[sigma_t | F_t].
```

Consequences:

* Exact mean gives equality of conditional booked and PMR expectations.
* Conditional fairness `d_t=0` gives expected PMR `-C_t` for any predictable
  pool, including a biased one. This is the strongest surviving guarantee.
* Under an edge, a finite or censored pool generally has a biased objective.
  Even an unconditionally unbiased uncensored pool does not suffice after
  conditioning on an action selected using that pool.

Tiny counterexample to the last point: iid magnitudes are `{0,2}`, mu=1, current
sign favorable with probability 3/4 independent of magnitude, and C=0. Build an
uncensored one-draw pool; trade q=1 iff its value is 2, otherwise stay flat.
Conditional on the traded action, PMR's expected reward is 1, while the booked
expectation is 1/2. Magnitude remains iid and independent of action and sign.

The simple factorization proof also needs joint/conditional independence, not
only pairwise marginal independence of magnitude from state and sign. Writing
the sufficient conditional moment condition above avoids that ambiguity.
For state-dependent magnitudes replace mu by a verified conditional function
`mu(F_t)` and retain the same bias identity; merely naming a conditional pool
does not prove it consistently estimates that function.

### 4. M alone does not remove the impossibility bound

Locations: `identification.tex:43,56-57`; exploratory `THEORY_PMR.md` makes the
same interpretation.

Both worlds in the 4/9 example, one with constant per-unit magnitude 1 and one
with constant magnitude 10, satisfy cross-state magnitude invariance M. With
tau=10 and |q|<=1 every query still returns 10. Pooling favorable rewards cannot
distinguish these worlds. A structural invariance assumption does not supply
the unknown common mean.

Correct statement: common-mean transfer can overcome local censoring if other
accessible exposures identify that common mean, or provide a justified
consistent uncensored calibration sample. Such exposures/calibration are an
additional identifiability/coverage assumption. Merely observing favorable
samples does not certify they are uncensored when tau is hidden. Under C and M
alone, the access table should say 'conditional unbiasedness with the exact
mean; a pooled estimate requires calibration/exogeneity conditions.'

### 5. Full M is not established 'by construction' for the chronological sleeve

Location: `supp_spr.tex:14` (last sentence). Source:
`study_t/envs_t.py:43-49,73-80,90-100`.

The source retains chronological real spread magnitudes and caps, samples
contiguous windows, and observes equity and remaining episode time. An
independent randomized sign is a construction for sign exogeneity. It does not
establish independence of current magnitude from equity, past magnitudes or
the executed exposure (which also uses the contemporaneous cap). Chronological
magnitudes can cluster, and equity/action histories can select such periods.
I did not read market data to assert the numerical size of this violation.

The random-action equity/leverage buckets in `pmr_accuracy.py:79-83` are useful
descriptive diagnostics; they do not verify equality of conditional laws or
state-unbiasedness. Preserve Study V's empirical results, qualify the theorem's
application, and test M or its conditional-moment implication if retaining it
as an explanation of why the learned reward worked.

## Boundary/scope corrections

* `identification.tex:15,43,57`, `supp_spr.tex:5`: the iff needs `g>0` AND
  **strictly** `g-C>tau`. At `g=1,C=0,tau=1` the favorable branch has zero
  credit but both rewards equal 1, so it is uncensored in the usual no-credit
  sense without being identified. State 'strictly above the threshold'.
  Continuous draws can make this a probability-zero boundary; the mathematical
  statement should still distinguish it. TIR's stated exactness for unequal
  pairs and g=0 remains correct.
* `identification.tex:67`, `supp_spr.tex:11`: the constants 1 and 10 in the
  regret example are per-unit magnitudes m, not gross magnitudes g after q has
  been chosen. Use m in this proposition to avoid conflict with `g=|q|m`.
* `identification.tex:57`: 4/9 is the value in the stated normalized two-world
  instance, not a universal magnitude for all censored states or all units.
* `identification.tex:7,70`: the text itself already discloses Study N's
  claim-increment exception. Do not present that known limitation as a newly
  discovered empirical failure. However, 'every rule ... at every step' must
  be scoped consistently if the owner-claim convention is included. For debt
  D=2,g=3,C=0 its reward pair is (1,0), impossible under `max(+-3,tau)` for one
  threshold. Its model is `max(+-g-C-D,0)`, and identified pairs recover g-D,
  not g. The fair-unbiasedness argument survives conditional sign symmetry.
* `identification.tex:47`: 'identifies nothing about booked profit' is too
  literal. Nonnegative credits at least imply booked reward <= reported
  reward. The proven result is failure of point identification in the
  unrestricted model, not absence of every bound or every special identifiable
  case.
* The real-valued theorem uses strict inequality, while the wrappers use a
  `1e-12` comparison tolerance. This needs a numerical-implementation qualifier;
  the exact real-valued error-dominance result is not literally a theorem for
  arbitrary sub-tolerance floating inputs. I found no evidence of a material
  registered-result effect from that tolerance.

## Statements that survive the skeptical check

1. Under C, unequal twins identify the gross magnitude and booked outcome;
   equal twins give the sharp interval `[0,r+C]`. TIR is exact on unequal
   pairs, equals SPR on tied pairs, and therefore has no larger absolute error
   in the real-valued model.
2. Under conditional sign fairness including the state, action and all
   sign-invariant inputs, any sign-odd twin magnitude with common own cost has
   mean `-C`; hence SPR and TIR have the same expected booked policy return
   over the original dynamics when the return is integrable. Marginal sign
   fairness alone is insufficient; that is an existing disclosed scope
   requirement, not a new empirical discovery.
3. TIR's zero-variance uniqueness on unidentified fair steps is correct within
   the stated conditionally unbiased estimator class: zero variance and mean
   -C force the estimator to equal -C almost surely there. It is a narrow,
   elementary uniqueness statement, not optimality over all risk criteria.
4. `supp_spr.tex:24,27,29,31`: the general-credit SPR expectation, constant-
   coefficient linear-class uniqueness, and conditional variance comparison
   under losing-branch credit F<=4g are correct. Coefficients satisfy
   `a-b=1`, `a+b=0`, `k-(a+b)=-1`. This gives `(a,b,k)=(1/2,-1/2,-1)`;
   it does not exclude nonlinear alternatives such as TIR. The losing-branch
   variance is `4p(1-p)(g-F/2)^2`, which is <= booked variance exactly when
   `|g-F/2|<=g`.
5. The 4/9 minimax value is correct. Put X=E[qz], Y=E|q|>=|X|. The two
   regrets are `Y-X/5` and `1-2X+Y`, so they are at least `4Y/5` and `1-Y`.
   Their maximum is >=4/9, attained at X=Y=5/9, hence q=(5/9)z. Every signed
   query is censored because its booked value is at most 9<10.
6. Own executed cost known and unchanged under the twin is a real access
   assumption, correctly listed in `prevention.tex:9`. A raw action alone
   need not reveal executed q when fills/caps intervene. TIR/SPR need a sign
   reversal operation, but not necessarily a separately observed realized
   sign: for an unequal pair its sign follows from `sign(r-r')` (as the public
   generic TIR wrapper already uses). PMR does require the gross sign and q.

## A small useful theory addition, rather than more PPO runs

The 4/9 construction generalizes exactly. Let q in [-Q,Q], own cost c|q|,
`E[eps|z]=d*z`, and per-unit magnitudes a and b with `d*a<c<d*b`. If every query
is censored by a common deficit `D>=Q*max(b-c,0)`, both worlds are observationally
equivalent even under M. Define A=c-d*a and B=d*b-c. The minimax regret is

```
Q*A*B/(A+B), attained by q=Q*[B/(A+B)]*z.
```

Proof: the two regrets are >= A*Y and B*(Q-Y), where Y=E|q|; balancing them
is achievable by q proportional to z. This turns the isolated 4/9 numerical
example into a reusable impossibility statement, and proves directly why M
alone is not sufficient. It requires no new empirical study. A second useful
addition is the exact predictable-pool bias identity above; a finite-sample
confidence bound would additionally need magnitude bounds, representative
sampling, and a bound on censoring contamination, none supplied by C alone.

## Independent verification record

Ran the new `exact_theory_audit.py`, which imports no project/environment code.
`exact_theory_results.json` binds SHA-256 hashes of the reviewed theory/source
inputs. It checks 2,070 rational parameter cases, the corrected identified
sets/endpoints, SPR's bound, TIR dominance and conditional fairness; zero
failures of those corrected claims. In 638 cases the SPR magnitude is strictly
below the already identified singleton; 80 positive-g cases hit the no-credit
boundary where twins tie. The minimax value and counterexamples are checked
with exact fractions. No author exact-check result was assumed as evidence.

No fixes were made to the manuscript or frozen/public source because these
changes alter a claim's meaning, which the user explicitly required reporting
before correction. The empirical Study V/T/U verdicts are outside this theory
checker and are being audited separately.
