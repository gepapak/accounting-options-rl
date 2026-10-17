# Established fixed-sample boundary: unrestricted negative tails

Source-only theoretical check, 2026-10-16. No native endpoints, checkpoint contents or numerical experiment outputs were inspected. This is an elementary specialization of an established nonparametric-testing impossibility, not a new audit theorem.

## Exact statement, including randomized procedures

Fix a finite sample size N and error level delta in (0,1). Let P be the class of all real-valued distributions with finite absolute first moment. A certificate may be randomized: after averaging its independent internal randomness, write its conditional probability of declaring “mean > 0” as a measurable function phi:R^N -> [0,1]. Suppose it has **uniform finite-sample validity**:

    sup_{P in P: E_P X <= 0} E_{P^N} phi <= delta.

Then, for every Q in P with positive mean,

    E_{Q^N} phi <= delta.

Thus no such certificate has power exceeding its allowed false-certificate probability at even one positive-mean alternative. “Impossible” does not mean a positive declaration can never occur; a data-independent delta-probability declaration is valid but scientifically uninformative.

### Checked contamination proof

Fix Q with mean m>0 and any epsilon in (0,1). Choose

    A_epsilon = [(1-epsilon)m + 1]/epsilon,
    P_epsilon = (1-epsilon) Q + epsilon point_mass(-A_epsilon).

Each P_epsilon has finite absolute first moment and mean exactly -1. Generate its N observations using independent mixture indicators. With probability (1-epsilon)^N all observations come from Q; conditional on this event their joint law is exactly Q^N. Nonnegativity of phi therefore gives

    delta >= E_{P_epsilon^N} phi
          >= (1-epsilon)^N E_{Q^N} phi.

Letting epsilon decrease to zero proves the claim. Independent randomization is already included in phi. Equivalently, the entire N-sample laws can be coupled to disagree with probability at most 1-(1-epsilon)^N, even though their means have opposite signs.

If Q is bounded, every contaminating P_epsilon is itself bounded and has moments of every order. What is missing is a **common known** lower endpoint or quantitative tail/moment restriction across the class. Existence of a finite variance or of all moments, with unknown unrestricted sizes, does not repair the result.

## Consequences and precise limits for this audit

- A uniformly valid lower confidence limit L_N for the mean would make the event L_N>0 such a test. Under the unrestricted class, its probability at any positive-mean Q is consequently at most delta. This concerns uniform finite-N validity; it does not deny pointwise laws of large numbers or pointwise asymptotic approximations.
- Any finite same-law FIT/TUNE/CONFIRM data, model fitting and model selection can be treated as one measurable randomized procedure using the total finite number of sampled paths. A fitted cost predictor, held-out prediction error or the largest observed fee cannot exclude an unseen rare catastrophic cost. An externally justified parametric law, validated uniform tail bound or true expectation oracle is additional information and changes the admissible class.
- Upper capping min(X,b) lowers the mean but leaves an uncontrolled negative tail. Lower clipping max(X,l) removes that tail by **increasing** the mean; it is not a sound lower-mean substitute without a valid correction for the removed negative expectation. Bounded centered controls do not by themselves eliminate an unrestricted catastrophic negative tail.
- A deterministic executed-cost bound is one **sufficient**, not necessary, route to the realized-residual lower endpoint. The impossibility applies only to classes admitting the contamination above for the actual target/observed records. Accounting cancellation, a separately proved target lower bound, an appropriate uniform tail/moment envelope, or stronger known conditional moments can exclude it. In particular, an alpha=0 cost-only event at nonnegative cash e has injection I=(C-e)^+ and contribution I-C=-min(C,e): unbounded C does not create an unbounded negative contribution when e is bounded. Do not infer impossibility merely from unbounded cost support.
- This note makes no claim about arbitrary sequential stopping or unlimited adaptive simulator queries. It already establishes the requested fixed-budget boundary. Fresh independent whole episodes are the sampling units in the application.

Suggested checked remark:

> Without a structural lower endpoint or quantitative control of the target's negative tail, finite-mean assumptions alone cannot support a nontrivial uniformly valid fixed-sample positive-mean certificate: arbitrarily rare negative contamination preserves the observed sample law while reversing its mean. This is the classical Bahadur–Savage boundary. Fitted cost predictions do not supply the missing tail restriction; a cost envelope is one sufficient accounting-specific way to provide it.

## Verified antecedents and access limits

1. **Bahadur, R. R., and Savage, L. J. (1956).** *The Nonexistence of Certain Statistical Procedures in Nonparametric Problems*. **The Annals of Mathematical Statistics 27(4):1115–1122**. [Original DOI](https://doi.org/10.1214/aoms/1177728077); [official JSTOR issue listing](https://www.jstor.org/stable/i312788); [original article record](https://www.jstor.org/stable/2237199). This is the closest classical antecedent. Some later references incorrectly print volume 25; the original issue is volume 27. Original publisher full text was not retrievable in this check, so no original theorem number or quotation is asserted. The special case needed here is proved explicitly above.
2. **Romano, Joseph P. (2004).** *On Non-parametric Testing, the Uniform Behaviour of the t-test, and Related Problems*. **Scandinavian Journal of Statistics 31(4):567–584**, DOI [10.1111/j.1467-9469.2004.00407.x](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-9469.2004.00407.x). The primary publisher abstract explicitly revisits and extends Bahadur–Savage and contrasts uniform behavior with restrictions controlling tails. Its full article was not retrieved, so no precise internal theorem number is attributed.
3. **Shah, Rajen D., and Peters, Jonas (2020).** *The hardness of conditional independence testing and the generalised covariance measure*. **The Annals of Statistics 48(3):1514–1538**, DOI [10.1214/19-AOS1857](https://doi.org/10.1214/19-AOS1857); [authors' primary full text](https://arxiv.org/pdf/1804.07203). Sections 1.2–1.3 explicitly define randomized tests, distinguish fixed-N uniform level from pointwise asymptotics, and credit Bahadur–Savage for mean untestability. They identify total-variation density of null laws as a sufficient mechanism. Their conditional-independence theorem is a different result; it is not being relabeled as the financial claim here.

The [University of Chicago teaching proof](https://home.uchicago.edu/~amshaikh/webfiles/bahadur_savage.pdf), Theorem 1.1 and Lemma 1.1, independently checks the same mixture/no-contamination argument for mean tests. It is an authoritative exposition rather than an additional primary research claim. No bibliography or manuscript edit is implied by this note.
