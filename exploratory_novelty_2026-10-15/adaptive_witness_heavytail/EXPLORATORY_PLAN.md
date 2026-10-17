# Cost-bounded certificate under unbounded price tails

Exploratory mathematical/generative validation, 15 October 2026. This plan is
written before drawing its new search/confirmation samples. Distributional
truth is analytically known; this is not a preregistered learner experiment.
No Study R artifact, trained outcome, DP solution or market panel is read.

Use two independent fair-sign Pareto-magnitude laws, shape 1.5 and 2.5, minimum
magnitude 1. Both have finite first moment; shape 1.5 has infinite variance.
The action is one fixed unit, equity 2, deterministic cost .1. Net payoff is
X=epsilon*Z-.1. Forgiveness is F=(-2-X)+. For each law, search on 8192 IID paths
and confirm on 262144 independent paths. Seeds 2026101501 and 2026101502 are
fixed phase seeds, shared across laws for comparability.

Select an upper cap of the accounting residual at alpha=.8 from {4,16,64,256}
using SEARCH only. Equivalently cap F at U=(cap+.1)/.2. Maximize a projected
confirmation-size empirical-Bernstein lower score computed on SEARCH, keeping
all cap scores and selecting the first maximum. This is standard cap selection,
not a new concentration rule. Confirmation data are never used to choose a cap.

Confirm a lower bound L_F on E[min(F,U)] with the one-sided Maurer-Pontil bound,
sample variance ddof=1. Allocate delta=.05/2 across the two laws. Then
J_alpha >= (1-alpha)*L_F-.1 simultaneously for every alpha in [0,1], for this
same fixed action and physical dynamics. Booked expectation is exactly -.1.
Report alpha=.8 and alpha=1, the full nonempty certified charge interval, and
the exact analytic truth. No additional alpha-wise testing is needed. Full
booking is an analytic negative control; failure to confirm is inconclusive.

The lower bound on the untruncated residual is the cost-only value -.1, so
upper capping is sound without any bound on price or gross-return variance.
Direct bounded empirical-Bernstein evaluation of the raw alpha=.8 reward has
no finite known support here. Mark it unsupported; never use the sample maximum
or sample variance to pretend the law is bounded or finite-variance.

Verify the analytical mean h=.5*(u-cost)^(1-p)/(p-1), and the capped mean
.5*((u-cost)^(1-p)-(u-cost+U)^(1-p))/(p-1). Save all predetermined cases,
including any confidence failure or abstention; do not change seeds, sample
size, cap grid, law or alpha after viewing results. This is a capability test of
the accounting reduction, not evidence that its policy search or acquisition
prediction beats generic alternatives. Its payoff law differs from the finite
Nordic empirical-panel benchmark and makes no claim about real market tails.
