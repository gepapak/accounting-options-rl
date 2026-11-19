# Exact one-step conditional-sign diagnostic — exploratory

Written before executing this estimator audit. Existing O/P/Q outcomes are
already known. No confirmatory claim, new training, DP solve, registered-file
mutation, or reference selection is permitted. Earlier failed estimators and
the failed two-fee feasibility design remain unchanged.

For max leverage 16, a uniformly sampled full empirical TRAIN row gives
X = epsilon*m - c, epsilon equally likely ±1. At equity e the partial-liability
reward is X + (1-alpha)*(-e-X)^+. Flat continuation has zero later reward, so
its expected high-versus-flat first-action contrast is exactly

    g_alpha(e) = -mean(c) + (1-alpha)*h(e),
    h(e) = mean_over_rows_and_signs[(-e-X)^+].

Sort losses -X and precompute suffix sums to evaluate h at arbitrary equity
with binary search. No sampled signs, trajectories, horizon truncation, critic
or value iteration remain. The primary diagnostic applies the existing
seven-probe/log-interpolation/400-point positive-part functional and the full
empirical one-period payoff-noise denominator. A secondary direct 400-point
evaluation reports interpolation error without selecting a preferred result.
At alpha=1 the gap is exactly -mean(c), so its positive-part index is zero.

This is an expected one-step incentive, **not** Q-star or a theorem predicting
PPO. Positive immediate expected reward certifies a feasible profitable
one-trade policy under the empirical no-edge model. A nonpositive diagnostic
does not rule out a delayed or alternative-action exploit. All statements
condition on empirical data; removal of Monte Carlo error does not remove
model/data/learner uncertainty. Bootstrap inference to a population of markets
would need a defensible dependence model and is not invented here.

Compute all 26 existing condition records and all prior flat-reference Monte
Carlo comparisons. Then calibrate only on the original ten F/H/J/K labels
(also report revised-FI exclusion), using the same geometric midpoint if
positive separation exists. Freeze that threshold across O/P/Q and report
every failure. Learning outcomes are read only after new predictions have
been saved. Legacy rank is reported separately from the shared alpha family.

The new-fee exploratory reanalysis uses exactly the previously inspected fees
0.40 and 1.50 EUR/MWh, standard caps/no impact, zones DK1/DK2 and alpha grid
{0,.25,.5,.7,.8,.9,.95,1}. The threshold comes solely from original calibration.
Apply the earlier 1.5-times confident-margin rule in both zones and report all
eligible high/low arms and every same-alpha cross-fee disagreement. This is
candidate-design evidence only; no training or Study R is launched.

Run from the code repository:

```
python exploratory_novelty_2026-10-15/estimator_audit/exact_one_step.py
```

Outputs stay in this new folder. Runtime and input hashes are recorded. The
local FI panel revision is disclosed, rather than treated as original J data.
