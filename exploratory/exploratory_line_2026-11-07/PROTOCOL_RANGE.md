# Protocol, fixed before the run (7 November 2026)

Question. On the gym-mtsim state already used as the figure in the formula-gap audit, can any number of logged rewards whose price moves stay inside 0.4% tell a floored account from the same account with the two balance-floor lines removed? The active probe at 16% is the comparison. The reader of the log is ordinary least squares. It does not receive the contract size.

## Setup

- The same open position as `results.json` example entry (the figure state): unmodified gym-mtsim 2.0.0, and the control that deletes only the two floor lines.
- Passive log: 5,000 returns drawn uniformly from [-0.004, 0.004], seed 20261108. Each return is one probed third-bar close, public and control, from a fresh clone of the open.
- Active probe: the same position at a return of -0.16.
- Fit: public reward on the change in price, intercept free, ordinary least squares on the 5,000 points.

## Rules

1. Indistinguishability. The public and control rewards differ by at most 0.01 USD on every one of the 5,000 logged returns.
2. Slope. The least-squares price slope is within 1e-3 relative of volume times contract size.
3. Extrapolation. At the -16% move, the fitted line differs from the public reward by at least 5,000 USD and from the control reward by at most 1 USD.
4. The public and control rewards at -16% themselves differ by at least 5,000 USD.

A miss is reported as a miss. The truncation 0.4% is not moved after the run.

## Not a rule

One exploratory log of 5,000 Student-t(3) returns with standard deviation 0.15%, the scale used in Study G, seed 20261109. Count how many of those returns make the public and control rewards differ by more than 0.01 USD. This count is descriptive. It is not a decision.
