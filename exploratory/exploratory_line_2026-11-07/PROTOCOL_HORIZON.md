# The passive horizon on unmodified gym-mtsim, and range by path (exploratory, 7 November 2026)

Written in an AI-assisted session (Claude in Cursor) after `run_formula_gap.py` and `run_range_logs.py`, before `run_passive_horizon.py`
and `check_path.py` were run. Exploratory: no learner, no registered decision rule. Everything that was run is reported.

## Question
"Logs do not buy range" says that two accountings agreeing on an interval write the same log inside it. The quantitative form asks how
often a log leaves the interval: the per-step probability p of a move past the floor's kink decides the number of transitions,
n_50 = ln 2 / p, at which a passive log has an even chance of holding one informative move. The tail class, not the standard deviation,
decides p. A path of k steps reaches the range of one step of size k (range by path), up to the fixed writes.

## What is run (`run_passive_horizon.py`, seed 20261109)
1. The 100 held positions of `run_formula_gap.py` (same seed and construction). The floor's distance in return units is
   lambda* = equity before the step / (exposure slope x entry price), read from the diagram.
2. The package's own EURUSD series (`FOREX_DATA_PATH`): 88 daily bars, 87 log returns. Its standard deviation fixes a Gaussian law and a
   Student-t(3) law of the same sd; the hourly scale is sd / sqrt(24). For the median lambda*: p and n_50 under each law at both scales,
   and the empirical share of returns past the floor.
3. Simulator check, positions 0-9, 1,000 draws each: (a) a bootstrap of the package's daily returns, (b) Student-t(3) daily returns at the
   same sd. On every draw the public reward and the reward of the control without the two floor lines are compared on clones. The number
   of draws on which they differ must equal the number of draws past lambda*. 20,000 clone pairs.
4. Range by path, positions 0-9: sixteen consecutive steps of -1% of the entry price each (linear, as on the line) against one step of
   -16%; the summed public reward against the single-step public reward, and the same for the control.

## Results (`results_horizon.json`)
- lambda*: 0.01191 to 0.01242 (median 0.01215): the floor sits at a fall of about 1.2% of the entry price for every position (the opening
  rule puts 80% of the free margin at leverage 100 into the order).
- Empirical returns: sd 0.00339, worst day -0.01095. No return of the package's own data reaches the floor: a bootstrap log from these
  data never leaves the interval at any length.
- Daily scale, lambda*/sd = 3.6: Gaussian p = 1.7e-4, n_50 = 4,100 transitions (about 16 years of trading days); Student-t(3) at the same
  sd p = 4.2e-3, n_50 = 165. Hourly scale, lambda*/sd = 17.6: Gaussian p = 2.4e-69 (never); Student-t(3) p = 3.9e-5, n_50 = 17,800 hours.
- Simulator check: the rewards differed on exactly the draws past lambda* on all 20 position-law pairs (bootstrap: 0 of 1,000 on every
  position; Student-t(3): 0 to 8 of 1,000 per position). The crossing identity holds on the real code.
- Range by path: on 6 of 10 positions the path's summed public reward equals the single step's, minus the equity (-967.50 to -974.50 USD).
  On 4 of 10 the first -1% step took the margin level below the stop-out, which closed the order with a loss of about 798 USD, below the
  equity; the path then ends with a positive balance and no floor is met (sum -798 against -951 to -956 for the single step). The stop-out
  is a position-responding rule outside the floored-program model, which a single step at -16% also triggers but at a loss the floor
  then hides. For the control the stop-out closes the order at step 1 or 2 on every position (sum -798 to -1,600 USD against -12,700 for
  the single step). A path is not a line: it meets the simulator's position rules, which the line, by freezing the position, does not.

## Exact check (`check_path.py`, seed 20261107, 100,000 rational checks)
For random coherent floored programs (1-3 accounts, 1-3 writes each, balances 0-20, gross coefficients 1-9, fixed writes -4 to 4):
(1) with no fixed writes, k unit steps in either direction sum to phi(+-k), k = 1..8 (40,000 checks); (2) with fixed writes, the path
differs from phi(+-k) by at most (k - 1) A, A the sum of absolute fixed writes (40,000); (3) same-sign steps of unequal rational sizes
with no fixed writes sum to phi of their total (20,000). No failure.

## What this does not show
The horizon numbers are for one position size (80% of free margin at leverage 100) and one series; a smaller position moves the floor
further and lengthens every horizon. The Gaussian and Student-t laws are illustrations of two tail classes at the data's sd, not fits.
No agent was trained. The path check uses the public simulator's own stop-out; on a simulator without position rules the path and the
single step agree exactly, as the rational check shows for the model.
