# Exploratory terminal-bonus timing audit (Study N)

This new audit changes no registered source, environment, result, verdict or
run-log byte. It isolates a timing mismatch in the frozen terminal dynamic
program, using exactly its 300 magnitude bins, 1,801-point ledger grid from
−3 to 6 allocations, discount .99, fee 1 and 192-period horizon. Only DK1/DK2
TRAIN panels are read. Booked controls are flat analytically.

The environment adds the forgiven terminal deficit to its final period's
reward. From a state with tau periods remaining, that bonus is weighted by
gamma^(tau−1). The frozen recurrence starts V0 = deficit and computes
Q_tau = mean(pnl) + gamma*P*V_(tau−1), weighting the bonus by gamma^tau.
Starting instead with V0 = deficit/gamma corrects this single factor while
retaining the rest of the recurrence.

The audit constructs sparse interpolation matrices algebraically equivalent
to `Interp(V).mean(axis=2)`. It first reproduces the stored original terminal
probe actions and values, then compares the corrected timing at every
registered terminal probe and across all grid state-times. This is a
sensitivity check within the existing approximate model, not a new claim of
exact environment-optimal control.

Run from the code repository:

```
python exploratory_novelty_2026-10-15/terminal_timing_audit/audit_terminal_timing.py
```

The script uses one numerical-library thread and a 240-second computation cap,
checked between matrix construction/backward steps. Its executed numerical
run completed in 10.49 seconds while the independent R training continued.
There is no training, multiprocessing or write outside this audit folder.

Outputs:

- `exploratory_terminal_timing.json`: all probe action values/gaps, original
  reproduction errors, source/script hashes and summary.
- `exploratory_terminal_probe_comparison.csv`: all 216 registered terminal
  probes (108 forgiven and 108 booked across both zones).
- `exploratory_terminal_grid_change_by_tau.csv`: full-grid changes at each
  horizon, separately by zone.
- `EXPLORATORY_RESULTS.md`: numerical findings and materiality limits.

The old pre-registration's hypotheses and its original model predictions stay
historical records. The corrected audit supersedes only the reward-timing
interpretation of that benchmark; it does not rewrite any learning result.
