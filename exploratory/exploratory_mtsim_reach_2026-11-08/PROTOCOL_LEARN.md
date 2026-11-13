# Public learner on gym-mtsim, fixed before the run

Simulator: gym-mtsim 2.0.0, EURUSD contract, leverage 100, published fee, equity 10,000 USD, lot = 80% of free margin. Actions: flat, or that lot held for one bar. Flat returns 0. Seeds 0–19 are contiguous blocks of 1,000 Student-t(3) unit-variance log returns from seed 20261108. The line reward is the published reward at a zero move plus the slope between relative price bumps of ±0.1%, times the realized price move, plus the open fee. It does not read the ledger and does not delete the floor.

Rules, scored after the run and not changed:

1. At scale 0.0015, on every seed, the published and unfloored sample means differ by less than 0.05 USD, and both learners commit to flat (at least 18 of 20).
2. At scale 0.012, the published learner commits to the maximum lot on at least 18 of 20 seeds, and the unfloored learner and the line-reward learner commit to flat on at least 18 of 20.
3. On 200 returns spread from −20% to +20%, the line reward and the unfloored reward differ by less than 0.05 USD.
