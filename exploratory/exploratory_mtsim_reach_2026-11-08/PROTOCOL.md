# Public-code reach: gym-mtsim 2.0.0, written before the run

Study G trained PPO in the published action box and found no difference between the floor and the control. The prediction, fixed before this run, is that the null is a reach of zero at the lot the box actually explores, not an absence of the option.

Market: hourly Student-t(3) log returns, scale 0.0015, the Study G placebo. Leverage 100, stop-out 0.2, published fee, EURUSD contract from the package. One bar after the open. Seed 20261108.

Predictions:

1. At the minimum lot, equity in {500, 10000}, the floored and unfloored one-bar rewards agree on at least 99% of 500 draws, and the mean gap is under 1 USD.
2. At the margin-feasible maximum (80% of free margin), equity 500, the mean floored reward exceeds the mean unfloored reward by more than 1% of equity. At equity 10000 the two means agree to within 1% of equity.
3. On every draw, the published observation (balance, equity, margin, order volume, order profit) matches across the floor if and only if the rewards match, up to 1e-6 USD.
4. Two-action REINFORCE, actions {flat, maximum}, 800 episodes, seeds 0, 1, 2, learning rate 0.05, logistic policy started at 1/2. At equity 500 the floored runs finish above 0.8 probability on maximum and the unfloored runs finish below 0.2. At equity 10000 both finish below 0.2.

The learner sees only the published step reward. No ledger term is added.
