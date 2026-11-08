# Protocol, fixed before the run (7 November 2026)

Question. On unmodified gym-mtsim 2.0.0, the reward is defined as the change in equity. Does a correct implementation of that definition match the reward on every probed price, including prices at which the zero-balance write forgives a loss? If it does, can the shape of that same reward, queried on clones, still recover the exposure and the forgiven amount, using no contract size, no fee, and no order object?

The reader receives only the pairs (t, reward). Ground truth (order profit, contract size, the balance before the floor write) is computed afterwards for scoring and is not an input to the reader.

## Setup

- Package: gym-mtsim 2.0.0, imported, not copied. Symbol specification from its bundled forex table. One symbol, EURUSD, USD account, hedge on, leverage 100, stop-out level 0.2, the package's default fee 0.0005.
- Prices: a three-bar synthetic path. The open is filled on the first bar and marked to the second, which has the same close, so the probed step starts from a held position. The third close is the probed input. No training. The economic profit of the probed step is the change in the order's profit over that step. The fee is already in the pre-step profit, so it cancels in the step and the reward at a zero move is about zero. This corrects the two-bar wording above, and it was fixed before any completed run: the first execution raised `IndexError` because the open itself consumes a bar.
- Position: one buy, volume 80% of the largest step-legal volume the free margin accepts, opened at the first bar. The probed step holds.
- Line: second close = entry * (1 + 0.01 t), t on the grid {-16,-8,-4,-2,-1,0,1,2,4,8,16}. The step 0.01, rather than a smaller one, is fixed from the margin identity before any run: a buy of 80% of affordable volume then loses about 3,200 USD by t = -4, against a balance of 1,000, so both leftmost intervals lie past a total loss of the balance. Price is required to stay positive.
- Clones: one deep copy of the environment per t, taken after the open and before the probed step.
- Control: the same states on a subclass that deletes only the two lines that set a negative balance to zero. This control is not the public program. It exists to show the kink is those two lines.
- States: 100 independent entry prices, uniform in [0.8, 1.6], seed 20261107. The same 100 prices on the control.

## Reader

On the reward grid only:

- solvent slope in price: the slope of reward against t on the two intervals that contain t = 0, divided by the change in price per unit t, if those two slopes agree within 1e-6 relative; otherwise refuse. This is the only slope the reader uses. It is scored against volume times contract size, which the reader does not receive;
- flat tail: the two leftmost intervals both have slope about 0, and the far-left reward is at least 1 USD below the reward at t = 0;
- booked estimate at each grid t, in reward units: (solvent slope in t) * t + reward(0);
- forgiven estimate: reward(t) - booked estimate. The scored shortfall is reward minus economic profit.

## Rules, all 100 public-program states

1. Agreement. The absolute gap between the returned reward and equity_after - equity_before is at most 1e-6 USD at every grid point.
2. Deletion. At the most adverse grid point, the open-order block of the observation is empty, equity equals the returned balance, and the economic profit stored on the closed order is below the reward by at least 1 USD. The step's `closed_orders` info, which records only closes the action requested, is empty.
3. Shape. The solvent slope exists. Its relative error against volume * contract_size, the truth the reader does not receive, is at most 1e-4. The far-left reward is flat. The kink lies at or before the price where economic profit equals minus the pre-step equity.
4. Recovery. At every grid point the booked estimate matches the order's economic profit within 0.05 USD. On the flat tail the reward's shortfall against that profit is at least 1 USD, and the forgiven estimate matches that shortfall within 0.05 USD.
5. Control. On the subclass with the floor lines removed, the far-left reward is not flat, and the reward matches economic profit within 0.05 USD at every grid point.

A rule is met only if it holds on all 100 states. Partial counts are reported either way. No threshold above is changed after the run.
