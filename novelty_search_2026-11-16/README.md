# Novelty search, 16 November 2026 (exploratory; nothing here is registered, in the manuscript or in either repository)

This was an AI-assisted session (Claude Opus 5.5 in Claude Code), run at the author's request after Study DR to dig for a stronger
contribution. Every direction that was checked is listed below, including the nulls, so that none is later chosen after the fact. Read
together with `../study_dr_2026-11-15/posthoc_2026-11-16/` and the 16 November entries of `../study_dr_2026-11-15/RUN_LOG_DR.md`.

**UPDATE 16 Nov, 23:15.** The registered Study XE (`../study_xe_2026-11-16/`) supported 4 of 7 decisions:
- **Supported:**
  - PPO learns backtesting.py's floor: 12 of 16 seeds against 1 of 16 in backtrader, p = 8.5e-5.
  - Learned agents report 27 to 31 points above booked.
  - The engines agree exactly inside the range.
  - One query classifies every engine.
- **Not supported:**
  - the engines' reported means differing past the kink on a fresh sample (it drifted upward);
  - the optimizer half for backtrader;
  - the short bias at the defaults (6 of 12).

The exploratory summary below is from before the registered run; where they differ, the registered results take precedence.

## Summary for the author (as of 13:00, 16 Nov; all exploratory)

1. **backtesting.py (9,243 stars) forgives losses beyond the account.** It sets negative equity to zero and stops. This happens with
   leverage, and also at the default margin = 1 for shorts. backtrader, vectorbt, zipline and NautilusTrader book the loss.
2. **Engines that agree on every solvent trade report opposite verdicts past the kink.**
   - On 2,000 fair paths the two engines are identical up to L = 5.
   - At L = 200, backtesting.py reports +26.0% [11.7, 40.9] against -7.4% in backtrader and the ledger.
   - An optimizer of the reported return picks L = 200 in backtesting.py in 100% of resamples.
   - This qualifies arXiv 2603.20319, whose strategies take no shorts or leverage and whose engines did not include backtesting.py.
   - One query past the kink classifies every engine.
3. **PPO trained through the unmodified engines learns the engine's accounting.**
   - Six actions: 4 of 10 backtesting.py seeds bet 200x near ruin, against 0 of 6 in backtrader.
   - Two actions: 3 of 4 against 0 of 4.
   - Over leverages 20 to 200 with two actions: 11 of 16 against 1 of 16 (pooled post hoc).
   - Learned agents report +20% to +47% from 1,000 USD while booking about 0.
4. **At the defaults, the accounting picks the direction.**
   - On fair crypto-like paths a no-edge short reports +1.2% (2021 volatility) and +5.1% (x2); longs cannot gain.
   - PPO: 3 of 4 backtesting.py seeds went short and none long; backtrader split both ways.
5. **A closed form.** The forgiveness is a discretely monitored barrier's overshoot. P(ruin) x 0.5826 x L x sigma_bar predicts it within
   5% on EURUSD-like bars, and the ruin probability within 0.007.
6. **Failures and limits.**
   - Net-edge predictions of when PPO learns failed at low leverage (batch 5, 1 of 3 supported). The drift explanation then failed
     too (batch 6).
   - The floor does not bind in backtesting.py's own ML tutorial setting.
   - Seed counts are small and nothing is registered.

The next decisions are in `NEXT_STEPS.md`. The registered version is in `PROPOSAL_STUDY_XE.md`, and the draft manuscript text in
`INTEGRATION_NOTES.md`.

## Directions checked

| Direction | What was done | Result |
|---|---|---|
| LLM agents under limited liability | Prior-art check | Already shown. Jarne Ornia et al. 2025 (arXiv 2505.23436, cited by the paper) ran six LLMs as investment managers with 1 or 10 USD: 74–100% chose the leveraged bet at 1 USD in four of the six (the larger ones), and 0–4% did at 10 USD in five of the six. Also arXiv 2509.22818 (LLM gambling). Not new. |
| SB3 reward clipping as a hidden floor | Clip thresholds read from Study DR's saved normalization states | Does not bind. The thresholds are 13,000–74,000 USD in one step (10 standard deviations of the discounted return), far above any low-equity bust. |
| Storage environments that pay for undelivered energy | `step()` bodies of the 12 storage and bidding repositories in the paper's electricity-RL audit, read in memory (`scan_delivery.py`, which writes the excerpts it reads to `delivery_evidence/`; they are third-party code and are not included) | Two minor cases. albanpuech's discrete environment pays a requested charge or discharge whatever the state of charge, and energy-py's proof-of-concept battery pays price x requested power. The main environments (energy-py, aws-samples, masa2203, RichardFindlay) settle delivered energy. Small, student-level repositories. (the excerpts are third-party code and are not redistributed; `scan_delivery.py` regenerates them) |
| Mainstream backtesting engines | Ruin and liquidation code of backtesting.py, backtrader, freqtrade, vectorbt and Jesse, read in memory (`scan_engines.py`, `engines_scan.txt`) | backtesting.py floors equity at zero and stops (`_Broker.next`, "If equity is negative, set all to 0 and stop the simulation"). backtrader rejects orders that would push cash negative but books mark-to-market losses in full. freqtrade and Jesse liquidate at the liquidation price; freqtrade exits a gap past it at the open and books the loss. vectorbt carries short debt. |
| freqtrade's RL module (FreqAI) | Source of `freqai/RL/` and the default `ReinforcementLearner` reward | The example reward is exploitable in the open: +25 for every entry, and the exit reward doubles above a profit target. It is labelled "not for live production". Episodes end at an 80% drawdown. Not hidden forgiveness. |
| gym-anm (distribution-network RL benchmark) | Code read; ANM6Easy installed in a scratch environment; raw costs recorded before the environment clips them (`anm_reach.py`) | Its cost clipping (energy loss at 1, violation penalty at 100 per step) almost never binds. Over 600 steps per policy, energy loss never exceeded 1 under any policy. The penalty exceeded 100 on 0% of steps under PPO's initial policy and the zero action, and on 0.5% under uniform random actions (forgiven mean 0.01). The reach is about zero. Network collapse costs 20,000. |
| A predictor of exploit acquisition from PPO's initial policy | `../study_dr_2026-11-15/posthoc_2026-11-16/initial_gradient_dr.py` | Too noisy to read: ledger noise dominates. |
| The trained critic as an audit (does V favour poorer states under a floor?) | `../study_dr_2026-11-15/posthoc_2026-11-16/critic_probe_dr.py`, statistic fixed before it ran | Null: AUC 0.554, and V16's comparison went in the wrong direction (p = 0.94). |
| Cross-engine verdicts on one leveraged trade | `engines_pilot.py` (below) | See below. |

## The cross-engine pilot (`engines_pilot.py`)

arXiv 2603.20319 (Yin et al., March 2026) compared five engines on 15 S&P 500 strategies: its own reference implementation, bt,
vectorbt, Backtrader and cvxportfolio. backtesting.py was not among them. It found that "all five engines agree exactly" at zero cost,
and that "all engines agree on the sign of every performance metric". As described, its strategies take no short positions and no
leverage, so they never lose more than the account holds; the paper does not discuss negative equity, margin calls or bankruptcy.
(This was checked against the paper's text on 16 Nov.)

The pilot takes one trade and puts it through two of the most-used engines, unmodified:
- backtesting.py 0.6.6 (9,243 GitHub stars) and backtrader 1.9.78 (23,748 stars);
- the trade: the largest position at leverage L, bought on bar 1 and held to bar 86;
- the market: fair placebo paths of 88 daily bars, with EURUSD magnitudes from gym-mtsim's bundle and fair signs, so no policy has an
  edge;
- the fee: 2 basis points per side.

Both engines are checked against two references computed independently: LEDGER (every loss booked) and FLOOR+STOP (equity set to 0
at the first close at or below 0).

### Results (2,000 paths per leverage; `engines_pilot_out.txt`, `engines_analyze_out.txt`, `engines_pilot.csv`)

| Leverage | Paths reaching zero | backtesting.py reported | backtesting.py, its own trade records | backtrader reported | Ledger |
|---|---|---|---|---|---|
| 1 | 0% | -0.04 | -0.04 | -0.04 | -0.04 |
| 5 | 0% | -0.18 | -0.18 | -0.18 | -0.18 |
| 20 | 8.7% | -0.23 | -0.56 | -0.74 | -0.74 |
| 50 | 48.9% | +1.84 | -3.70 | -1.84 | -1.84 |
| 100 | 70.5% | +8.39 [-0.41, 17.48] | -7.72 | -3.68 | -3.68 |
| 200 | 82.8% | **+26.01 [11.65, 40.93]** | -11.35 | **-7.37 [-33.31, 20.03]** | -7.37 |

Mean returns are in % of the starting cash, with 95% bootstrap intervals over paths.

- **Agreement.** At leverage 1, 2 and 5 the two engines agree on every path, exactly as 2603.20319 found. At 10, 0.1% of paths reach
  zero.
- **Divergence.**
  - From leverage 20 the engines diverge. backtesting.py reports equity floored at zero.
  - Its own trade list books the loss: the forgiven amount averages 0.33, 5.54, 16.10 and 37.36 points at leverage 20, 50, 100 and
    200.
  - backtrader equals the ledger to 1.4e-12 points.
  - From leverage 50, the mean reported return is positive in backtesting.py and negative in backtrader. On individual paths the signs
    differ in 8.2% (L 50) to 33.0% (L 200).
- **An optimizer of the mean reported return.** It picks leverage 200 in backtesting.py in 100% of 4,000 bootstrap resamples. In
  backtrader it picks leverage 1 in 70.2%; the rest is noise at high leverage, where the interval spans -33 to +20.
- **References.** backtesting.py matches the FLOOR+STOP rule in mean (26.01 against 26.05 at leverage 200). It differs on 13 of 8,000
  path-leverage cases, all paths that the reference leaves at -59% to -97%: backtesting.py's ruin test evidently includes the closing
  fee. The mean absolute deviation is 0.013 points.
- **A bug in this script, fixed before the run reported here.** A first 2,000-path run stalled for hours because strategy classes were
  defined inside the per-run functions (backtrader keeps a registry of them). It was stopped, the classes were moved to module level,
  and the run was repeated; a 40-path check gave the same values as the smoke tests.

### Census: the same trade in four engines (`census_mtsim.py`, `census_vbt.py`)

| Engine (stars) | Past the kink | Forgiven at leverage 200, points of starting cash |
|---|---|---|
| backtesting.py 0.6.6 (9,243) | Equity set to 0, the run stopped | 37.4 (2,000 paths) |
| gym-mtsim 2.0.0 as published (528) | Stop-out at margin level 0.2, then the balance floored at 0 and trading continues | 24.3 (300 paths; 5.0 at L 100, 0.6 at L 50) |
| gym-mtsim, floor lines deleted | Loss booked | 0 |
| backtrader 1.9.78 (23,748) | Loss booked; the account may stay negative | 0 (equal to the ledger to 1.4e-12) |
| vectorbt 1.0.0 (9,860) | No leverage in the free version (shorts beyond 1x cash are rejected at the defaults). A 1x short that loses 3x equity is carried as a value of -10,000 | 0 (by construction) |

Over the leverages that reach the kink (20 to 200), backtesting.py and backtrader rank the five levels in exactly opposite order
(Kendall tau = -1).

From documentation and forums only (these engines were not run):
- NautilusTrader: its margin account "allows negative balances (normal for margin trading)", and its cash account raises
  `AccountBalanceNegative` rather than borrow. It books the loss.
- zipline: there is no native margin model and cash can go negative. It books the loss.
- QuantConnect LEAN: a forum reply says the backtest stops when portfolio equity is negative at the end of a day, so it books the loss
  and then stops.
- MetaTrader 5's Strategy Tester stops on a margin call; whether it floors a negative balance was not confirmed.

Among the engines checked, backtesting.py is the one that replaces negative equity by zero.

### Real prices: the 14 series bundled with gym-mtsim (`real_series_witness.py`, `real_series_witness_out.txt`)

The same trade, long and short, ran on the 14 real daily series that gym-mtsim 2.0.0 ships: 7 forex pairs, 3 cryptocurrencies and 4
stocks, May to September 2021, 86 to 124 bars. That is 28 trades per leverage, each one path.

| Leverage | Trades reaching zero | Largest gap between the engines (points) | Trades with opposite signs |
|---|---|---|---|
| 1 | 0 of 28 | 0 (identical) | 0 |
| 2 | 1 | 4 | 0 |
| 5 | 4 | 164 | 1 |
| 10 | 9 | 428 | 2 |
| 20 | 11 | 956 | 3 |
| 200 | 26 | 10,691 | 12 |

- backtrader equals the ledger to 1.5e-11 points throughout.
- The extreme case is BCHUSD, long, at leverage 200: backtesting.py reports -100% where backtrader books -10,603%, a loss of 106 times
  the account.
- On real 2021 crypto paths the kink is reached already at leverage 2.
- Means over the 28 trades are not reported as evidence. Long and short cancel in the ledger, and at high leverage both sides ruin,
  so backtesting.py's absorption at -100% mixes with forgiveness.

### Scope: the setting of backtesting.py's own machine-learning tutorial (`tutorial_reach.py`, `tutorial_reach_out.txt`)

The library's tutorial "Trading with Machine Learning" backtests EURUSD long and short with `Backtest(data, ..., commission=.0002,
margin=.05)`, i.e. 20:1.
- **Setup:** 200 zero-skill strategies (random direction flips, mean holding 1 to 50 bars, the full margin-allowed size) on the
  library's bundled hourly EURUSD sample (5,000 bars, April 2017 to February 2018).
- **Result: null.** The strategies lost 93.5% on average to fees, but none reached zero equity, so the floor never bound. Each flip
  re-sizes to current equity, and no hourly bar moves the 5% that 20:1 needs to cross zero in one step. This is Proposition "The passive
  horizon", case (3), in the paper.
- **Where it binds:** the floor binds for leveraged multi-day holding (8.7% of placebo paths at 20x over 86 daily bars) and for jumpy
  assets (one real 2021 crypto trade at 2x). It does not bind in the tutorial's own hourly-forex setting.

### Theory: the forgiveness is a discretely monitored barrier's overshoot (`overshoot_bgk.py`, `overshoot_bgk_out.txt`)

A floor-and-stop engine checks for ruin only at bar closes and forgives how far below zero the close lands. That is the overshoot of a
discretely monitored barrier. For a Gaussian walk its mean is about 0.5826 x the step's standard deviation, the Broadie-Glasserman-Kou
(1997) continuity-correction constant -zeta(1/2)/sqrt(2 pi). The predicted forgiveness per path is therefore
P(ruin) x 0.5826 x L x sigma_bar, in units of the initial equity. It was checked on 20,000 paths of 88 bars, without fees:

| Bars | L | P(ruin) | Measured | Predicted | Ratio |
|---|---|---|---|---|---|
| Gaussian, sd 0.3% | 20 / 50 / 100 / 200 | 0.06 to 0.81 | 0.002 to 0.284 | 0.002 to 0.284 | 0.96 to 1.00 |
| EURUSD magnitudes, fair signs | 20 / 50 / 100 / 200 | 0.10 to 0.83 | 0.004 to 0.340 | 0.004 to 0.324 | 1.02 to 1.05 |
| 2021 crypto magnitudes, fair signs | 2 / 5 / 10 | 0.28 to 0.84 | 0.017 to 0.401 | 0.020 to 0.302 | 0.82 to 1.33 (heavy tails, steps close to the barrier distance) |

- The engine's forgiveness, the paper's credit for this rule, follows from leverage, bar volatility and the ruin probability, before
  any backtest.
- P(ruin) carries the same continuity correction, P(ruin) = 2 Phi(-(1/L + 0.5826 sigma)/(sigma sqrt(T))). Checked on 20,000 paths, it
  is accurate to 0.003 to 0.005 on Gaussian bars and to 0.001 to 0.007 on EURUSD magnitudes, at L 20 to 200 and T = 87.
- The whole credit is therefore 2 Phi(-(1/L + 0.5826 sigma)/(sigma sqrt(T))) x 0.5826 x L x sigma, a one-line certificate from classic
  barrier-option results.
- Under continuous monitoring there would be no overshoot. Forgiveness exists because ruin is checked at discrete closes.
- backtesting.py's pending fix for intrabar liquidation (PR #1441) would shrink it to the overshoot past each bar's extreme, but would
  not remove it. Only booking the negative equity does.

### A direction bias at the default settings (`direction_bias.py`, `direction_bias_out.txt`; no engine, no learner)

At backtesting.py's default margin (1, no leverage), a long can lose at most the account, because the price cannot go below zero. A
short can lose more, and the floor forgives the excess. On a fair market, the reported return therefore favours shorts. This was
checked with 1x positions held for 87 bars, on 20,000 fair paths per case.

| Market | 1x long, reported | 1x short, reported | Short, booked | Shorts reaching zero |
|---|---|---|---|---|
| EURUSD daily moves (x1 to x16) | about 0 | about 0 (short minus long between -0.30 and 0.00) | about 0 | 0 to 10.6% |
| BTC/ETH/BCH 2021 daily moves | -0.45 [SE 0.44] | **+1.17 [0.37]** | +0.01 | 13.0% |
| the same, x2 | -1.07 [1.14] | **+5.10 [0.56]** | +0.46 | 32.4% |

- On crypto-like volatility, the floor alone turns a zero-edge short into a reported profit. An optimizer, or an RL agent choosing a
  direction, should prefer shorts at the library's defaults: the accounting picks the direction.
- A random-policy check of the extended wrapper (150 episodes, crypto x2) agrees: backtesting.py's short reports +4.1% against -0.5%
  booked.
- The PPO test of this prediction is batch 3 (below).

### The paper's one-query reading on the engines (`reading_engines.py`, `reading_engines_out.txt`)

The setup: the largest position at leverage 200, no fees, then one next-bar move, either inside the range of ordinary moves (-0.2%)
or past the kink (-5%, a booked loss of 89,990 USD on 10,000).
- Inside the range, every engine reports the booked value.
- Past the kink, backtesting.py reports 0 (it forgives 89,990), backtrader reports -89,990, and vectorbt (a 1x short and a +300%
  move) reports -20,000. Both of the last two book the loss.
- **At backtesting.py's default settings (margin = 1, no leverage),** a 1x short whose price triples is reported at 0, while its own
  trade records book -19,997. It forgives a loss of twice the account. The floor therefore binds without any leverage, for short
  positions in assets that can more than double (squeezed stocks, crypto).

backtesting.py's public documentation does not mention the rule. The docstring of `Backtest` says only that "`margin` is the
required margin (ratio) of a leveraged account", with 0.02 suggested for 50:1; the rule is in a source comment ("If equity is
negative, set all to 0 and stop the simulation", `_Broker.next`, line 866 of 0.6.6).

**Prior art in backtesting.py's own tracker** (issues read on 16 Nov):
- **#939** (2023, open), with **PR #1441** (12 Nov 2026, open): liquidation is checked only at the close. A bar whose low wipes out
  equity is ignored if the close recovers; the example backtest "finishes at +18% even though intraday equity dropped to -8,000". The
  maintainer replied "Nice catch!". This is a second, intrabar forgiveness. The pilots above build High and Low from Open and Close,
  so they do not exercise it.
- **PR #1429** (24 Oct 2026, open): after a total loss, the annualized return is reported as 0% instead of -100%. That is a third
  forgiveness, in the statistics.
- **PR #1392** (Aug 2026) treats "force-closes every open Trade when equity drops to <= 0" as correct behaviour.
- No thread found says that setting negative equity to zero forgives the loss beyond it, which is the finding of this pilot.

One query past the widest move classifies each engine, and no number of queries inside the range separates them. This is
Proposition "Logs do not buy range" (`range_logs.tex`), applied to engine validation by comparison as 2603.20319 practises it.

Figure: `fig_engines.pdf` and `.png` (`fig_engines.py`). The left panel shows the mean reported return against leverage, with 95%
bootstrap bands, for backtesting.py reported, backtesting.py's own trade records, and backtrader (= ledger). The right panel shows the
share of paths that reach zero equity.

**Reading.**
- Agreement on trades that stay solvent does not validate an engine for optimization. The engines differ only past the kink, and an
  optimizer that maximizes the reported return goes exactly there. This is the paper's "range, not samples" on mainstream software.
- backtesting.py has 9,243 GitHub stars, against 528 for gym-mtsim.
- This is exploratory (one trade family, one market, two engines). It is not registered.

## PPO trained through the unmodified engines (`rl_engine_env.py`, `rl_engine_pilot.py`, `rl_engine_aggregate.py`)

The engine runs one backtest per episode in a worker thread. Its strategy's `next()` hands each bar's observation, and the engine's own
reported equity, to the environment, then waits for the agent's action. Every fill, fee, margin check and ruin rule is therefore the
engine's.
- Market: the pilot's fair placebo paths.
- Episodes start at an equity log-uniform in [0.05, 2] x 10,000 USD.
- Actions are target leverage in {0, 1, 4, 16, 64, 200}. The observation is equity / 10,000, current leverage / 200 and bars left / 88.
- The reward is the change in the engine's reported equity / 10,000. The agent never sees the ledger.
- Learner: the paper's standard PPO.
- Speed: 0.56 ms per step through backtesting.py and 1.24 through backtrader.

**Batch 1: seeds 1 to 4, 507,904 steps per run (`rl_engine_aggregate_batch1.txt`).**

| Engine | Seeds that choose 200x at half or more of the low-equity probes | What the others learned | Tests from 10,000 USD: reported - booked | From 1,000 USD |
|---|---|---|---|---|
| backtesting.py | 2 of 4 | flat or 1x | +8.53 [6.89, 10.27] | +18.91 [16.75, 21.21] |
| backtrader | 0 of 4 | 1x, 4x or 16x at every equity | 0 (identical) | 0 (identical) |

- backtesting.py seed 1 learned a gamble for resurrection:
  - At equity 0.05 it bets 200x at every point of the episode, and at 0.1 when at least half of the episode is left. From 0.25
    upward it stays flat.
  - From 1,000 USD, 80% of its test paths end at zero. The engine reports -100% on them; their median booked loss is -126.5% and the
    worst -300%.
  - Mean reported return +46.9% against +9.0% booked.
- Seed 4 bets 200x at every equity up to 10,000 USD: from 10,000 it reports +46.0% against +11.9% booked, with 72% of paths ending at
  zero.
- backtrader's agents never choose 200x. With no edge and one-time fees, moderate leverage costs them almost nothing.

**Batch 2: fresh seeds, the paper's full budget of 1,007,616 steps (`rl_engine_aggregate_batch12.txt`).**

| Engine | Seeds at 200x at half or more of the low-equity probes | The others |
|---|---|---|
| backtesting.py (seeds 5 to 10) | 2 of 6: seeds 5 and 7, at 200x up to equity 0.25 and flat or 1x above (a gamble for resurrection) | seed 9 partial (69x at 0.05); 16x, 16x and 1x |
| backtrader (seeds 5 and 6) | 0 of 2 | flat; 1x |

- From 1,000 USD, the backtesting.py agents reported +20.60% against -2.96% booked. Reported minus booked is +23.56 [21.59, 25.60] over
  1,200 test paths.
- backtrader's agents reported exactly what they booked.

**Batches 1 and 2 together.**
- backtesting.py: 4 of 10 seeds bet 200x near ruin, three of them only near ruin (seeds 1, 5, 7). backtrader: 0 of 6.
- At the seed level this is not significant (Fisher exact, one-sided p = 0.115). Reported minus booked is significant on test paths,
  in every backtesting.py group, and zero in every backtrader group.
- A registered version needs more seeds per engine (16 or more for 40% against 0%), or a design that learns more reliably.

**Batch 3: the direction chosen by the accounting** (`--levels -1 0 1 --lmax 1 --mags crypto_mags.npy --magscale 2 --tag _dir`).
Default settings: margin 1, no leverage, short / flat / long. Fair placebo with 2021 crypto magnitudes x2. Seeds 21 to 24, 507,904
steps.

| Engine | Seeds short at half or more of all probes | Seeds long at half or more | Reported - booked, tests from 10,000 USD |
|---|---|---|---|
| backtesting.py | 3 of 4 (s22 short everywhere; s23 61%; s24 78%, long only at equity 2); s21 flat | 0 of 4 | +2.07 [1.64, 2.56] |
| backtrader | 1 of 4 (s22) | 1 of 4 (s23, plus s24 long at 33%) | 0 |

- As predicted, backtesting.py's agents lean short and none goes long, while backtrader's split both ways, as a fair market leaves them.
- With four seeds per engine this is not significant. On a direction index (share short minus share long), a one-sided Mann-Whitney
  gives p = 0.17.

**Batch 4: two actions, flat or 200x** (`--levels 0 200 --tag _two`). Seeds 31 to 34, 507,904 steps.

| Engine | Seeds at 200x at half or more of the low-equity probes | Reported - booked, tests from 10,000 USD | From 1,000 USD |
|---|---|---|---|
| backtesting.py | 3 of 4 (s31, s32, s34 bet 200x at every equity) | +28.36 [25.89, 30.96] | +34.78 [32.18, 37.37] |
| backtrader | 0 of 4 (all flat) | 0 | 0 |

- The fourth backtesting.py seed (s33) is flat at the probes, but it gambles during episodes: from 1,000 USD it reports +9.3% against
  -16.4% booked, with 57% of paths ending at zero.
- Removing the intermediate leverages raised the share of seeds that learn the floor from 4 of 10 (six actions) to 3 of 4. On fair
  paths the mean reported return at 20x and 50x is about -0.2% and +1.8%, so intermediate levels sit in a valley that hides the payoff
  at 200x. This matches the learnability reading of the paper and Study DR's leverage-500 arm.
- The fully learned exploit bets the maximum at every equity, because the forgiveness is proportional to equity. "Only near ruin" in
  the six-action runs was partial learning.
- At the seed level: 3 of 4 against 0 of 4 (Fisher one-sided p = 0.071). Pooled over both designs, 7 of 14 against 0 of 10
  (p = 0.0099), but pooling is post hoc.
- A registered study should use the two-action design for power.

**Batch 5 predictions, written at 11:00 on 16 Nov, before batch 5 ran.** The dose-response design: backtesting.py, two actions {0, L},
L in {20, 50, 100}, seeds 41 to 44, 507,904 steps. L = 200 is batch 4.

- Predictor: the net edge of holding L for the whole episode, from the 2,000-path census (reported, fees included): -0.23 points at
  L 20, +1.84 at 50, +8.39 at 100, +26.01 at 200.
- Predicted: L 20 learns in 0 of 4 seeds; L 50 in at most 1 of 4; L 100 in at least 2 of 4. The share should rise with L.
- A learner bets L at half or more of the low-equity probes.

**Batch 5 results: 1 of 3 predictions supported.**

| L | Net edge (pts) | Predicted learners | Observed | Reported - booked, from 10,000 | From 1,000 |
|---|---|---|---|---|---|
| 20 | -0.23 | 0 of 4 | 2 of 4 (s41, s42 bet 20x up to equity 1) | +0.15 [0.09, 0.22] | +0.15 |
| 50 | +1.84 | at most 1 of 4 | 2 of 4 (s41 only near ruin, s42 up to 0.5) | +1.35 [1.03, 1.70] | +2.71 |
| 100 | +8.39 | at least 2 of 4 | 4 of 4 (s41 near ruin; s42 to s44 everywhere) | +11.33 [10.10, 12.60] | +15.11 |
| 200 (batch 4) | +26.01 | - | 3 of 4 | +28.36 | +34.78 |

- The forgiveness collected rises steeply with L, as the closed form says.
- The learner counts do not follow the net edge at low L.
- A post hoc explanation, untested: at L = 20 holding costs almost nothing (a 0.8% round-trip fee against a -0.23 net edge), so PPO may
  take the position by drift, as backtrader's agents held 16x in batch 1. Choosing the largest level then measures exploitation only
  where that level is costly without the floor.
- Batch 6 tests this with backtrader controls at the same levels.

**Batch 6 predictions, written before it ran.** backtrader, two actions {0, L}, L in {20, 50, 100}, seeds 41 to 44, 507,904 steps.
If the batch 5 "learners" at L 20 and 50 are drift, backtrader agents will also take L there in some seeds (at L 20 at least 1 of 4),
and fewer as L rises (at L 100, at most 1 of 4, since the round trip costs 4%).

**Batch 6 results: the drift explanation fails.**

| L | backtrader seeds holding L at low equity | backtesting.py (batches 4 and 5) |
|---|---|---|
| 20 | 0 of 4 (predicted at least 1: not supported) | 2 of 4 |
| 50 | 1 of 4 (s42 holds 50x everywhere) | 2 of 4 |
| 100 | 0 of 4 (s42 holds 100x only at equity 1 and 2) | 4 of 4 |
| 200 | 0 of 4 | 3 of 4 |
| all four | 1 of 16 | 11 of 16 |

- backtrader agents kept reported equal to booked throughout.
- The backtesting.py agents that took 20x and 50x were responding to the floor's small credit, +0.15 and +1.35 points, even where the
  net edge was slightly negative (-0.23 at L 20). Their counterparts in backtrader, which books the loss, stayed flat.
- So the sign of the net edge is too strict as a before-training predictor; this repeats Study DR's lesson. The contrast between
  engines at the same leverage is robust.
- Pooled over the four leverages: 11 of 16 against 1 of 16 (Fisher exact, one-sided p = 3.2e-4). The pooling was chosen after the
  results, so this is exploratory.
- The booked means are positive only by noise: at 200x a single path's return has a spread of hundreds of percent.

## Rerun

The environments were made in the session's scratch folder, and the versions are recorded here:
- `venv-vbt`, created with `--system-site-packages`: Python 3.10.0, backtesting 0.6.6, backtrader 1.9.78.123, vectorbt 1.0.0, SB3 2.7.0,
  torch 2.8.0+cpu, gymnasium 1.2.0, numpy 1.26.4.
- `venv-mtsim`: gym-mtsim 2.0.0, as in Study GC.

```powershell
venv-vbt\Scripts\python -B engines_pilot.py 2000          # cross-engine trade, 2,000 paths (writes engines_pilot.csv)
python engines_analyze.py                                   # bootstrap intervals, optimizer choice, signs
venv-mtsim\Scripts\python -B census_mtsim.py 300            # gym-mtsim as published and floor-deleted
venv-vbt\Scripts\python -B reading_engines.py               # the one-query reading
venv-vbt\Scripts\python -B real_series_witness.py           # the 14 real 2021 series
venv-vbt\Scripts\python -B tutorial_reach.py 200            # the ML tutorial's setting
python direction_bias.py                                    # long against short at the default margin
venv-vbt\Scripts\python -B rl_engine_pilot.py --steps 500000 --seeds 1 2 3 4                          # batch 1
venv-vbt\Scripts\python -B rl_engine_pilot.py --steps 1007616 --seeds 5 6 7 8 9 10 --engines btpy     # batch 2
venv-vbt\Scripts\python -B rl_engine_pilot.py --steps 1007616 --seeds 5 6 --engines bt                # batch 2
venv-vbt\Scripts\python -B rl_engine_pilot.py --steps 500000 --seeds 21 22 23 24 --levels -1 0 1 --lmax 1 --mags crypto_mags.npy --magscale 2 --tag _dir   # batch 3
python rl_engine_aggregate.py
python fig_engines.py
```

## A registered study it would support (proposal only, not started)

**Question.** Do mainstream backtesting engines that agree exactly on solvent trades disagree in sign once a trade's loss can exceed
equity? Does an optimizer of the reported return then pick opposite leverage in each? Does the paper's reading, one query past the
widest move made before any optimization, predict which engines forgive?

**Design.**
- Engines: backtesting.py, backtrader and vectorbt, with gym-mtsim and Gym-Trading-Env as the paper's RL references.
- Strategies: the conventional long-only set of 2603.20319 at leverage 1, and the same set with leverage L from 1 to 200 and with
  shorts.
- Markets: fair placebo paths, and real EURUSD and equity history.
- Optimizers: grid search, backtesting.py's own `optimize`, Optuna, and PPO through a step wrapper if that proves feasible.

**Predictions, fixed before running.**
- The reading classifies every engine correctly.
- The engines agree exactly wherever the computed reach is zero.
- The mean reported returns differ in sign from the leverage at which reach becomes material.
- The optimizer picks the largest leverage in forgiving engines and the smallest in engines that book the loss.

**Cost.** The backtests take minutes. PPO, if included, takes about a day.
