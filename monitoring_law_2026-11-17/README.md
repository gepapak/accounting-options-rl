# The monitoring law: what three accounting rules forgive, in closed form (exploratory, 17 November 2026)

**Not registered (exploratory, with predictions written before each test).** Written in an AI-assisted session (Claude Fable 5.2 in
Claude Code) at the author's request to keep researching for a stronger contribution. Reported in the paper (Section 3.4, Figure 2 and
the supplement) and copied byte for byte into the code repository on 17 and 18 November 2026; on 18 November the data paths of
`test_d_existing.py` and `test_d4b_nordic.py` were made to work in the repository layout as well, with no change to any computation. Every test is listed with its outcome, including the two that
failed their pre-stated criteria.

## The result in one paragraph

The paper's theory orders the accounting rules (the ladder) and bounds the zero floor by twice terminal limited liability. This note
gives the amounts. For a constant position, the loss a rule forgives is a classical quantity of the bar-move random walk: an overshoot
for the capped-loss rule, a discretely monitored lookback for the zero floor, and a put for terminal limited liability. The formulas
need only leverage, bar volatility, horizon, time step and fee. They predicted what an unmodified mainstream backtesting engine
forgives within 6% in 18 of 18 cells, 12 of them fully blind (finer time steps, never run before), and they predict what the trained
PPO agents of the registered Study XE over-report within 2%. They also give a new qualitative statement: as the simulator's time step shrinks, the capped-loss rule forgives less and less
(like the square root of the step), while the zero floor forgives more, up to exactly twice terminal limited liability.

## The law

Setting: the paper's Section 3. A constant position with notional L times the initial equity, a market without edge, n bars whose
moves have standard deviation s. All amounts are in units of the initial equity. rho is the mean overshoot of the bar-move walk over a
far barrier: 0.5826 s for Gaussian bars (the constant -zeta(1/2)/sqrt(2 pi) of Siegmund 1985 and Broadie, Glasserman and Kou 1997),
0.5 s for fixed-size moves, and Siegmund's integral over the characteristic function in general (`law.rho_siegmund`).

| Rule | What it forgives | Closed form (additive profit and loss) |
|---|---|---|
| Capped loss, the account closes (LL; backtesting.py) | the overshoot at the ruin bar, and the exit fee | P(ruin) x (L rho + fee x (L - 1)), with P(ruin) = 2 Phi(-(1/L - fee + rho) / (s sqrt(n))) |
| Terminal limited liability (TERM) | the final deficit | v phi(1/v) - Phi(-1/v), with v = L s sqrt(n) |
| Zero floor, trading continues (ZF) | the deepest point of the ledger below zero | 2 [v phi(d) - (1 + L rho) Phi(-d)], with d = (1 + L rho) / v |

Consequences:
- **Time step.** Checking ruin m times per bar divides rho by sqrt(m) and leaves v unchanged. So LL falls like 1/sqrt(m) towards the
  forgiven exit fee, TERM does not move, and ZF rises to 2 x TERM. The two floor rules, which differ only in whether trading continues,
  sit at opposite ends in the fine-step limit.
- **From zero equity.** ZF / TERM = 2 - sqrt(2 pi) rho / (s sqrt(n)). This is the gap to the factor two of the paper's Theorem 2.
- **Fees.** A capped-loss engine also forgives the exit commission of the ruin bar, which finer time steps do not remove.
- **Compounded prices.** For a long position in a price that compounds its bar returns, the same law holds with the log price as the
  walk (`law.ln_forms`). The difference matters when ruin needs a move of several percent (leverage 20 and below here).

## Tests and outcomes

Predictions were written before the corresponding tests ran: `PREDICTIONS.md` at 01:47:05 (Test A's output is from 01:48:33) and
`PREDICTIONS_v2.md` at 01:50:40 (the engine test started at 01:51 and finished at 02:09; Test A2's output is from 01:55:52). The
prediction files carry their own timestamps.

| Test | What | Blind? | Criterion | Outcome |
|---|---|---|---|---|
| A | Additive forms against simulated compounded prices: three rules, leverage 20/50/200, 1 to 64 checks per day, 200,000 paths | yes for finer steps, TERM and ZF | every cell within 5%; LL slope against m between -0.56 and -0.42; ZF/TERM rising to 2 | **Not met.** Slopes -0.47 to -0.49 and ZF/TERM within 1%, as predicted. But at leverage 20 all three rules were 12% high, TERM included: the additive forms ignore compounding. Leverage 50 and 200 were within 3.1%. |
| B | The same forms on empirical daily bars (EURUSD, 2021 crypto), capped loss | no (values seen on 16 Nov) | EURUSD within 8%, crypto within 25% | **Not met**: EURUSD at leverage 20 was 9% high and crypto at leverage 2 was 2 times too high (compounding again). The other five cells were within 2%. |
| A2 | Additive forms on additive profit and loss, the paper's own ledger model | **yes** | every cell within 5% | **Met.** All 36 comparisons within 1.8%. |
| A3, B3 | Compounded-price forms on the cells of A and B | no (post hoc) | as A and B | Met: Gaussian within 1.8%, EURUSD within 1.1%, crypto within 14%. |
| C | **Unmodified backtesting.py 0.6.6**, 8,000 fine paths given to the engine as bars 1, 4 or 16 times a day; leverage 20/50/200; fee 0 and 2 bp | **yes** for 4 and 16 checks a day (12 cells: the engine had never been run at finer steps); the 6 daily cells resemble the censuses of 16 November, whose values were known | every cell within 10% or inside the 95% interval; ruin share within 0.03; fee-free slope between -0.56 and -0.42 | **Met.** All 18 cells within 6% (ratios 0.94 to 1.02); ruin shares within 0.008; slopes -0.465 to -0.484. The forgiven exit fee shows as predicted: 11.78 with fees against 8.31 without at leverage 200 and 16 checks a day. |
| D1 | The registered Study XE census and the 16 Nov pilot census | no (values known) | within 10% (25% at leverage 20) | Met: 37.47 predicted against 37.82 (XE) and 37.36 (pilot) at leverage 200; within 1 to 5% at 20, 50 and 100 in XE. The pilot at leverage 20 is 16% below the prediction (0.330 against 0.394), inside its 25% criterion. |
| D2 | What Study XE's trained PPO agents over-report | no | none set | The nine agents that bet 200x at every probe report 36.96 points above booked (range 36.46 to 37.03); the law gives 37.47. Their ruin share is 0.833 against 0.828. In the 16 November pilots (`test_d2b_pilot_agents_out.txt`), three agents that bet 100x everywhere report 15.11 above booked against 15.91, and two at 200x report 37.81 against 37.47; each group shares one bank of 200 test paths. The same-engine control run after this note (`../xe_patch_control_2026-11-17/`, test bank never seen before) gives 36.48 (standard error 2.72) against 37.47, with a ruin share of 0.825 against 0.828. |
| D3 | gym-mtsim 2.0.0 as published: a stop-out at margin level 0.2, then the floor | no | none set | Roughly right: 0.58, 5.32 and 20.24 predicted at leverage 50, 100 and 200, against 0.40 to 0.62, 4.79 to 4.96 and 21.8 to 24.3. At 200 the pooled value is 23.07 [20.50, 25.77], so the prediction sits just below the interval. |
| D4 | The paper's Nordic ladder, 24 zone-years: ZF/TERM against the first-order formula 2 - 1.2533 kappa | yes | Spearman at least 0.8 and mean absolute error at most 0.05 | **Not met.** Spearman 0.963, but the mean absolute error is 0.126: the first-order formula is far too low for spiky years (it assumes moves of similar size). |
| D4b | The same ratios against 2 - sqrt(2 pi) rho / sigma_year, with rho from each zone-year's hourly spreads | no (post hoc) | none set | All 24 zone-years within two standard errors; mean absolute error 0.017 against a Monte Carlo error of 0.022; Spearman 0.972. |

Figure: `fig_monitoring_law.png` shows (a) the engine against the law at three time steps, (b) the zero floor rising to twice
terminal limited liability, and (c) the 24 Nordic zone-years.

## What this adds, and what it does not

- **Adds.**
  - Amounts where the paper has an ordering and a bound: the "credit" of each rule and the "reach" (the ruin probability) in closed
    form, checked blind in a mainstream engine.
  - A new statement about simulator design: refining the time step removes the capped-loss subsidy but not the zero floor's.
  - An account of numbers the paper already reports: the Nordic ratios of 1.62 to 1.99, Study XE's 37.8 points, and the trained
    agents' over-reporting.
- **Does not add.**
  - A new phenomenon or a new training method.
  - A predictor of when agents learn. The law gives how much a rule pays; the learning predictors tried so far (Study DR, batches 5
    and 6 of the 16 November pilots) failed.
- **Prior art.** The constant and the correction are classical (Siegmund 1985; Broadie, Glasserman and Kou 1997), and gap risk under
  discrete trading is known in finance (Balder, Brandl and Mahayni 2009, for portfolio insurance). The searches on 17 November found no
  work that applies them to the accounting rules of simulators, to the time-step contrast between the two floor rules, or to what
  trained agents over-report. The paper should present the law as a corollary of classical results.

## Files

| File | What it is |
|---|---|
| `law.py` | The closed forms, Siegmund's integral and the Monte Carlo cross-checks |
| `predict.py`, `predictions.json`, `PREDICTIONS.md` | First set of predictions (additive forms) |
| `predict_v2.py`, `predictions_v2.json`, `PREDICTIONS_v2.md` | Second set (compounded prices), written after Tests A and B were scored |
| `test_a_sim.py`, `test_a2_sim.py` | Simulation tests A, B, A2, A3, B3, with their `_out.txt` and `.json` |
| `test_c_engine.py`, `test_c_engine.csv` | The blind engine test |
| `test_d_existing.py`, `test_d4b_nordic.py` | Tests on existing results (D1 to D4, D4b) |
| `fig_law.py`, `fig_monitoring_law.pdf/.png` | The figure |

## Rerun

```powershell
python -B predict.py
python -B test_a_sim.py
python -B predict_v2.py
python -B test_a2_sim.py
venv-vbt\Scripts\python -B test_c_engine.py 8000 6        # about 18 minutes
python -B test_d_existing.py
python -B test_d4b_nordic.py
python -B fig_law.py
```
