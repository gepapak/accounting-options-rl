# How the cross-engine finding could enter the paper (draft for the author; nothing has been changed in the manuscript)

All numbers below are from the exploratory pilots in this folder. Before they carry a claim in the paper, they should be either
reported as exploratory (the paper already has exploratory items: the settlement analysis and the engine-path checks) or replaced by a
registered Study XE (`PROPOSAL_STUDY_XE.md`).

## 0. UPDATE 16 Nov, 23:15: Study XE was registered and run (`../study_xe_2026-11-16/`, 4 of 7 supported)

Use only the registered claims in the paper:
- **XE1:** the engines agree exactly on every solvent trade.
- **XE4:** one query past the kink classifies every engine.
- **The forgiveness in XE2:** backtesting.py's reported return exceeds its own trade records by +37.8 points at 200x (lower bound
  +36.2).
- **XE5:** PPO learned the floor through backtesting.py in 12 of 16 seeds, against 1 of 16 through backtrader (p = 8.5e-5).
- **XE6:** learned agents report 27 to 31 points above booked.

Not supported, so the paper must not claim them, or must report them as failures:
- that the engines' reported means differ in sign at 200x on a fresh sample (XE2's difference; the sample drifted upward and backtrader
  itself reported +14.8%);
- that an optimizer avoids 200x in backtrader (XE3's backtrader half);
- that agents learn to short at the defaults (XE7).

The draft sentences below were written from the pilots. Replace their engine-difference numbers with the registered forgiveness, and
their PPO numbers with XE5 and XE6.

## 1. Where it fits

The section on where the conditions arise (`sections/practice.tex`, "Where the conditions arise") currently ends: "The hazard is
therefore specific to simulators with a capital account in which losses can exceed equity". It covers 18 public RL trading
environments and 21 electricity-RL environments. Mainstream backtesting engines are not covered. The new result fills that gap with
three to four sentences in the main text and one supplementary section, so the page cost is small. The paper is already over EAAI's
page limit, so the supplementary section is the safer home for tables.

## 2. Draft sentences for the main text (exploratory wording)

> The same rule sits in mainstream backtesting software. backtesting.py (9,243 GitHub stars) sets negative equity to zero and stops the
> run, while backtrader (23,748), vectorbt, zipline and NautilusTrader book the loss. On 2,000 fair placebo paths the two engines
> agreed exactly on every trade up to leverage 5. At leverage 200 backtesting.py reported +26.0% [11.7, 40.9] where backtrader and an
> independent ledger reported -7.4%, and a search over leverage picked the largest level in backtesting.py in every bootstrap resample.
> A recent comparison of five engines found that they "agree on the sign of every performance metric" (Yin et al., 2026). Its
> strategies take no short positions and no leverage, so their logs never reach the kink; backtesting.py was not among its engines. This is the paper's range statement on engine validation: one query past the widest
> move classified each engine.

Add one sentence on PPO trained through the engines, once the pilot is complete (see section 4).

## 3. Related work

Cite arXiv 2603.20319 (Yin, Miki, Lesnichenko and Gural, "Implementation Risk in Portfolio Backtesting", March 2026). It is the
closest engine-comparison study and the natural contrast: its strategies are unlevered and long-only, so its agreement is guaranteed by
Proposition "Logs do not buy range".

backtesting.py's own issue tracker documents two related forgivenesses, intrabar liquidation (#939, PR #1441) and the annualized return
after a total loss (PR #1429). They can be cited as software records or left out.

## 3b. A remark the theory section could carry (draft)

> **Remark (engine comparison inside the range).** Let two engines agree on every state with positive equity: the same fills, fees
> and marks. Then on any path where a strategy's equity stays positive they report the same result, and the expected difference in
> reported return equals the expected loss one of them forgives at the first close with equity at or below zero.
>
> For a family of strategies that cannot lose more than the account (long-only and unlevered), that difference is zero on every path.
> A comparison on such a family therefore cannot separate the engines, whatever its size. This is Proposition "Logs do not buy range",
> with the strategy family in the role of the log. One leveraged position held past the widest move, or one unlevered short of an
> asset that more than doubles, separates them.

This explains, rather than disputes, the finding of 2603.20319. As described there, its 15 strategies take no short positions and no
leverage (for example, "allocates 95% of the portfolio to a single asset"); the paper does not say "long-only" in so many words. Its
five engines were its own reference implementation, bt, vectorbt, Backtrader and cvxportfolio, so backtesting.py was not among them.
It does not discuss negative equity, margin calls or bankruptcy; its only margin event is Backtrader rejecting an order when running
cash falls below zero.

## 4. PPO through the engines (exploratory pilots complete, batches 1 to 6)

Draft sentence, exploratory wording:

> Trained through the unmodified engines on the same fair market, PPO learned to bet maximal leverage in backtesting.py in 11 of 16
> seeds across leverages 20 to 200, and in backtrader in 1 of 16. From 1,000 USD, pooled over seeds, the backtesting.py-trained agents
> reported +34.2% against +19.1% booked at leverage 100, and +37.5% against +2.7% at leverage 200; backtrader's agents reported exactly
> what they booked. At the library's default settings without leverage, 3 of 4 agents learned to sell short, the only direction the
> floor forgives.

If Study XE is registered and run, replace these numbers with its XE5 to XE7 results.

## 5. What it does not claim

- No claim that any user's published backtest is wrong.
- No claim about live trading.
- The pilots use fair placebo paths and 14 real 2021 series bundled with gym-mtsim.
- Forgiveness in backtesting.py binds only when a loss can exceed equity. That happens through leverage (margin < 1), or through
  shorts even at the default margin = 1: a 1x short whose price triples is reported at 0 against -19,997 booked.
- It does not bind in the library's own ML tutorial setting (20:1, hourly EURUSD). There, 200 zero-skill strategies lost to fees but
  never reached zero (`tutorial_reach.py`).
