# How the monitoring law could enter the paper (draft for the author; the manuscript has not been changed)

## 1. Where it fits

`sections/options_theory.tex` (Section 3.3, "Accounting rules are options") states the ladder and Theorem 2 ("the floor grants up
to twice the subsidy of terminal limited liability... in a non-degenerate Brownian limit the ratio tends to 2"). The law gives the
amounts that the ladder orders. It fits as one proposition after Theorem 2, plus one figure or table. It can replace the sentence on
exact enumeration and part of the paragraph on the Nordic ratios, so the page cost is small. The paper is over EAAI's page limit, so
the proofs and the full test table belong in the supplement.

## 2. Draft proposition (additive ledger, the paper's model)

> **Proposition (the monitoring law).** Let the position be constant with exposure $q$, the costs zero, and the per-period P\&L
> $x_t=q s_t$ independent, symmetric and identically distributed with standard deviation $\varsigma$. Let $\rho$ be the mean overshoot
> of the walk $\sum x_t$ over a distant level ($\rho=0.5826\,\varsigma$ for Gaussian $x_t$, $\varsigma/2$ for moves of fixed size,
> and Siegmund's integral in general). Write $v=\varsigma\sqrt{T}/e$ for initial equity $e>0$. Then, up to terms that vanish with
> $\rho/(\varsigma\sqrt T)$,
>
> (i) $S_{\mathrm{LL}}=p\,\rho$ with $p=2\Phi\!\left(-(e+\rho)/(\varsigma\sqrt T)\right)$ the probability of closure;
>
> (ii) $S_{\mathrm{TERM}}=e\,[v\,\varphi(1/v)-\Phi(-1/v)]$;
>
> (iii) $S_{\mathrm{ZF}}=2\left[\varsigma\sqrt T\,\varphi(d)-(e+\rho)\,\Phi(-d)\right]$ with $d=(e+\rho)/(\varsigma\sqrt T)$; from zero
> equity, $S_{\mathrm{ZF}}/S_{\mathrm{TERM}}=2-\sqrt{2\pi}\,\rho/(\varsigma\sqrt T)$.
>
> **Corollary (time step).** If the same horizon is monitored $m$ times as often, $\rho$ is divided by $\sqrt m$ and
> $\varsigma\sqrt T$ is unchanged. Hence $S_{\mathrm{LL}}=O(m^{-1/2})\to0$, $S_{\mathrm{TERM}}$ does not move, and
> $S_{\mathrm{ZF}}\uparrow 2S_{\mathrm{TERM}}$. A finer time step removes what a capped-loss rule forgives and not what the zero floor
> forgives.

Proof sketch for the supplement:
- (i) is the renewal theorem for ladder heights with the corrected-diffusion approximation for the crossing probability (Siegmund
  1985, Chapter X).
- (ii) is the Bachelier formula.
- (iii) is the paper's reflection identity, $F_T=\max\{0,\max_t(-L_t)\}$, with the continuity correction for a discretely monitored
  maximum (Broadie, Glasserman and Kou 1997). From zero equity it is Spitzer's identity, which the paper already uses in Theorem 1(2).

The paper should say that these are classical tools; what is new is the application to accounting rules and the time-step contrast.

## 3. Draft sentences for the main text (exploratory wording)

> The ladder has closed forms. For a constant position the capped-loss rule forgives an overshoot, $p\rho$, and the zero floor a
> discretely monitored lookback, where $\rho$ is the mean overshoot of the per-period P\&L walk ($0.5826$ standard deviations for
> Gaussian moves). Refining the time step by $m$ divides $\rho$ by $\sqrt m$: the capped-loss subsidy vanishes and the zero floor's
> rises to exactly twice terminal limited liability. In unmodified backtesting.py the forms predicted the forgiven amount within
> 6\% in 18 of 18 cells (leverage 20 to 200, one to sixteen checks a day, with and without fees), and they give the 36.96 points by
> which nine trained agents of Study XE over-reported to within 0.6 points (37.47 predicted). On the twelve Nordic zones the formula
> $2-\sqrt{2\pi}\rho/\sigma_{\rm year}$ reproduced the 24 fee-free ratios of Table~\ref{tab:ladder12} within their Monte Carlo error.

Caveats to keep with these sentences:
- All tests are exploratory. The engine test was blind at the finer time steps (12 of its 18 cells); the Nordic form was written
  after a first-order formula had failed its accuracy criterion.
- For prices that compound, the forms use the log price (the additive forms were 12% high at leverage 20).

## 4. What it changes elsewhere

- **The abstract's "Two quantities computed before any training run decide it".** The law makes the two quantities explicit for the
  standard rules. It does not make them decide learning. A wording that matches the evidence: "Two quantities, computed before any
  training run, bound it: ... For the standard rules both have closed forms."
- **Theorem 2's last clause** ("in a non-degenerate Brownian limit the ratio tends to 2") gains its rate: $2-\sqrt{2\pi}\rho/(\varsigma\sqrt T)$.
- **The sentence "on real spreads with placebo signs the fee-free ratio was 1.62--1.99"** gains its explanation: the overshoot
  constant of hourly spreads ranges from 0.9 (NO4, 2024) to 12.9 standard deviations (SE1 after the reform), because the spreads are
  spiky.
- **References to add:** Siegmund (1985), Broadie, Glasserman and Kou (1997), and Balder, Brandl and Mahayni (2009) for gap risk under
  discrete trading.

## 5. Not recommended

- Do not claim that the law predicts which agents learn. It predicts how much a rule pays.
- Do not present the first-order formula in kappa: it failed.
