# Accounting rules are options: theory note (exploratory, 14 Oct 2026)

Status: new results developed after Studies A–M. Not pre-registered. Exposition corrected on 15 October 2026 to match the manuscript's integrability, signed-payoff, scale-order and external-cost assumptions. This note is not listed in a registered hash or manifest; Study N's frozen protocol cites its question, but does not hash this file. Registered records and numerical outputs are unchanged. Exact numerical checks are in `verify_theory.py`, whose existing output is listed at the end.

## Setting

The setting is the no-edge market defined in the main paper and supplement.

- **Horizon and ledger.** Horizon T. Booked ledger L_t = e + Σ_{u<t} x_u, with initial equity e ≥ 0.
- **Increment.** x_t = q_t ε_t a_t − C_t.
  - The position q_t ≥ 0 is chosen before the sign (predictable, i.e. non-anticipating).
  - The magnitude is a_t ≥ 0. The cost is C_t ≥ 0 and is F_t-measurable.
  - ε_t is a fair ±1 sign, independent of F_t and of (a_t, C_t).
- **Martingale property.** L is a supermartingale, and a martingale when there are no costs.
- **Integrability and common process.** The finite horizon and admissible positions satisfy E[Σ_t(q_t a_t + C_t)] < ∞. Compare rules on one common predictable position process, extended to T when a rule closes earlier. Suprema use the same admissible class; this is not a pathwise comparison of separately trained policies.
- **Deficit.** D_t = (−L_t)^+ is the part of the booked loss that exceeds the initial equity.

An *accounting rule* decides what happens after the ledger first falls to zero. Write σ = inf{t ≥ 1 : L_t ≤ 0} ∧ T for that first-passage time. The table gives each rule's closing time τ. Its *subsidy* is the expected forgiven loss, S = E[D_τ]. For the zero floor it is the regulator F_T instead.

| Rule | What happens | Closing time τ | Owner's payoff | Real-world counterpart |
|---|---|---|---|---|
| FL, full liability | every loss is booked | σ | L_σ | real-time margining with recourse |
| LL, capped loss | sleeve closes at zero; the overshoot is forgiven | σ | L_σ^+ | "episode ends at bankruptcy" |
| LAG_W, settlement lag W | breach detected W periods later; trading continues ("zombie window") | (σ+W) ∧ T | L_τ^+ | Nordic imbalance settlement: invoiced and settled ~3 weeks after delivery |
| TERM, terminal limited liability | losses are capped only at the end | T | L_T^+ | Merton (1974): equity is a call on the firm |
| ZF, zero floor | reported equity is reflected at zero; trading continues | T, with subsidy F_T = max_{t≤T} D_t | E_T = L_T + F_T | the audited engine; gym-mtsim (capped variant) |

## Theorem 1 (the ladder)

For every predictable position process, every cost process C ≥ 0 and all 0 ≤ W ≤ W′:

  0 = S_FL ≤ S_LL ≤ S_{LAG,W} ≤ S_{LAG,W′} ≤ S_TERM ≤ S_ZF.

Without costs, the owner's expected payoffs are ordered in the same way. Full liability pays the signed L_σ, not its positive part: E[L_σ] = e. Limited-liability rules pay L_τ^+, with expectation e + E[D_τ], and the zero floor pays L_T + F_T, with expectation e + E[F_T]. Taking suprema over a common admissible class preserves the order. With costs the subsidy ordering remains; differing accumulated costs do not generally order total payoffs.

*Proof.*
- −L is a submartingale, and y ↦ y^+ is convex and non-decreasing. Hence D = (−L)^+ is a submartingale (Jensen).
- The closing times satisfy σ ≤ (σ+W)∧T ≤ (σ+W′)∧T ≤ T, and all are bounded. Optional sampling gives E[D_σ] ≤ E[D_{(σ+W)∧T}] ≤ E[D_{(σ+W′)∧T}] ≤ E[D_T].
- Finally, D_T ≤ max_{t≤T} D_t = F_T pathwise (the paper's reflection identity).
- Without costs, optional sampling gives E[L_τ] = e and L_τ^+ = L_τ + D_τ. Together with E[L_σ] = e for signed full liability and E[L_T + F_T] = e + E[F_T], this orders each rule's actual owner payoff. ∎

The rules differ only after σ, and the deficit up to τ depends only on positions taken before τ. "The same policy" therefore means the same position process up to each rule's closing time.

## Theorem 2 (the floor grants a lookback, up to twice Merton's call)

**Statement.** Take fixed (deterministic) positions, no costs, and an integrable magnitude vector independent of the entire fair, independent sign sequence. Conditional on that vector the increments are independent and symmetric. Then for every initial equity e ≥ 0:

  S_TERM ≤ S_ZF ≤ 2 S_TERM.

**Diffusion limit.** When no single increment matters relative to √(Σ q²a²), the ratio S_ZF / S_TERM tends to 2 for every e ≥ 0.

**Interpretation.**
- Under terminal limited liability the owner holds a European call on the ledger: Merton's equity-as-a-call.
- Under the zero floor the owner holds the lookback claim described by the paper's reflection and best-moment results, under their stated position and increment assumptions.
- In the diffusion limit the simulator therefore grants exactly twice the option that bankruptcy law grants a shareholder.

*Proof.*
- Y_t = −(L_t − e) is a sum of independent, symmetric increments.
- D_T = (Y_T − e)^+ and F_T = (max_{t≤T} Y_t − e)^+.
- Lévy's inequality gives P(max_t Y_t ≥ y) ≤ 2 P(Y_T ≥ y) for y ≥ 0.
- Integrating over y ≥ e gives E[(max Y − e)^+] = ∫_e^∞ P(max Y > y) dy ≤ 2 ∫_e^∞ P(Y_T > y) dy = 2 E[(Y_T − e)^+].
- In the Brownian limit the reflection principle turns the inequality into an equality for y ≥ 0. ∎

**The bound fails for adaptive policies.** A doubling ("martingale") policy stakes 2^{k−1} at round k until its first win. Its booked game is fair: expected terminal ledger = e.
- Under terminal limited liability its subsidy is (2^T − 1 − e) / 2^T < 1.
- Under the zero floor its subsidy is about T/2. Each deficit along the way is forgiven, and the recovering win is then credited in full.
- So the ratio grows without bound, and the expected reported profit of the textbook way to lose money grows linearly in the number of rounds.
- With a position cap, the supplementary variance bound for predictable positions caps every admissible policy, including this one.

## Proposition 3 (settlement lag, diffusion limit)

**Setting.** The ledger is a Brownian motion with volatility s per unit time, no costs, and initial equity e ≥ 0. Let σ_e be the time the ledger first hits zero.

**Statement.**

  S_{LAG,W} = E[ s √(min(W, T − σ_e)) / √(2π) ; σ_e < T ].

For W ≪ T this is approximately 2 Φ̄(e / (s√T)) · s √(W / (2π)).

So the forgiven loss grows like the square root of the detection lag. It vanishes in the limit of instant detection (LL; any remaining subsidy then comes only from jump overshoots) and approaches TERM as W reaches T.

*Proof.* Apply the strong Markov property at σ_e, and use E[(−B_u)^+] = s√u / √(2π). ∎

## Proposition 3b (the zombie window is a call on the window's P&L)

**Setting.** Under LAG_W, take a participant that has breached (ledger n < 0) with k periods left before detection. Its owner's claim at detection is max(n + S, 0), where S is the P&L over the remaining window. Only this claim matters to the owner.

**Statement.** The claim is a call on S with strike −n > 0.
- **Without costs.** Call value is non-decreasing under convex order, not under arbitrary variance ordering. In a symmetric no-edge scale family S = sZ, with integrable mean-zero symmetric Z, increasing s is a mean-preserving spread and weakly raises the call value. This supports maximal exposure when admissible actions change only that scale; strict increase additionally needs mass crossing the strike. It is not a theorem for arbitrary position-dependent payoff distributions.
- **With costs.** Ledger costs are inside S and are forgiven when the claim ends at zero. There is no general monotonicity guarantee with changing costs or impact. The dynamic program for the specific Study N sleeve (`study_n/dp_n2.py`, fee 1.0 EUR/MWh, W = 96) finds maximal leverage at every insolvent probe; this is a model-specific numerical result.
- **Booked control.** If the deficit is booked, the owner's terminal value is n + S, its expectation falls with costs, and staying flat is optimal.

**Reward timing.** Rewarding the change in the claim max(L, 0), or paying the forgiveness as a lump sum at detection, gives the same undiscounted return. With discounting, the two differ only in timing.

## Proposition 4 (incentive-compatible collateral is proportional to exposure)

**Setting.**
- A participant has buffer b: equity including posted collateral.
- It holds exposure q over an unsettled window of length W.
- Y_W is integrable gross zero-mean window P&L per unit of exposure, and c ≥ 0 is an external owner cost of risk per unit of exposure and time. The external charge cqW is non-forgivable and paid outside the limited-liability ledger; it is not a transaction fee booked inside Y_W.

**Gambling incentive.** The one-window incentive is

  G(b, q) = q [ E[(−b/q − Y_W)^+] − cW ].

G ≤ 0 for all q ≤ q̄ if and only if b ≥ κ_W q̄, where

  κ_W = inf{κ ≥ 0 : E[(−κ − Y_W)^+] ≤ cW}.

**Consequences.**
- *Fixed or lagged collateral can fail.* When 0 < κ_W < ∞, a fixed component C or one based only on exposure observed with a lag of at least W leaves G > 0 for any admissible current q > C / κ_W. This requires that such exposure is allowed; other constraints can prevent it.
- *Level-indexed collateral fails across regimes.* Take collateral proportional to a price level, C = m · V · P̄. Scaling the window P&L by λ ≥ 1 gives κ_W(λY; c) = λ κ_W(Y; c/λ) ≥ λ κ_W(Y; c), because κ_W falls as the cost rises. So when spread magnitudes rise by λ while the price level stays fixed, the coverage C / (κ_W q) falls by at least λ. No fixed multiplier m makes such a rule incentive-compatible in every volatility regime.
- *Zero external cost depends on support.* With c = 0, κ_W = ∞ only when the downside of Y_W is unbounded, because every finite κ leaves positive expected shortfall. If downside is bounded, κ_W = ess sup(Y_W^−) is finite; for Y_W equally likely ±1, κ_W = 1. The unbounded-support qualification is essential.

*Proof.* For q > 0 the displayed G is homogeneous of degree one in (b, q). The shortfall f(κ) = E[(−κ − Y)^+] is continuous and non-increasing, and tends to zero by integrability. For 0 < q ≤ q̄, b/q ≥ b/q̄, so the largest incentive per unit exposure is at q̄, giving the stated threshold with the usual extended-value convention. At c = 0, f reaches zero exactly when κ covers the essential downside bound. ∎

**Cost-location check.** With b = 0, Y equally likely ±1 and cW = 0.1, an external owner hurdle gives E[Y^+] − 0.1 = 0.4; an inside-ledger fee gives E[(Y − 0.1)^+] = 0.45. The coverage scripts use the external-hurdle definition. They are historical collateral-component stress calculations, not a claim about complete actual national collateral requirements.

## Proposition 5 (reported equity curves cannot identify booked performance)

**Statement.** Under the zero floor, the law of the reported equity path does not identify the expected booked P&L.

**Example.** Take one period from zero equity, with gain g = 10 with probability ½ and loss ℓ with probability ½.
- The reported law is {10 w.p. ½, 0 w.p. ½} for every ℓ.
- The booked mean is 5 − ℓ/2. It is +3 for ℓ = 4 and −10 for ℓ = 30.
- This extends to any horizon: a policy that trades only at zero equity hides every loss magnitude taken there.

**Consequences.**
- Every functional of reported paths is equal for the two cases, so none can rank them by booked performance. This includes the mean, median, IQM, Sharpe and Sortino ratios, maximum drawdown, Calmar ratio, win rate, and VaR or CVaR of reported returns.
- Adding the injection series identifies the ledger pathwise, by L = E − F (the paper's reflection identity).
- Alternatively, a known randomized no-edge model identifies the expected booked P&L from costs. This is the basis of the paper's cost-based certification result.
- Hence additional information beyond reported paths is necessary; injection reconciliation and a known randomized model are two sufficient routes.

## Checks (`python verify_theory.py`)

- **Ordering (Theorem 1).** Exact enumeration over all 2^12 sign paths, for 300 random markets and policies: fixed schedules, equity-dependent policies and loss-chasing policies, with and without fees. 0 violations.
- **Factor two, fixed positions, no fee (Theorem 2).** On random heavy-tailed magnitudes the maximum S_ZF/S_TERM is 1.69. For Gaussian increments the ratio is 1.87 at n = 100, 1.94 at n = 1,000, and 1.99–2.00 at n = 10,000.
- **Doubling policy.** S_ZF/S_TERM = 2.82, 3.77, 4.76 and 5.75 at T = 6, 8, 10 and 12, i.e. ≈ T/2.
- **Real Danish magnitudes, no fee, 1-year horizon.** S_ZF/S_TERM = 1.73–1.98 (`lag_ladder.py --fee 0`). Two 20,000-draw checks give 1.93 [1.91, 1.96] and 1.95 [1.92, 1.98].
- **Twelve Nordic zones, no fee, 2024 and the first post-reform year (24 zone-years, `ladder12.py`).** S_ZF/S_TERM = 1.62–1.99; all 24 are below the bound of 2.
- **Non-identifiability (Proposition 5).** The one-period example gives identical reported laws with booked means +3.0 and −10.0.

## Novelty boundary

**Classical results the note uses:**
- Lévy's inequality and the reflection principle;
- equity as a call option (Merton 1974);
- lookback options (Goldman, Sosin and Gatto 1979);
- optional sampling.

**What the note adds:**
- ordering the accounting conventions of RL trading simulators by the option each one grants;
- the sharp factor two between the simulator floor and terminal limited liability, and its failure for adaptive (loss-chasing) policies;
- placing real settlement rules on the same ladder;
- the level-versus-volatility incompatibility of level-indexed collateral.
