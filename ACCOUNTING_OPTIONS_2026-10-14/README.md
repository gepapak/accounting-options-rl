# Accounting rules as options: theory checks, historical settlement stress test and Studies N–R

This folder holds options-theory checks, completed settlement-lag, learnability, budget, learner-transfer and cross-cost studies N–R, and an exploratory historical Nordic collateral-component stress test. The paper and supplement state the mathematical assumptions and distinguish registered tests from exploratory analyses.

**Start with:**
- `THEORY_ACCOUNTING_OPTIONS.md`: statements and proofs, with exact checks in `verify_theory.py`;
- `FACTS_NORDIC_SETTLEMENT.md`: rule sources and modelling caveats;
- `study_n/`: the pre-registered Study N (settlement lag and terminal limited liability);
- `study_o/`: the pre-registered Study O (a learnability index computed before training; all four hypotheses supported);
- `study_p/`: Study P (3/4 registered hypotheses supported; shorter-budget count prediction failed);
- `study_q/`: internally registered Study Q (100 A2C/DQN runs, all six criteria met; DQN controls often gambled).
- `study_r/`: completed internally registered cross-cost test of an exact empirical one-step diagnostic; all 60 PPO agents included and the single composite R1 NOT SUPPORTED.
- `study_s/`: completed internally registered test of the sign-paired reward (180 PPO/A2C agents, 8/8 hypotheses supported): in an edge market it removed the exploit and kept the ledger-trained edge where a KL penalty failed at every strength; in Study O's environment it matched ledger training for PPO and A2C. `aggregate_s.py` gives the verdicts, `make_studyS_outputs.py` the figure; development and theory are in `../exploratory_claude_2026-10-16/nfl_training/`.

**Status and integrity.** N–R have internal hash-stamped protocols (`PREREGISTRATION_*.md` and `PREREGISTRATION_HASHES_*.txt`). These were not deposited with an independent registry. Run logs are append-only after `## After hashing`; the registered version is the preceding prefix. Preserve every registered script, verdict, raw result and `runs_*.csv` byte for byte. Run regeneration commands below only in a disposable copy. Q completed all 100 runs and was aggregated with its registered script unchanged; R automatically aggregated after all 60 planned agents completed. The settlement analysis, later baselines and independent audit are exploratory.

**Theory scope.** `THEORY_ACCOUNTING_OPTIONS.md` numbers results locally; use the named results in the current manuscript rather than carrying local numbers across versions. The ladder compares the five specified rules under common-path and closure assumptions. The factor-two bound concerns constant positions, symmetric independent increments and fee-free payoffs; adaptive policies need not obey it. The collateral result uses an external owner cost of risk and scales with open exposure.

**Verified study counts.** N supported 6/9 hypotheses, O 4/4, P 3/4, Q 6/6 and R 0/1 composite decision. O1 correctly classified 5/6 confident treatments, with α = 0.8 misclassified. The completed main-study cohort is 984 distinct agents, including R's 60 and excluding separate Study A's 280. C's snapshots, M's reused L checkpoints and O runs reused in P are counted once.

## Reproduce
Python 3.10, CPU. Run from this folder in a disposable copy; training takes hours. From the repository root, `python tools/verify_studies.py` reads frozen outputs without regeneration and checks registered hashes, completed counts and headlines.

**Theory checks**
```
python verify_theory.py         # exact checks: ladder ordering, factor two, doubling counterexample, non-identifiability
```

**Danish regimes** (official Energinet data in `../data/energinet/`, CC BY 4.0)
```
python build_panel.py           # -> panel.csv.gz (2019-11..2026-08, DK1/DK2)
python build_hourly.py          # -> panel_hourly.csv.gz (quarter-hours averaged to hours)
python lag_ladder.py --fee 0 --out ladder_fee0.csv   # factor two on real magnitudes (fee-free)
python ic_collateral_hourly.py  # modeled buffer under external owner risk cost -> ic_collateral_hourly.csv
python market_ladder.py         # forgiven loss per MW-year by rule and regime; collateral comparisons
python coverage.py              # monthly coverage of level-indexed collateral -> coverage_monthly.csv
python make_figures.py          # fig_ladder and fig_coverage (supplement, Danish regimes)
```

**Twelve Nordic zones** (eSett Open Data, not redistributed; download first)
```
python ../data/fetch_esett.py   # 4 March 2025 to 27 September 2026 -> ../data/esett/
python fetch_esett_pre.py       # 1 November 2021 to 4 March 2025 -> data_esett_pre/ (with manifest.json)
python ladder12.py              # the ladder per MW-year, 2024 vs the first post-reform year -> ladder12.csv
python coverage12.py 0.1        # historical volume-component stress diagnostic -> coverage12_fee0.1.csv, frontier12_fee0.1.csv
python coverage12.py 0.5        # sensitivity at a cost of risk of 0.50 EUR/MWh
python tails12.py               # spread distribution before and after the go-live -> tails12.csv
python make_fig12.py            # fig_coverage12 (the paper's coverage figure)
python make_tables.py           # tables/: the paper's ladder table and the supplement's settlement tables
```
eSett revises recent months, so a fresh download can differ slightly from the files used for the paper; `../data/esett/CONTENT_SHA256.json` and `data_esett_pre/manifest.json` record the versions used.

**Settlement interpretation.** The historical volume-term proxy falls below a modeled three-week buffer in 8.6% of pre-March-2025 FI/SE/NO zone-months and 36.8% afterwards. This ex-post component calculation assumes fair independent hourly payoff signs, open exposure equal to consumption, and an external owner cost of risk of 0.10 EUR/MWh (0.50 sensitivity). It omits the invoice-based component, minimum collateral, national exceptions and monitoring. Monthly price proxies and a buffer using the scored month are not a prospective collateral backtest. Finland's model changed in November 2024, so applying the old term later is counterfactual; Denmark's eSett comparison is also counterfactual. The figures identify neither actual total-collateral inadequacy, a causal reform effect nor participant conduct. A three-week settlement interval does not imply three weeks without intervention.

**Study N** (see `study_n/PREREGISTRATION_N.md`)
```
python study_n/portable.py aggregate_n.py         # verdicts from the shipped per-run results (study_n/results_n/jobs)
python study_n/portable.py make_studyN_outputs.py # the paper's Study N figure
python study_n/portable.py run_n.py --workers 7   # retrain all 40 runs (about 75 minutes on an 8-core CPU)
```
The registered files are kept byte-identical. `portable.py` adds `../LIABILITY_STUDY_2026-09-29` (the environments and utilities of Studies B–K) to the path and runs a script from `study_n/`. `tools/verify_studies.py` checks the registered hashes.

## Superseded scripts
- `dp_ladder.py`: the exploratory DP on short lags, stopped early. The gambling region is 1.26 / 1.27 / 1.29 allocations for lags of 0 / 4 / 16 quarter-hours, so a short lag barely moves the solvent gambling region.
- `ic_collateral.py`: the quarter-hour-sign version, superseded by the hourly version for cross-regime comparability.
- `coverage.py` evaluates the eSett formula on Danish prices as a counterfactual; the twelve-zone analysis is in `coverage12.py`.

**Study O** (see `study_o/PREREGISTRATION_O.md`)
```
python study_o/dp_o.py --zone DK1 ; python study_o/dp_o.py --zone DK2   # learnability index of every arm (before training)
python study_o/calibrate_o.py                     # threshold c* from Studies F, H, J, K
python study_o/run_o.py --workers 7               # 80 PPO runs (about 4 hours on an 8-core CPU)
python study_o/aggregate_o.py                     # verdicts O1-O4 -> VERDICTS_O.md, runs_o.csv
python study_o/make_studyO_outputs.py             # fig_studyO (the paper's Study O figure)
```

The index is 100 times the positive optimal leverage-16 versus flat gap, interpolated from seven probes and averaged on 400 log-spaced initial-equity reference points in [0.05,2], divided by binned one-period P&L noise. It is not measured training occupancy. Formula selection used earlier results; the 0.513% threshold was hashed before O. All four O decision rules were met; O1 permits one confident-arm error and scored 5/6. The standalone logistic c50 = 0.429% [0.302,0.573]% is exploratory. Pooled baseline AUC 0.963 includes calibration: prospective O AUC 0.915 is tied by local gap ratios at 2% and 10% equity, and 58/70 transfer accuracy is tied by the 10% gap. Dose alone (1−α) gives O AUC 0.919, an exploratory ranking comparison without a transferable cutoff; the shared α family does not establish that DP is needed.

**Study P** (see `study_p/PREREGISTRATION_P.md`)
```
python study_p/design_p.py                        # predictions at 250k and 4M steps from Study O's fit
python study_p/run_p.py --workers 7               # 90 PPO runs (about 6 hours; the 4M-step runs dominate)
python study_p/aggregate_p.py                     # verdicts P1-P4 -> VERDICTS_P.md, runs_p.csv
```

P adds 90 new agents; its 170-row CSV also contains O's 80 reused agents. The joint fit gives baseline c50 = 0.330% and β = −0.430 [−0.626,−0.215]. Its interval includes −1/2, a compatibility decision rather than proof of a learning law. The common-slope logistic model combines three budgets with unequal arm coverage. At α = 0.9, 8/10 agents at four million steps chose maximal leverage at the low-equity probe against 0/10 at one million. P3's one-sided Fisher p = 0.0018; P4's shorter-budget count test failed (p = 0.3029). Registered tests are unadjusted; sign draws are not additional agents, and two fixed zones do not establish population transfer.

**Study Q (complete).** `study_q/run_q.py` uses A2C-n64 and DQN-default, α in {0,0.5,0.8,0.9,1}, seeds 50–54 in DK1/DK2. All three criteria per learner were met. A2C/DQN correlations were 0.500/0.309 (registered permutation p = 0.0003/0.0149), full-forgiveness gaps 12.82/5.20, and α = 0.9 gaps 1.12/1.30. Q3 concerns the gap, not no gambling: DQN controls averaged 9.225 low-equity leverage (5/10 maximal), while α = 0.9 had 6/10 maximal. Exploratory c50 values were 0.538%/0.153%. Frozen outcomes are in `VERDICTS_Q.md` and `runs_q.csv`; `python study_q/aggregate_q.py` regenerates them only in a disposable copy.

**Study R (complete).** All 60 PPO agents finished under the frozen two-fee,
two-zone, alpha 0/.8/1 design with seeds 60–64. The automatic registered
aggregation returned **R1 NOT SUPPORTED**. The same-alpha .8 acquisition contrast
had to meet every gate:

| R1 gate | Result | Met? |
|---|---|---|
| G(.40)>=4 | 4.500 | Yes |
| G(1.50)<4 | 4.475 | No |
| Mean paired-seed interaction >=4 | .025 | No |
| Exact one-sided seed-cluster sign p<=.05 | .8125; 2/5 positive | No |

Alpha-0 anchors were maximal in 10/10 runs at both fees. Alpha-.8 mean leverages
were 7.600/5.600, against booked controls at 3.100/1.125 (1/10 and 0/10 maximal).
The descriptive paired-seed bootstrap interval is [-4.475,5.125]. It does not
replace the sign test; operational cohort thresholds do not prove equivalence,
no gambling or market-population transfer. This failed acquisition prediction
does not change the separate policy-specific accounting certificates and gives
no superiority claim over simple fee-aware scores. Full evidence is preserved
in `study_r/runs_r.csv`, `analysis_r.json`, `VERDICTS_R.md` and the frozen protocol
and predictions. Do not rerun its result writers over these completed files.

**Independent exploratory audit (from the repository root).**
```
python exploratory_codex_2026-10-15/statistical_audit.py --seed-fit-bootstrap 199
```
This independently recomputes the pre-Q cohort, O/P point estimates, Fisher tests and baselines, and adds explicitly exploratory seed/dependence sensitivity. `python exploratory_codex_2026-10-15/diagnose_completed_q.py`, also from the repository root, covers completed Q: its paired-seed Q2 intervals remain positive and common-arm permutation p = 0.025 for each learner. These are sensitivity analyses, not replacements for frozen verdicts. Training visitation and critic Q-values were not serialized and cannot be reconstructed. See `../exploratory_codex_2026-10-15/STATISTICAL_CHANGES.md`.

The completed DP-free fixed-policy rollout analysis is in `../exploratory_codex_2026-10-15/model_free/EXPLORATORY_RESULTS.md`, with reproduction commands in its README. Its primary surrogate had DP-index rank correlation 0.950 overall but 0.552 among legacy conditions, matched 58/70 O classifications with wide Monte Carlo uncertainty, and failed both 4M arm predictions. All fixed references and failures are retained. This is exploratory computational feasibility, not a confirmed scalable optimal-gap predictor; that method did not lead to a registered study at that stage. The later exact empirical one-step diagnostic's distinct cross-cost Study R is complete and R1 is not supported. Neither later work changes the earlier rollout failure.
