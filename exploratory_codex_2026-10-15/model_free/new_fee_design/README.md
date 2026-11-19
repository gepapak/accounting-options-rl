# Exploratory new-fee design feasibility

This design-only check was specified after the existing model-free analysis, before executing new-fee simulations. It is **not Study R**, does not register hypotheses and launches no learner training. It never reads learner outcomes or DP outputs and never solves a DP.

Frozen inputs: DK1/DK2 empirical TRAIN payoffs, fees .4 and 1.5 EUR/MWh, standard 25% cap, no impact, gamma .99, fixed threshold-1 continuation, alpha grid `{0,.25,.5,.7,.8,.9,.95,1}`, 24 batches × 256 independent antithetic pairs and the existing calibration midpoint `c=.366768%`. Two thousand complete-batch resamples yield marginal 95% Monte Carlo intervals, enlarged by the conservative discount-tail allowance (target .01 percentage points).

Before execution, candidate choice is fixed as follows. Alpha1 is the matched control and cannot qualify as a treatment. At each fee, a high treatment must have its **combined lower bound at least 1.5c=.550152% in both zones**; a low treatment must have its **combined upper bound at most c/1.5=.244512% in both zones**. Choose the largest eligible high alpha and smallest eligible low alpha, the closest alpha arms from their respective sides. If either side is unavailable, report inability. Do not search other fees, weaken interval criteria, or select another reference.

Threshold uncertainty is not folded into this explicitly fixed bracketing rule: the earlier midpoint's MC interval is [.214692,.759701]%, conditional on the market/outcomes. A bracket passing this rule is only a feasibility result, not evidence of learning or a reliability guarantee. The bootstrap does not represent uncertainty in future agents or unseen market data. Reference-policy bias and rare-tail Monte Carlo error remain possible.

Run from the code repository:

```powershell
python exploratory_codex_2026-10-15/model_free/new_fee_design/exploratory_new_fee_design.py
```

The script runs one numerical thread, sequentially, with a 300-second cap between batches. It saves resumable aggregates only under this folder. All prior model-free results stay unchanged. Any confirmatory study requires separate authorization, a newly written/hashed protocol and analysis, fresh seeds and a detached resume-safe launcher after this feasibility result is reviewed.
