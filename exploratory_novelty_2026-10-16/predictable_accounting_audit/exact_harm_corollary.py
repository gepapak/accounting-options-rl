"""Exact optional-trade harm corollary checks; no Monte Carlo or old inputs."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZERO, ONE, GAMMA = F(0), F(1), F(99, 100)
COST, H0, RHO = F(2, 5), F(7, 10), F(7, 4)
ALPHA_STAR = F(3, 7)


def num(x):
    return {"exact": str(x), "decimal": float(x)}


def optional_trade(account, initial, alpha, probability, time, cost=COST, horizon=2):
    """Private pre-payoff coin trades once or remains flat for the whole episode."""
    gross_mean = booked = forgiveness = ZERO
    for gross in (F(-1), ONE):
        p = probability/2
        net = gross-cost
        booked += p*GAMMA**time*net
        gross_mean += p*GAMMA**time*gross
        weight = GAMMA**time if account == "floor" else GAMMA**(horizon-1)
        forgiveness += p*weight*max(-initial-net, ZERO)
    b = probability*GAMMA**time*cost
    charged = booked+(1-alpha)*forgiveness
    assert gross_mean == 0 and booked == -b
    return charged, booked, b, forgiveness


def main():
    records = []
    for account in ("floor", "terminal"):
        for initial in (ZERO, F(7, 10)):
            for time in (0, 1):
                for probability in (ZERO, F(1, 4), ONE):
                    for alpha in (ZERO, F(1, 5), ALPHA_STAR, F(4, 5), ONE):
                        charged, booked, b, forgiveness = optional_trade(account, initial, alpha, probability, time)
                        kappa = (1-alpha)*RHO-1
                        assert charged <= kappa*b
                        record = dict(account=account, initial=num(initial), trade_time=time,
                            trade_probability=num(probability), alpha=num(alpha), kappa=num(kappa),
                            charged=num(charged), booked=num(booked), expected_predictable_cost=num(b),
                            expected_forgiveness=num(forgiveness), uniform_upper=num(kappa*b),
                            can_remain_flat=probability < 1, mandatory_trade=probability == 1)
                        if kappa > 0 and charged > 0:
                            # An exact conservative lower bound, not a sampled CI.
                            ell = charged/2
                            harm_upper = -ell/kappa
                            assert booked <= harm_upper < 0
                            record.update(exact_lower_bound=num(ell),
                                          booked_harm_upper_from_charge_only=num(harm_upper))
                        if probability == 0:
                            assert charged == booked == b == 0
                        if alpha >= ALPHA_STAR:
                            assert charged <= 0
                        records.append(record)
    highlighted = next(r for r in records if r["account"] == "floor"
        and r["initial"]["exact"] == "0" and r["trade_time"] == 0
        and r["trade_probability"]["exact"] == "1/4" and r["alpha"]["exact"] == "0")
    assert highlighted["charged"]["exact"] == "3/40"
    assert highlighted["booked"]["exact"] == "-1/10"
    assert highlighted["kappa"]["exact"] == "3/4"
    assert highlighted["booked_harm_upper_from_charge_only"]["exact"] == "-1/20"

    free_charged, free_booked, _, _ = optional_trade("floor", ZERO, F(4, 5), F(1, 4), 0, cost=ZERO)
    assert free_charged == F(1, 40) > 0 and free_booked == 0
    # A loose but valid moment envelope yields a sufficient, nonnecessary rule.
    envelope_rho = 2*RHO
    envelope_threshold = 1-1/envelope_rho
    assert ALPHA_STAR < F(1, 2) < envelope_threshold
    assert (1-F(1, 2))*H0-COST < 0

    result = dict(status="exploratory exact harm-corollary verification",
        cases=len(records), all_checks_passed=True, no_monte_carlo=True,
        law="Independent fair gross +/-1, cost2/5; flat action available",
        rho=num(RHO), alpha_star=num(ALPHA_STAR), optional_policy_example=highlighted,
        optional_fixed_policy_cases=records,
        zero_cost_counterexample=dict(alpha=num(F(4, 5)), trade_probability=num(F(1, 4)),
            charged=num(free_charged), booked=num(free_booked), rho="infinity",
            explanation="Positive charged value with zero booked harm; finite-rho premise cannot be dropped"),
        conditional_envelope_is_only_sufficient=dict(true_threshold=num(ALPHA_STAR),
            loose_envelope_rho=num(envelope_rho), loose_envelope_threshold=num(envelope_threshold),
            charge_with_actual_uniform_repair_but_envelope_condition_fails=num(F(1, 2))),
        zero_admissibility_scope=dict(rule="Trade allowed only at ledger>=2; otherwise flat",
            maximum_unit_net_loss=num(F(7, 5)), smallest_post_trade_ledger=num(F(3, 5)),
            implication="From solvent starts this restriction never reaches the floor/debt, so all-policy repair holds even below the unconstrained zero-state threshold"),
        scope="The ratio and probability arguments are classical. The corollary specializes their economic-harm implication under the declared accounting kernels.",
        inputs_read=[], learners_trained=0, dp_solved=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"exact_harm_corollary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases":len(records),"all_checks_passed":True,
                      "optional_policy_example":highlighted,
                      "zero_cost_counterexample":result["zero_cost_counterexample"]},indent=2))


if __name__ == "__main__":
    main()
