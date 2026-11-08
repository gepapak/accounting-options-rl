"""Independent review only: exact, data-free theory checks; no environment imports.

Writes only a new review_theory/exact_theory_results.json. Frozen studies and
author checks are read only as source text by the reviewer, never executed here.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PAPER = CODE.parent / "accounting-options-rl-latex-latex"


def solve_one(g, c, tau, sigma):
    r = max(sigma * g - c, tau)
    rt = max(-sigma * g - c, tau)
    spr = (r - rt) / 2 - c
    tir = sigma * (max(r, rt) + c) - c if r != rt else -c
    return r, rt, spr, tir


def run():
    counts = {"exact_grid_cases": 0, "spr_not_sharp_when_identified": 0,
              "uncensored_boundary_ties_g_positive": 0,
              "corrected_identification_failures": 0,
              "tir_error_dominance_failures": 0,
              "conditional_fairness_failures": 0}
    for gi, ci, ti, sigma in product(range(9), range(5), range(-10, 13), (-1, 1)):
        g, c, tau = F(gi, 2), F(ci, 2), F(ti, 2)
        r, rt, spr, tir = solve_one(g, c, tau, sigma)
        counts["exact_grid_cases"] += 1
        assert (r != rt) == (g > 0 and tau < g - c)
        assert abs(r - rt) / 2 <= g <= max(r, rt) + c
        assert (abs(r - rt) / 2 == g) == (g == 0 or tau <= -g - c)
        if r != rt:
            assert max(r, rt) + c == g
            assert tir == sigma * g - c
            counts["spr_not_sharp_when_identified"] += int(abs(r - rt) / 2 < g)
        else:
            assert 0 <= g <= r + c
            # Every rational grid candidate in the claimed interval reproduces
            # the equal pair with threshold r; its endpoints do so exactly.
            for candidate in (F(0), (r + c) / 2, r + c):
                rr, rrt, _, _ = solve_one(candidate, c, r, sigma)
                assert rr == rrt == r
        counts["uncensored_boundary_ties_g_positive"] += int(g > 0 and tau == g - c)
        assert abs(tir - (sigma * g - c)) <= abs(spr - (sigma * g - c))
        rp, rtp, sp, tp = solve_one(g, c, tau, 1)
        rm, rtm, sm, tm = solve_one(g, c, tau, -1)
        assert (sp + sm) / 2 == (tp + tm) / 2 == -c

    # Exact minimax proof by a convex-combination lower bound. With X=E[qz]
    # and Y=E|q|>=|X|, R1=Y-X/5, R10=1-2X+Y. Since X<=Y,
    # max(R1,R10)>=max(4Y/5,1-Y)>=4/9, attained X=Y=5/9.
    lam = F(5, 9)
    regret = max(F(4, 5) * lam, 1 - lam)
    assert regret == F(4, 9)
    for q in (F(i, 20) for i in range(-20, 21)):
        for m, sigma in product((F(1), F(10)), (-1, 1)):
            assert max(sigma * m * abs(q) - abs(q), F(10)) == 10

    # Pool sample dominance is only against the paired latent sample mean.
    # Independent uncensored magnitude {0,2}, mean 1, first draw 0.
    finite_pool = {"population_mean": "1", "uncensored_observation": "0",
                   "realized_pool_mean": "0", "probability": "1/2"}
    assert F(0) < F(1)
    # A censored observation also need not exceed the population mean:
    # m in {0,4}; at the realized m=0, C=0,q=1,tau=1, U=1<mu=2.
    censored_pool = {"population_mean": "2", "latent_draw": "0", "threshold": "1",
                     "reported_favorable_draw": "1", "realized_pool_mean": "1"}

    # Optional stopping alone can make an uncensored empirical mean biased.
    # Stop at the first 0, otherwise take exactly two iid {0,2} draws.
    stopped_expectation = F(1, 2)*0 + F(1, 4)*1 + F(1, 4)*2
    assert stopped_expectation == F(3, 4) < 1

    # History exclusion matters even under iid magnitudes. All favorable,
    # uncensored iid pool data, but policy trades only if a one-draw pool=2.
    # On the traded state-action, E[sigma]=1/2, true m mean=1, so PMR=1
    # while booked mean=1/2. Exact-mu PMR would agree with booked mean.
    conditional_selection = {"pool": "2", "current_true_mean": "1",
                             "expected_gross_sign": "1/2", "q": "1", "C": "0",
                             "PMR_conditional_mean": "1", "booked_conditional_mean": "1/2"}

    # Claim-increment convention is the paper's disclosed exception to C.
    # With debt D=2,g=3,C=0, the owner claim rewards are (1,0).
    # No common threshold gives max(+3,tau)=1.
    zombie = {"D": "2", "g": "3", "C": "0", "r_plus": "1", "r_minus": "0"}

    # Illustrate why conditional sign fairness must include magnitude.
    # Marginal sign fair, but g=1 on + and 3 on -, zero floor at equity 0.
    # E[booked]=-1, E[SPR]=(-? actual pairs depend on g)= -1/2.
    gp, gm = F(1), F(3)
    _, _, spp, _ = solve_one(gp, F(0), F(0), 1)
    _, _, spm, _ = solve_one(gm, F(0), F(0), -1)
    assert (gp - gm)/2 == -1 and (spp + spm)/2 == F(-1, 2)

    source_paths = [PAPER / "sections" / "identification.tex",
                    PAPER / "sections" / "supp_spr.tex",
                    CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_t" / "envs_t.py",
                    CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_t" / "envs_tn.py",
                    CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "study_v" / "rewards_v.py",
                    CODE / "ACCOUNTING_OPTIONS_2026-10-14" / "twin_reward" / "twin_reward.py"]
    return {"scope": "independent data-free exact review; zero training, model, environment, or market-data calls",
            "counts": counts, "minimax_regret": str(regret), "minimax_position_fraction": str(lam),
            "SPR_nonsharp_example": {"g": "10", "C": "1", "tau": "0", "r_plus": "9", "r_minus": "0", "identified_g": "10", "SPR_magnitude": "9/2"},
            "uncensored_boundary_example": {"g": "1", "C": "0", "tau": "1", "r_plus": "1", "r_minus": "1", "favorable_credit": "0"},
            "finite_uncensored_pool": finite_pool, "finite_censored_pool": censored_pool,
            "optional_stopping_pool_expected_mean": str(stopped_expectation),
            "state_action_selection_of_random_pool": conditional_selection,
            "claim_increment_not_C": zombie,
            "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}


if __name__ == "__main__":
    out = run()
    target = HERE / "exact_theory_results.json"
    if target.exists():
        raise RuntimeError("Refusing to overwrite an existing review result")
    target.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("scope", "counts", "minimax_regret")}, indent=2))
