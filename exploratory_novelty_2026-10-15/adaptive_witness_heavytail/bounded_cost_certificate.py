"""EXPLORATORY: cost-bounded accounting certificates for unbounded price laws.

No training or existing study outcome is read. See EXPLORATORY_PLAN.md.
The statistical inequality and cap selection are established tools.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

HERE = Path(__file__).resolve().parent
N_SEARCH, N_CONFIRM = 8192, 262144
EQUITY, COST, TARGET_ALPHA = 2., .1, .8
CAPS = (4., 16., 64., 256.)
SHAPES = (1.5, 2.5)
DELTA = .05 / len(SHAPES)


def lower_mean(values, lower, upper, delta, projected_n=None):
    """Fixed-budget one-sided M&P bound; projection is SEARCH scoring only."""
    values = np.asarray(values, dtype=float)
    assert len(values) > 1 and lower <= upper and 0 < delta < 1
    assert np.all(values >= lower) and np.all(values <= upper)
    n = len(values) if projected_n is None else projected_n
    log = np.log(2 / delta)
    variance = float(values.var(ddof=1))
    radius = float(np.sqrt(2 * variance * log / n)
                   + 7 * (upper - lower) * log / (3 * (n-1)))
    return {'n': len(values), 'scoring_n': n, 'mean': float(values.mean()),
            'sample_variance': variance, 'range': [lower, upper],
            'delta': delta, 'radius': radius, 'lower_mean': float(values.mean()-radius)}


def draw_forgiveness(rng, shape, n):
    magnitude = 1 + rng.pareto(shape, n)
    sign = rng.choice(np.array([-1., 1.]), n)
    net = sign * magnitude - COST
    return np.maximum(-EQUITY-net, 0.)


def analytic_mean(shape, cap=None):
    a = EQUITY-COST
    mean = .5 * a**(1-shape) / (shape-1)
    if cap is not None:
        mean -= .5 * (a+cap)**(1-shape) / (shape-1)
    return float(mean)


def main():
    start = time.perf_counter()
    search_rng = np.random.default_rng(2026101501)
    confirm_rng = np.random.default_rng(2026101502)
    rows = []
    for shape in SHAPES:
        search = draw_forgiveness(search_rng, shape, N_SEARCH)
        candidates = []
        for cap in CAPS:
            fcap = (cap+COST)/(1-TARGET_ALPHA)
            score = lower_mean(np.minimum(search, fcap), 0., fcap, DELTA, N_CONFIRM)
            candidates.append({'residual_cap': cap, 'forgiveness_cap': fcap,
                               'projected_target_lower': (1-TARGET_ALPHA)*score['lower_mean']-COST,
                               'search_scoring_only': score})
        chosen = max(candidates, key=lambda x: x['projected_target_lower'])
        cap = chosen['forgiveness_cap']
        confirmation = draw_forgiveness(confirm_rng, shape, N_CONFIRM)
        bound = lower_mean(np.minimum(confirmation, cap), 0., cap, DELTA)
        lower_f = bound['lower_mean']
        true_f, true_capped_f = analytic_mean(shape), analytic_mean(shape, cap)
        certified_limit = 1-COST/lower_f if lower_f > COST else None
        rows.append({
            'pareto_shape': shape, 'price_support_upper': None,
            'gross_variance_finite': shape > 2, 'expected_booked_return': -COST,
            'search_candidates': candidates, 'selected_cap': chosen['residual_cap'],
            'selected_forgiveness_cap': cap, 'confirmation': bound,
            'true_forgiveness_mean': true_f, 'true_capped_forgiveness_mean': true_capped_f,
            'confidence_covers_capped_truth': lower_f <= true_capped_f,
            'true_critical_alpha': 1-COST/true_f,
            'certified_alpha_interval': [0., certified_limit] if certified_limit is not None else None,
            'certified_alpha_upper_is_open': True,
            'target_alpha': TARGET_ALPHA,
            'target_true_proxy_mean': (1-TARGET_ALPHA)*true_f-COST,
            'target_proxy_lower': (1-TARGET_ALPHA)*lower_f-COST,
            'target_positive_witness_confirmed': (1-TARGET_ALPHA)*lower_f-COST > 0,
            'full_booking_mean': -COST, 'full_booking_positive_witness': False,
            'raw_bounded_Bernstein_at_target': 'unsupported: raw negative tail is unbounded',
            'model_scope': 'Known synthetic IID fair-sign law, finite first moment, bounded executed cost; not future-market coverage.',
        })
    result = {'status': 'EXPLORATORY completed generative certificate validation',
              'n_search_per_law': N_SEARCH, 'n_confirmation_per_law': N_CONFIRM,
              'joint_error_budget': .05, 'confidence_unit': 'IID complete paths',
              'all_alpha_uniform_for_fixed_action': True, 'rows': rows,
              'elapsed_seconds': time.perf_counter()-start,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'plan_sha256': hashlib.sha256((HERE/'EXPLORATORY_PLAN.md').read_bytes()).hexdigest()}
    (HERE/'exploratory_heavytail_certificate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'seconds': result['elapsed_seconds'],
                     'cases': [{k: row[k] for k in ('pareto_shape', 'selected_cap',
                       'target_true_proxy_mean', 'target_proxy_lower', 'target_positive_witness_confirmed',
                       'certified_alpha_interval', 'confidence_covers_capped_truth')} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
