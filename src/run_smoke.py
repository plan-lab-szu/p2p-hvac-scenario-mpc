"""Portable synthetic solver check, NOT reproduction of SGSC paper results."""
import json
import sys
from pathlib import Path
import importlib.metadata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
from scenario_mean_qp import ScenarioMeanQP
from robust_mpc import RiskOptimizer


def main():
    cfg = json.loads((ROOT / 'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    n, S, H = 2, 3, cfg['controller']['horizon']
    rng = np.random.default_rng(2030)
    load = rng.uniform(0.5, 2.0, (n, S, H))
    pv = rng.uniform(0, 0.5, (n, S, H))
    f = dict(load=load, pv=pv, ambient=rng.uniform(29, 32, (S, H)),
             point_load=load.mean(axis=1), point_pv=pv.mean(axis=1))
    temp = np.full(n, 24.0)
    previous = np.zeros(n)
    prices = np.full(H, 0.25)
    a = ScenarioMeanQP(cfg, n, S).solve(f, temp, previous, prices)
    b = RiskOptimizer(cfg, n, S, risk='mean').solve(f, temp, previous, prices)
    np.testing.assert_allclose(a[0], b[0], atol=2e-4, rtol=0)
    np.testing.assert_allclose(a[2]['feasible_objective'], b[2]['feasible_objective'], atol=1e-6, rtol=1e-7)
    assert abs(sum(a[1])) < 1e-8
    print(json.dumps(dict(status='passed', scope='synthetic central solver equivalence only',
        sgsc_results_reproduced=False, python=sys.version, objective=a[2]['feasible_objective'],
        max_cooling_difference=float(abs(a[0]-b[0]).max()),
        versions={p: importlib.metadata.version(p) for p in ('numpy','scipy','cvxpy','clarabel','pandas')}), indent=2))


if __name__ == '__main__':
    main()
