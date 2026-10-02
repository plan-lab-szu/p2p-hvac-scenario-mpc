"""Run a central scenario-MPC check with locally supplied, prepared inputs.

No data are downloaded, no original household IDs are used, and no upload occurs.
Default is a short check, not the full paper experiment.
"""
import argparse
import json
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
from scenario_mean_qp import ScenarioMeanQP
from fitted_scenario_provider import provider
from forecast_closed_loop import thermal_step, price_at, settle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--homes', type=int, default=2)
    parser.add_argument('--start', type=int, default=76 * 48)
    parser.add_argument('--steps', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('No overwrite: choose a new output path')
    cfg = json.loads((ROOT / 'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    key = 'sydney_utc10_end'
    with np.load(args.inputs, allow_pickle=False) as src:
        data = {k: src[k] for k in ('time', 'base_kw', key+'_pv', key+'_ambient')}
    n, H, S = args.homes, 8, 30
    if not (1 <= n <= 100 and args.start >= 40*48 and args.steps > 0):
        raise ValueError('Invalid population, fitting-pool boundary or step count')
    if args.start + args.steps > len(data['time']):
        raise ValueError('Insufficient realized input rows')
    if not np.all(np.diff(data['time']) == np.timedelta64(30, 'm')):
        raise ValueError('Inputs must have consecutive half-hour timestamps')
    th, p, dt = cfg['thermal'], cfg['controller'], cfg['dt_hours']
    temp = np.full(n, th['initial_C'])
    prev = np.zeros(n)
    opt = ScenarioMeanQP(cfg, n, S)
    arr = lambda name: np.array(th[name][:n])
    rows = []
    for t in range(args.start, args.start + args.steps):
        f = provider(data, t, key, n, H, S, np.array(cfg['pv']['ac_kw'][:n]), seed=202)
        prices = price_at(np.arange(t, t+H), p)
        cool, trade, stats = opt.solve(f, temp, prev, prices)
        planned = f['point_load'][:, 0] + cool - f['point_pv'][:, 0] - trade
        actual = settle(data['base_kw'][t, :n], data[key+'_pv'][t, :n], cool, trade, planned, prices[0], p, dt)
        temp = thermal_step(temp, float(data[key+'_ambient'][t]), cool, arr('R_C_per_kw'), arr('C_kwh_per_C'), arr('COP'), arr('internal_gain_kw_thermal'), dt)
        assert abs(trade.sum()) < 1e-8
        if actual['net'].max() > p['grid_import_max_kw'] + 1e-7 or actual['net'].min() < -p['grid_export_max_kw'] - 1e-7:
            raise ValueError('Realized connection violation; no hidden repair')
        rows.append(dict(time=str(data['time'][t]), bill=float(actual['bill']), balancing_fee=float(actual['fee']),
                         min_temperature=float(temp.min()), max_temperature=float(temp.max()),
                         cold_degree_hours=float(dt*np.maximum(th['comfort_low_C']-temp, 0).sum()),
                         hot_degree_hours=float(dt*np.maximum(temp-th['comfort_high_C'], 0).sum()),
                         relative_objective_gap=stats['relative_objective_gap']))
        prev = cool
    result = dict(scope='central scenario MPC; not distributed-PJ reproduction', homes=n, steps=args.steps,
                  input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(), records=rows)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', homes=n, steps=len(rows), output=str(args.output))))


if __name__ == '__main__':
    main()
