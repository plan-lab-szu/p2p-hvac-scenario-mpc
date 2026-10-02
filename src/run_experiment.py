"""Portable closed-loop entry point; uses the unchanged packaged optimizers.

Input acquisition is separate. No downloads, original household IDs or uploads.
The default four-step run is a smoke check, not a reproduction of paper tables.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'code'))
import numpy as np
from scenario_mean_qp import ScenarioMeanQP
from reduced_mean_qp import ReducedMeanQP
from box_envelope_qp import BoxEnvelopeQP
from component_box import component_bounds, fit_components
from balanced_tolerance_candidate import BalancedToleranceCandidate
from fitted_scenario_provider import provider
from forecast_closed_loop import thermal_step, price_at, settle


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['central', 'pj', 'mean_open_loop', 'component_residual', 'network'], default='pj')
    ap.add_argument('--mode', choices=['scenario', 'deterministic'], default='scenario')
    ap.add_argument('--homes', type=int, default=2)
    ap.add_argument('--start', type=int, default=3648)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--warmup', type=int, default=0)
    ap.add_argument('--seed', type=int, default=202)
    ap.add_argument('--calibration', type=Path)
    ap.add_argument('--network-case', type=Path)
    ap.add_argument('--background', type=float, default=0.5)
    ap.add_argument('--vary', choices=['R_C_per_kw', 'C_kwh_per_C', 'COP', 'cooling_max_kw_electric', 'pv'])
    ap.add_argument('--factor', type=float, default=1.)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('No overwrite: choose a new output directory')
    if not (1 <= args.homes <= 100 and args.start >= 1920 and args.steps > args.warmup >= 0 and args.factor > 0):
        raise ValueError('Invalid population, fitting boundary, steps, warmup or factor')
    if args.method in ('mean_open_loop', 'component_residual', 'network') and args.mode != 'scenario':
        raise ValueError('These baselines use the scenario forecast')
    cfg = json.loads((ROOT / 'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    # Worker partitioning needs ordered labels, not original customer identifiers.
    cfg['household_ids'] = ['home_%03d' % (i+1) for i in range(100)]
    key = 'sydney_utc10_end'
    with np.load(args.inputs, allow_pickle=False) as src:
        data = {k: src[k].copy() for k in ('time', 'base_kw', key+'_pv', key+'_ambient')}
    if str(data['time'][0])[:10] != '2013-11-01' or args.start + args.steps > len(data['time']):
        raise ValueError('Expected study time origin and sufficient realized input rows')
    if not np.all(np.diff(data['time']) == np.timedelta64(30, 'm')):
        raise ValueError('Expected consecutive half-hour time labels')
    if data['base_kw'].shape != (len(data['time']), 100) or data[key+'_pv'].shape != data['base_kw'].shape:
        raise ValueError('Expected 100 columns in the fixed parameter-assignment order')
    for name, value in data.items():
        if name != 'time' and not np.isfinite(value).all():
            raise ValueError('Nonfinite input: '+name)
    if args.vary == 'pv':
        for name in ('dc_kw', 'ac_kw'):
            cfg['pv'][name] = (np.array(cfg['pv'][name])*args.factor).tolist()
        data[key+'_pv'] *= args.factor
    elif args.vary:
        cfg['thermal'][args.vary] = (np.array(cfg['thermal'][args.vary])*args.factor).tolist()
    calibration = None
    if args.method == 'component_residual':
        calibration = (json.loads(args.calibration.read_text())['populations'][str(args.homes)]
                       if args.calibration else fit_components(data, args.homes))
    n, H, S = args.homes, 8, 1 if args.mode == 'deterministic' else 30
    th, p, dt = cfg['thermal'], cfg['controller'], cfg['dt_hours']
    temp, prev = np.full(n, th['initial_C']), np.zeros(n)
    arr = lambda name: np.array(th[name][:n])
    cls = {'central': ScenarioMeanQP, 'mean_open_loop': ReducedMeanQP, 'component_residual': BoxEnvelopeQP}
    if args.method == 'network':
        if args.network_case is None:
            raise ValueError('--network-case is required; third-party case data are supplied separately')
        os.environ['SEGAN_NETWORK_CASE'] = str(args.network_case.resolve())
        from network_mean_aggregated import AggregatedNetworkMean
        from network33 import ac_flow, nodal, matrices
    args.output.mkdir(parents=True)
    records = []
    result = dict(status='running', method=args.method, mode=args.mode, homes=n, seed=args.seed,
                  start=args.start, steps=args.steps, warmup=args.warmup, variation=args.vary, factor=args.factor,
                  input_sha256=digest(args.inputs), configuration=cfg, records=records,
                  scope='Portable wrapper; frozen optimizer modules unchanged. Timing is machine-dependent.')
    opt = None
    begun = time.perf_counter()
    try:
        if args.method == 'pj':
            opt = BalancedToleranceCandidate(cfg, n, S, 'pj', workers=4)
        elif args.method == 'network':
            opt = AggregatedNetworkMean(cfg, n, S, background=args.background)
            result.update(background=args.background, network_case_sha256=digest(args.network_case))
        else:
            opt = cls[args.method](cfg, n, S)
        for i, t in enumerate(range(args.start, args.start+args.steps)):
            f = provider(data, t, key, n, H, S, np.array(cfg['pv']['ac_kw'][:n]), mode=args.mode, seed=args.seed)
            if calibration is not None:
                f['box_low'], f['box_high'] = component_bounds(f, calibration, np.array(cfg['pv']['ac_kw'][:n]), 'component_residual')
            prices = price_at(np.arange(t, t+H), p)
            cool, trade, stats = opt.solve(f, temp, prev, prices)
            if args.method == 'pj':
                np.savez_compressed(args.output / ('certificate_%04d.npz' % i), **opt.last_clearing_certificate)
            planned = f['point_load'][:, 0]+cool-f['point_pv'][:, 0]-trade
            actual = settle(data['base_kw'][t, :n], data[key+'_pv'][t, :n], cool, trade, planned, prices[0], p, dt)
            T = thermal_step(temp, float(data[key+'_ambient'][t]), cool, arr('R_C_per_kw'), arr('C_kwh_per_C'), arr('COP'), arr('internal_gain_kw_thermal'), dt)
            clearing = float(abs(trade.sum()))
            balance = float(abs(data['base_kw'][t, :n]+cool-data[key+'_pv'][t, :n]-actual['net']-trade).max())
            excess = max(0., float(actual['net'].max()-p['grid_import_max_kw']), float(-actual['net'].min()-p['grid_export_max_kw']))
            if clearing >= 1e-8 or balance >= 1e-8 or excess >= 1e-7:
                raise ValueError('Electrical feasibility failed; no hidden repair')
            low, high = np.maximum(th['comfort_low_C']-T, 0), np.maximum(T-th['comfort_high_C'], 0)
            records.append(dict(step=i, time=str(data['time'][t]), warmup=i<args.warmup,
                grid_bill=float(actual['bill']), balancing_fee=float(actual['fee']),
                comfort_penalty=float(dt*p['comfort_slack_AUD_per_degree_hour']*(low+high).sum()),
                tracking_cost=float(dt*p['comfort_tracking_weight']*((T-th['setpoint_C'])**2).sum()),
                smoothing_cost=float(p['smoothing_weight']*((cool-prev)**2).sum()),
                cold_degree_hours=float(dt*low.sum()), hot_degree_hours=float(dt*high.sum()),
                hvac_kwh=float(dt*cool.sum()), temperature_min_C=float(T.min()), temperature_max_C=float(T.max()),
                balance_error_kw=balance, clearing_error_kw=clearing, capacity_violation_kw=excess,
                relative_objective_gap=float(stats['relative_objective_gap']), iterations=int(stats.get('iterations', 0)),
                algorithm_seconds=stats.get('algorithm_and_repair_seconds')))
            if args.method == 'network':
                P, Q = nodal(data['base_kw'][t, :n], data[key+'_pv'][t, :n], cool, args.background)
                ac = ac_flow(P, Q)
                loading = float((ac['branch_S_mva']/matrices(n)['ratings_mva']).max())
                transformer = float(np.hypot(ac['slack_P_kw'], ac['slack_Q_kvar'])/5000.)
                records[-1]['network'] = dict(voltage_min=float(ac['voltage_pu'].min()),
                    voltage_max=float(ac['voltage_pu'].max()), max_line_loading=loading,
                    transformer_loading=transformer, line_violation=loading>1+1e-8,
                    voltage_violation=bool(ac['voltage_pu'].min()<.9-1e-8 or ac['voltage_pu'].max()>1.1+1e-8),
                    transformer_violation=transformer>1+1e-8)
            temp, prev = T, cool
            if i % 6 == 0:
                print(json.dumps(dict(step=i, method=args.method, homes=n)), flush=True)
        ev = records[args.warmup:]
        costs = ['grid_bill', 'balancing_fee', 'comfort_penalty', 'tracking_cost', 'smoothing_cost']
        summary = {k: sum(r[k] for r in ev) for k in costs+['cold_degree_hours', 'hot_degree_hours', 'hvac_kwh']}
        summary.update(monetary_bill=summary['grid_bill']+summary['balancing_fee'],
                       realized_stage_cost=sum(summary[k] for k in costs), evaluated_steps=len(ev),
                       temperature_min_C=min(r['temperature_min_C'] for r in ev),
                       temperature_max_C=max(r['temperature_max_C'] for r in ev))
        result.update(status='passed', summary=summary)
    except Exception as error:
        result.update(status='failed', error=repr(error), completed_steps=len(records))
        raise
    finally:
        if args.method == 'pj' and opt is not None:
            opt.close()
        result['elapsed_seconds'] = time.perf_counter()-begun
        (args.output / 'run.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(args.output), summary=result['summary'])))


if __name__ == '__main__':
    main()
