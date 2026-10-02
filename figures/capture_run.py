"""Trajectory-capturing replica of run_experiment.py (central / deterministic / mean_open_loop / component_residual / network).

Uses the unchanged packaged optimizers; the closed-loop logic is copied line-for-line from
run_experiment.py, but per-home arrays are additionally saved for plotting.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse, json, sys, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'src'
DATA = REPO / 'data'
sys.path.insert(0, str(ROOT / 'code'))
from scenario_mean_qp import ScenarioMeanQP
from reduced_mean_qp import ReducedMeanQP
from box_envelope_qp import BoxEnvelopeQP
from component_box import component_bounds, fit_components
from fitted_scenario_provider import provider
from forecast_closed_loop import thermal_step, price_at, settle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=['central', 'mean_open_loop', 'component_residual', 'network'], default='central')
    ap.add_argument('--mode', choices=['scenario', 'deterministic'], default='scenario')
    ap.add_argument('--homes', type=int, default=50)
    ap.add_argument('--start', type=int, default=3648)
    ap.add_argument('--steps', type=int, default=672)
    ap.add_argument('--warmup', type=int, default=48)
    ap.add_argument('--seed', type=int, default=202)
    ap.add_argument('--background', type=float, default=0.5)
    args = ap.parse_args()
    cfg = json.loads((ROOT / 'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    cfg['household_ids'] = ['home_%03d' % (i+1) for i in range(100)]
    key = 'sydney_utc10_end'
    with np.load(DATA / 'study_inputs.npz', allow_pickle=False) as src:
        data = {k: src[k].copy() for k in ('time', 'base_kw', key+'_pv', key+'_ambient')}
    calibration = fit_components(data, args.homes) if args.method == 'component_residual' else None
    n, H, S = args.homes, 8, 1 if args.mode == 'deterministic' else 30
    th, p, dt = cfg['thermal'], cfg['controller'], cfg['dt_hours']
    temp, prev = np.full(n, th['initial_C']), np.zeros(n)
    arr = lambda name: np.array(th[name][:n])
    cls = {'central': ScenarioMeanQP, 'mean_open_loop': ReducedMeanQP, 'component_residual': BoxEnvelopeQP}
    if args.method == 'network':
        os.environ['SEGAN_NETWORK_CASE'] = str((DATA / 'network_case.npz').resolve())
        from network_mean_aggregated import AggregatedNetworkMean
        from network33 import ac_flow, nodal, matrices
        opt = AggregatedNetworkMean(cfg, n, S, background=args.background)
    else:
        opt = cls[args.method](cfg, n, S)
    keep = {k: [] for k in ('T', 'cool', 'trade', 'net', 'planned', 'base', 'pv', 'price', 'bill', 'fee', 'solve_s')}
    amb, times, netrec = [], [], []
    begun = time.perf_counter()
    for i, t in enumerate(range(args.start, args.start+args.steps)):
        f = provider(data, t, key, n, H, S, np.array(cfg['pv']['ac_kw'][:n]), mode=args.mode, seed=args.seed)
        if calibration is not None:
            f['box_low'], f['box_high'] = component_bounds(f, calibration, np.array(cfg['pv']['ac_kw'][:n]), 'component_residual')
        prices = price_at(np.arange(t, t+H), p)
        s0 = time.perf_counter()
        cool, trade, stats = opt.solve(f, temp, prev, prices)
        keep['solve_s'].append(time.perf_counter()-s0)
        planned = f['point_load'][:, 0]+cool-f['point_pv'][:, 0]-trade
        actual = settle(data['base_kw'][t, :n], data[key+'_pv'][t, :n], cool, trade, planned, prices[0], p, dt)
        T = thermal_step(temp, float(data[key+'_ambient'][t]), cool, arr('R_C_per_kw'), arr('C_kwh_per_C'), arr('COP'), arr('internal_gain_kw_thermal'), dt)
        for k, v in (('T', T), ('cool', cool), ('trade', trade), ('net', actual['net']), ('planned', planned),
                     ('base', data['base_kw'][t, :n]), ('pv', data[key+'_pv'][t, :n]), ('price', prices[0]),
                     ('bill', actual['bill']), ('fee', actual['fee'])):
            keep[k].append(np.asarray(v, dtype=float))
        amb.append(float(data[key+'_ambient'][t])); times.append(str(data['time'][t]))
        if args.method == 'network':
            P, Q = nodal(data['base_kw'][t, :n], data[key+'_pv'][t, :n], cool, args.background)
            ac = ac_flow(P, Q)
            netrec.append([float(ac['voltage_pu'].min()), float(ac['voltage_pu'].max()),
                           float((ac['branch_S_mva']/matrices(n)['ratings_mva']).max()),
                           float(np.hypot(ac['slack_P_kw'], ac['slack_Q_kvar'])/5000.)])
        temp, prev = T, cool
        if i % 24 == 0:
            print(json.dumps(dict(step=i, elapsed=round(time.perf_counter()-begun, 1))), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {k: np.array(v) for k, v in keep.items()}
    out.update(ambient=np.array(amb), time=np.array(times), warmup=args.warmup, homes=n,
               T_init=th['initial_C'], low=th['comfort_low_C'], high=th['comfort_high_C'])
    if netrec:
        out['network'] = np.array(netrec)
    np.savez_compressed(args.output, **out)
    ev = slice(args.warmup, None)
    bill = float(out['bill'][ev].sum() + out['fee'][ev].sum())
    print(json.dumps(dict(done=str(args.output), monetary_bill=bill, elapsed=time.perf_counter()-begun)))


if __name__ == '__main__':
    main()
