"""Fixed-state timing; run serially on an otherwise idle machine.

Numerical failures are retained as failures, never successful speedup samples.
"""
import argparse
import copy
import json
import time
from run_experiment import ROOT, digest, provider, price_at, BalancedToleranceCandidate
import numpy as np
from bounded_repair_gs import BoundedRepairGS


def main():
    from pathlib import Path
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--states', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--job-index', type=int, help='Run one indexed job; omit to run all 24')
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    cfg = json.loads((ROOT/'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    cfg['household_ids'] = ['home_%03d' % (i+1) for i in range(100)]
    key = 'sydney_utc10_end'
    with np.load(a.inputs, allow_pickle=False) as src:
        data = {k:src[k] for k in ('time','base_kw',key+'_pv',key+'_ambient')}
    jobs = json.loads(a.states.read_text())['jobs']
    if a.job_index is not None:
        if not 0 <= a.job_index < len(jobs):
            raise ValueError('Invalid job index')
        jobs = [jobs[a.job_index]]
    a.output.mkdir(parents=True)
    rows = []
    for job in jobs:
        n, t, method = job['n'], job['t'], job['method']
        if str(data['time'][t]) != job['label']:
            # Precision suffix can differ; the datetime value must match.
            if data['time'][t] != np.datetime64(job['label']):
                raise ValueError('Timing input/state time mismatch')
        row = {k:job[k] for k in ('n','step','repeat','method','label')}
        opt = None
        try:
            begin = time.perf_counter()
            f = provider(data,t,key,n,8,30,np.array(cfg['pv']['ac_kw'][:n]),seed=202)
            prices = price_at(np.arange(t,t+8),cfg['controller'])
            row['forecast_seconds'] = time.perf_counter()-begin
            begin = time.perf_counter()
            opt = (BoundedRepairGS(copy.deepcopy(cfg),n,30) if method=='gs'
                   else BalancedToleranceCandidate(cfg,n,30,'pj',workers=4))
            row['initialization_seconds'] = time.perf_counter()-begin
            if method=='gs':
                stats = opt.solve(f,np.array(job['temp']),np.array(job['previous']),prices)
                seconds, gap = stats['algorithm_seconds'],stats['relative_gap']
            else:
                _,trade,stats = opt.solve(f,np.array(job['temp']),np.array(job['previous']),prices)
                assert abs(trade.sum())<1e-8
                seconds, gap = stats['algorithm_and_repair_seconds'],stats['relative_objective_gap']
            assert gap<1e-4
            row.update(status='passed',algorithm_seconds=seconds,relative_gap=gap,
                cold_control_latency_seconds=row['forecast_seconds']+row['initialization_seconds']+seconds)
        except Exception as error:
            row.update(status='failed',error=repr(error))
        finally:
            if method=='pj' and opt is not None:
                opt.close()
        rows.append(row)
        print(json.dumps(row),flush=True)
        (a.output/'timing.json').write_text(json.dumps(dict(input_sha256=digest(a.inputs),
            state_sha256=digest(a.states), results=rows,
            warning='Machine-dependent timings; capped failures are not speedup denominators.'),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
