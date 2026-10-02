"""Record per-iteration residual traces of PJ (10/50/100 homes) and GS (10 homes) on the packaged fixed timing states."""
import copy, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from run_experiment import ROOT, provider, price_at, BalancedToleranceCandidate  # noqa: E402
from bounded_repair_gs import BoundedRepairGS  # noqa: E402


def main():
    cfg = json.loads((ROOT/'configuration_without_identifiers.json').read_text())
    cfg['controller'].update(cfg['final_run_overrides'])
    cfg['household_ids'] = ['home_%03d' % (i+1) for i in range(100)]
    key = 'sydney_utc10_end'
    data_dir = REPO / 'data'
    with np.load(data_dir/'study_inputs.npz', allow_pickle=False) as src:
        data = {k: src[k] for k in ('time', 'base_kw', key+'_pv', key+'_ambient')}
    jobs = json.loads((data_dir/'timing_states.json').read_text())['jobs']
    only = sys.argv[1] if len(sys.argv) > 1 else None
    pick = [j for j in jobs if j['t'] == 3624 and j['repeat'] == 0 and (only is None or j['method'] == only)]
    out = {}
    for job in pick:
        n, t, method = job['n'], job['t'], job['method']
        f = provider(data, t, key, n, 8, 30, np.array(cfg['pv']['ac_kw'][:n]), seed=202)
        prices = price_at(np.arange(t, t+8), cfg['controller'])
        opt = (BoundedRepairGS(copy.deepcopy(cfg), n, 30) if method == 'gs'
               else BalancedToleranceCandidate(cfg, n, 30, 'pj', workers=4))
        try:
            opt.solve(f, np.array(job['temp']), np.array(job['previous']), prices)
            status = 'passed'
        except Exception as e:  # GS is expected to hit the iteration cap
            status = 'failed: %r' % (e,)
        finally:
            if method == 'pj':
                opt.close()
        trace = getattr(opt, 'trace', None) or getattr(opt, 'history', [])
        out['%s_%d' % (method, n)] = dict(status=status, trace=trace,
                                          penalty_events=getattr(opt, 'penalty_events', []))
        print(method, n, status[:80], len(trace), flush=True)
    out_dir = REPO / 'results' / 'figure_data'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ('convergence%s.json' % ('_' + only if only else ''))).write_text(json.dumps(out, default=float))


if __name__ == '__main__':
    main()
