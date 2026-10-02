"""Saved-cost attribution and predeclared same-state risk/information ablations."""
import json
import numpy as np
import cvxpy as cp
from benchmark_validation import ROOT,digest
from robust_mpc import RiskOptimizer
from uncached_central_optimizer import DirectProblem
from forecast_closed_loop import price_at,thermal_step
from scenario_stability_study import sample_forecast


class OpenLoopRiskOptimizer(RiskOptimizer):
    def __init__(self,cfg,n,S,method='central',risk='mean',save_certificate=False):
        super().__init__(cfg,n,S,method,risk,save_certificate)
        extra=[]
        if S>1:
            for h in self.homes:
                extra += [h.cool[1:,:]==cp.vstack([h.cool[0,:]]*(S-1)),h.g[1:,:]==cp.vstack([h.g[0,:]]*(S-1))]
        self.problem=DirectProblem(self.problem.objective,self.problem.constraints+extra)


def attribute(run):
    cfg=run['full_configuration'];p=cfg['controller'];th=cfg['thermal'];dt=cfg['dt_hours'];rows=[]
    for row in run['records'][run['warmup']:]:
        worst=int(np.argmax(row['scenario_community_costs']));cold=hot=0.
        for home in row['risk_certificate']:
            T=np.array(home['temp'])[worst,1:]
            cold+=dt*p['comfort_slack_AUD_per_degree_hour']*np.maximum(th['comfort_low_C']-T,0).sum()
            hot+=dt*p['comfort_slack_AUD_per_degree_hour']*np.maximum(T-th['comfort_high_C'],0).sum()
        rows.append(dict(label=row['label'],worst_index=worst,worst_cost=row['worst_horizon_cost'],
            cold_penalty=float(cold),hot_penalty=float(hot),cold_dominates=bool(cold>row['worst_horizon_cost']/2)))
    return dict(steps=len(rows),cold_majority_steps=int(sum(x['cold_dominates'] for x in rows)),
        weighted_cold_fraction=float(sum(x['cold_penalty'] for x in rows)/sum(x['worst_cost'] for x in rows)),records=rows)


def main():
    cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text());data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'))
    folder=ROOT/'results/heat_diagnosis_20260914_v1';folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(homes=[3,10],labels=['2014-01-15T08:00','2014-01-15T12:00','2014-01-16T08:00','2014-01-16T12:00'],
        state='Saved hull-minmax trajectory, same state/previous action for all ablations',
        variants=['mean_two_stage','worst_two_stage','worst_open_loop'],formal_test_run=False,
        source_sha256=digest(__file__),optimizer_sha256=digest(ROOT/'robust_mpc.py'))
    out=folder/'results.json'
    if out.exists():raise FileExistsError(out)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    result=dict(attribution={},pairs=[],protocol=protocol)
    for n in protocol['homes']:
        path=next((ROOT/f'results/robust_validation_20260914_v1/n{n}_hull_minmax').glob('*w48.json'))
        run=json.loads(path.read_text());result['attribution'][n]=attribute(run)
        th=cfg['thermal'];R=np.array(th['R_C_per_kw'][:n]);C=np.array(th['C_kwh_per_C'][:n]);cop=np.array(th['COP'][:n]);gain=np.array(th['internal_gain_kw_thermal'][:n])
        for label in protocol['labels']:
            i=next(i for i,r in enumerate(run['records']) if np.datetime64(r['label'])==np.datetime64(label))
            row=run['records'][i];temp=np.array(row['temperature_before_C']);prev=np.array(run['records'][i-1]['cooling_kw'])
            t=int(np.searchsorted(data['time'],np.datetime64(label)))
            f=sample_forecast(data,t,run['case'],n,8,30,np.array(cfg['pv']['ac_kw'][:n]),strategy='fixed_days',seed=202)
            variants={}
            for name in protocol['variants']:
                cls=OpenLoopRiskOptimizer if name.endswith('open_loop') else RiskOptimizer
                opt=cls(cfg,n,30,risk='mean' if name.startswith('mean') else 'worst')
                cool,trade,stats=opt.solve(f,temp,prev,price_at(np.arange(t,t+8),cfg['controller']))
                nxt=thermal_step(temp,data[run['case']+'_ambient'][t],cool,R,C,cop,gain,cfg['dt_hours'])
                variants[name]=dict(first_cooling_kw=cool.tolist(),total_cooling_kw=float(cool.sum()),
                    actual_next_temperature_C=nxt.tolist(),hot_degree_hours=float(cfg['dt_hours']*np.maximum(nxt-th['comfort_high_C'],0).sum()),
                    objective=stats['feasible_objective'],max_first_temperature_difference_to_saved=float(abs(nxt-row['temperature_after_C']).max()))
            result['pairs'].append(dict(n=n,label=label,variants=variants,source_sha256=digest(path)))
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(attribution={n:{k:v for k,v in a.items() if k!='records'} for n,a in result['attribution'].items()},pairs=result['pairs']),indent=2))


if __name__=='__main__':main()
