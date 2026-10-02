"""Independent causal closed loops with point/scenario forecasts and shared physics.

Validation dates only. Serial synchronous PJ-ADMM is NOT a parallel timing test.
Existing causal validation outputs and fixed parameters are never overwritten.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy

from forecast_closed_loop import (ROOT, SOLVER, MARKET_TOL_KW, Home,
                                 cleared_snapshot, forecast, price_at, settle, thermal_step)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Optimizer:
    def __init__(self,cfg,n,S,method):
        if method not in ('central','pj'):
            raise ValueError(method)
        self.method=method; self.cfg=cfg
        self.homes=[Home(i,cfg,S) for i in range(n)]
        self.center=cp.Problem(cp.Minimize(sum(h.cost for h in self.homes)),
            sum([h.constraints for h in self.homes],[])+[sum(h.trade for h in self.homes)==0])
        D=self.homes[0].D
        self.z=np.zeros((n,D)); self.u=self.z.copy(); self.old=self.z.copy()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter(); p=self.cfg['controller']; homes=self.homes
        for i,h in enumerate(homes):
            h.set(f,temp[i],previous[i],prices)
        self.center.solve(**SOLVER)
        if self.center.status!='optimal':
            raise RuntimeError(('central status',self.center.status))
        optimum=float(self.center.value)
        center_seconds=time.perf_counter()-start
        assert abs(sum(h.cost_check(h.snapshot()) for h in homes)-optimum)<1e-5
        iterations=0; primal=dual=proximal=0.
        algorithm_start=time.perf_counter()
        if self.method=='pj':
            n,D=self.z.shape
            for iterations in range(1,1001):
                gamma=[]
                for i,h in enumerate(homes):
                    h.target.value=self.z[i]-self.u[i]; h.old.value=self.old[i]
                    h.problem.solve(warm_start=True,**SOLVER)
                    if h.problem.status!='optimal':
                        raise RuntimeError(('local status',iterations,i,h.problem.status))
                    gamma.append(h.trade.value.copy())
                gamma=np.array(gamma); v=gamma+self.u; z=v-v.mean(axis=0,keepdims=True)
                primal=float(np.linalg.norm(gamma-z))
                dual=float(p['rho']*np.linalg.norm(z-self.z))
                proximal=float(p['beta']*np.linalg.norm(gamma-self.old))
                self.u+=gamma-z; self.old=gamma.copy(); self.z=z
                ep=1e-4*np.sqrt(n*D)+1e-6*max(np.linalg.norm(gamma),np.linalg.norm(z))
                ed=1e-6*np.sqrt(n*D)+1e-6*np.linalg.norm(p['rho']*self.u)
                if primal<=ep and dual<=ed and proximal<=ed and abs(gamma.sum(axis=0)).max()<=MARKET_TOL_KW:
                    break
            else:
                raise RuntimeError(('ADMM iteration limit',iterations,primal,dual,proximal))
        else:
            gamma=np.array([h.trade.value.copy() for h in homes])
            z=gamma-gamma.mean(axis=0,keepdims=True)
        raw=[h.snapshot() for h in homes]
        raw_constraint=max(float(np.max(c.violation())) for h in homes for c in h.constraints)
        assert raw_constraint<1e-6
        repaired=[cleared_snapshot(h,v,z[i]) for i,(h,v) in enumerate(zip(homes,raw))]
        repair=max(float(abs(a['g']-b['g']).max()) for a,b in zip(repaired,raw))
        objective=sum(h.cost_check(v) for h,v in zip(homes,repaired))
        gap=abs(objective-optimum)/max(1,abs(optimum))
        assert objective>=optimum-1e-6*max(1,abs(optimum)) and gap<1e-4
        assert abs(sum(v['g'] for v in repaired)).max()<1e-8
        for h,v in zip(homes,repaired):
            assert abs(h.load.value+v['cool']-h.pv.value-v['buy']+v['sell']-v['g']).max()<1e-8
            assert np.minimum(v['buy'],v['sell']).max()<1e-6
        cool=np.array([v['cool'][0,0] for v in repaired])
        trade=np.array([v['g'][0,0] for v in repaired])
        assert max(float(abs(v[k][:,0]-v[k][0,0]).max()) for v in repaired for k in ('cool','g'))<1e-7
        return cool,trade,dict(iterations=iterations,primal=primal,dual=dual,proximal=proximal,
            central_objective=optimum,feasible_objective=objective,relative_objective_gap=gap,
            raw_horizon_clearing_kw=float(abs(gamma.sum(axis=0)).max()),max_trade_repair_kw=repair,
            raw_first_trade_kw=[float(v['g'][0,0]) for v in raw],
            center_reference_seconds=center_seconds,algorithm_and_repair_seconds=time.perf_counter()-algorithm_start)


def simulate(outdir,mode,method,n=3,case='sydney_utc10_end',start='2014-01-14',steps=144,warmup=48,
             scenario_count=None,forecast_fn=None,experiment=None):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    output=outdir/f'{mode}_{method}_{case}_n{n}_{start}_{steps}_w{warmup}.json'
    if output.exists():
        raise FileExistsError(output)
    cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text())
    data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'))
    p=cfg['controller']; th=cfg['thermal']; dt=cfg['dt_hours']; H=p['horizon']
    S=1 if mode=='deterministic' else (scenario_count or p['scenarios'])
    i0=int(np.searchsorted(data['time'],np.datetime64(start)))
    assert 0<=warmup<steps and 1<=n<=len(cfg['household_ids'])
    assert data['time'].max()<np.datetime64('2014-01-17')
    assert data['time'][i0]==np.datetime64(start) and i0>=60*48 and i0+steps<=len(data['time'])
    R=np.array(th['R_C_per_kw'][:n]); C=np.array(th['C_kwh_per_C'][:n])
    COP=np.array(th['COP'][:n]); gain=np.array(th['internal_gain_kw_thermal'][:n])
    cap=np.array(cfg['pv']['ac_kw'][:n])
    temp=np.full(n,th['initial_C']); previous=np.zeros(n)
    opt=Optimizer(cfg,n,S,method); records=[]
    payload=dict(status='running',mode=mode,method=method,scenarios=S,case=case,
        household_ids=cfg['household_ids'][:n],start=start,steps=steps,warmup=warmup,
        full_configuration=cfg,parameter_sha256=digest(ROOT/'formal_parameters_v1.json'),
        input_sha256=digest(ROOT/'data/formal_validation_inputs.npz'),
        runner_sha256=digest(__file__),core_sha256=digest(ROOT/'forecast_closed_loop.py'),
        environment=dict(python=platform.python_version(),cvxpy=cp.__version__,scipy=scipy.__version__,numpy=np.__version__),
        solver_settings=SOLVER,formal_test_run=False,experiment=experiment,
        numerical_acceptance=dict(primal_abs_kw=1e-4,relative=1e-6,dual_abs=1e-6,
            raw_clearing_limit_kw=MARKET_TOL_KW,iteration_limit=1000,feasible_relative_gap=1e-4),
        timing_note='Serial execution; PJ also solves an independent centralized reference every interval. No speedup claim.',records=records)
    begun=time.perf_counter()
    try:
        for step,t in enumerate(range(i0,i0+steps)):
            f=(forecast_fn or forecast)(data,t,case,n,H,S,cap,mode=mode)
            prices=price_at(np.arange(t,t+H),p)
            cool,trade,stats=opt.solve(f,temp,previous,prices)
            # Commit before accessing any current realized input.
            planned=f['point_load'][:,0]+cool-f['point_pv'][:,0]-trade
            pred=thermal_step(temp,f['point_ambient'][0],cool,R,C,COP,gain,dt)
            load=data['base_kw'][t,:n]; pv=data[case+'_pv'][t,:n]; amb=float(data[case+'_ambient'][t])
            actual=settle(load,pv,cool,trade,planned,prices[0],p,dt)
            nxt=thermal_step(temp,amb,cool,R,C,COP,gain,dt)
            a=np.exp(-dt/(R*C))
            assert abs(nxt-pred-(1-a)*(amb-f['point_ambient'][0])).max()<1e-9
            deviation=(load-f['point_load'][:,0])-(pv-f['point_pv'][:,0])
            assert abs(actual['deviation']-deviation).max()<1e-9
            balance=float(abs(load+cool-pv-actual['net']-trade).max())
            capacity=max(0,float(actual['buy'].max()-p['grid_import_max_kw']),float(actual['sell'].max()-p['grid_export_max_kw']))
            assert balance<1e-8 and abs(trade.sum())<1e-8 and capacity<1e-7
            low=np.maximum(th['comfort_low_C']-nxt,0); high=np.maximum(nxt-th['comfort_high_C'],0)
            row=dict(step=step,label=str(data['time'][t]),warmup=step<warmup,
                temperature_before_C=temp.tolist(),temperature_after_C=nxt.tolist(),
                predicted_temperature_C=pred.tolist(),cooling_kw=cool.tolist(),p2p_kw=trade.tolist(),
                actual_load_kw=load.tolist(),actual_pv_kw=pv.tolist(),ambient_actual_C=amb,
                forecast_load_kw=f['point_load'][:,0].tolist(),forecast_pv_kw=f['point_pv'][:,0].tolist(),
                ambient_forecast_C=float(f['point_ambient'][0]),max_source_index=f['max_source_index'],
                actual_grid_kw=actual['net'].tolist(),planned_grid_kw=planned.tolist(),balancing_kw=actual['deviation'].tolist(),
                grid_bill=actual['bill'],balancing_fee=actual['fee'],comfort_low_C=low.tolist(),comfort_high_C=high.tolist(),
                tracking_cost=float(dt*p['comfort_tracking_weight']*np.sum((nxt-th['setpoint_C'])**2)),
                comfort_penalty=float(dt*p['comfort_slack_AUD_per_degree_hour']*(low+high).sum()),
                smoothing_cost=float(p['smoothing_weight']*np.sum((cool-previous)**2)),
                balance_error_kw=balance,clearing_error_kw=float(abs(trade.sum())),capacity_violation_kw=capacity,**stats)
            records.append(row); temp=nxt; previous=cool
            if step%24==0:
                print(mode,method,n,step,'iterations',stats['iterations'],'T',temp.round(2),flush=True)
                output.with_suffix('.checkpoint.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        ev=records[warmup:]
        arr=lambda name:np.asarray([v[name] for v in ev])
        total=lambda name:float(arr(name).sum())
        summary=dict(evaluated_steps=len(ev),temperature_min_C=float(arr('temperature_after_C').min()),
            temperature_max_C=float(arr('temperature_after_C').max()),cold_degree_hours=dt*total('comfort_low_C'),
            hot_degree_hours=dt*total('comfort_high_C'),max_cold_C=float(arr('comfort_low_C').max()),
            max_hot_C=float(arr('comfort_high_C').max()),hvac_kwh=dt*total('cooling_kw'),
            grid_bill=total('grid_bill'),balancing_fee=total('balancing_fee'),
            comfort_penalty=total('comfort_penalty'),tracking_cost=total('tracking_cost'),smoothing_cost=total('smoothing_cost'),
            balancing_kwh=dt*float(np.abs(arr('balancing_kw')).sum()),
            max_iterations=max(v['iterations'] for v in records),
            max_relative_gap=max(v['relative_objective_gap'] for v in records),
            max_balance_error_kw=max(v['balance_error_kw'] for v in records),
            max_clearing_error_kw=max(v['clearing_error_kw'] for v in records),
            max_trade_repair_kw=max(v['max_trade_repair_kw'] for v in records))
        summary['monetary_bill']=summary['grid_bill']+summary['balancing_fee']
        summary['realized_stage_cost']=sum(summary[k] for k in ('monetary_bill','comfort_penalty','tracking_cost','smoothing_cost'))
        # Rolling terminal look-ahead values are not repeatedly charged to realized operation.
        payload.update(status='passed',summary=summary,elapsed_seconds=time.perf_counter()-begun)
    except Exception as error:
        payload.update(status='failed',error=repr(error),failure_step=len(records),elapsed_seconds=time.perf_counter()-begun)
        output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        raise
    output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    output.with_suffix('.checkpoint.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return payload


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['deterministic','scenario'],required=True)
    ap.add_argument('--method',choices=['central','pj'],required=True)
    ap.add_argument('--outdir',type=Path,required=True); ap.add_argument('--homes',type=int,default=3)
    ap.add_argument('--case',default='sydney_utc10_end'); ap.add_argument('--start',default='2014-01-14')
    ap.add_argument('--steps',type=int,default=144); ap.add_argument('--warmup',type=int,default=48)
    args=ap.parse_args()
    simulate(args.outdir,args.mode,args.method,args.homes,args.case,args.start,args.steps,args.warmup)
