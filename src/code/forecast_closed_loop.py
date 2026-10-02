"""Causal retrofit MPC validation; no final-test data and no current-interval oracle.

Cooling and P2P trades are fixed before the interval. Utility exchange is modeled
as fast balancing recourse; actual interval energy is settled after realization.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

ROOT=Path(__file__).resolve().parent
SOLVER=dict(solver='CLARABEL',tol_gap_abs=1e-11,tol_gap_rel=1e-11,tol_feas=1e-11,max_iter=300)
MARKET_TOL_KW=1e-3  # 1 W over each community trading component, before explicit repair.


def cleared_snapshot(home,raw,z):
    """Explicitly implement the coordinator's zero-sum trades, with grid recourse.

    This is a recorded numerical feasibility repair, never a silent replacement.
    No realized disturbance is used to construct these pre-interval actions.
    """
    v={k:a.copy() for k,a in raw.items()}
    v['g'][:,0]=z[0]; v['g'][:,1:]=z[1:].reshape(home.S,home.H-1)
    net=home.load.value+v['cool']-home.pv.value-v['g']
    v['buy']=np.maximum(net,0); v['sell']=np.maximum(-net,0)
    p=home.cfg['controller']
    violation=max(0,float(np.abs(v['g']).max()-p['trade_max_kw']),
                  float(v['buy'].max()-p['grid_import_max_kw']),float(v['sell'].max()-p['grid_export_max_kw']))
    assert violation<1e-6, 'Zero-sum repair violates a device/grid limit'
    return v


def price_at(index,p):
    hours=(np.asarray(index)%48)/2
    return np.where((hours>=p['peak_start_hour']) & (hours<p['peak_end_hour']),
                    p['buy_peak_AUD_per_kwh'],p['buy_offpeak_AUD_per_kwh'])


def forecast(data,t,key,n,H,S,ac_cap,mode='scenario'):
    """No realized value at index >=t is read; residual library ends Dec 30."""
    lag_indices=t+np.arange(H)-48
    train_end=int(np.searchsorted(data['time'],np.datetime64('2013-12-31')))
    assert 48<=t and lag_indices.max()<t and t>=train_end
    point_load=data['base_kw'][lag_indices,:n].T
    point_pv=data[key+'_pv'][lag_indices,:n].T
    point_ambient=data[key+'_ambient'][lag_indices]
    if mode=='deterministic':
        if S!=1:
            raise ValueError('Deterministic forecasts require exactly one point trajectory')
        return dict(load=point_load[:,None,:].copy(),pv=point_pv[:,None,:].copy(),
                    ambient=point_ambient[None,:].copy(),point_load=point_load,
                    point_pv=point_pv,point_ambient=point_ambient,
                    max_source_index=int(lag_indices.max()))
    if mode!='scenario':
        raise ValueError('Unknown forecast mode: '+mode)
    starts=np.arange(48+t%48,train_end-H+1,48)
    sampled=np.random.default_rng(20260912+t).choice(starts,size=S,replace=False)
    load=[]; pv=[]; ambient=[]
    for s in sampled:
        ix=s+np.arange(H)
        assert ix.max()<train_end and ix.max()<t
        load.append(np.maximum(0,point_load+(data['base_kw'][ix,:n]-data['base_kw'][ix-48,:n]).T))
        pv.append(np.clip(point_pv+(data[key+'_pv'][ix,:n]-data[key+'_pv'][ix-48,:n]).T,0,ac_cap[:,None]))
        ambient.append(point_ambient+data[key+'_ambient'][ix]-data[key+'_ambient'][ix-48])
    return dict(load=np.stack(load,axis=1),pv=np.stack(pv,axis=1),ambient=np.array(ambient),
                point_load=point_load,point_pv=point_pv,point_ambient=point_ambient,
                max_source_index=int(max(lag_indices.max(),sampled.max()+H-1)))


def thermal_step(temp,ambient,cool,R,C,COP,gain,dt):
    a=np.exp(-dt/(R*C))
    return a*temp+(1-a)*(ambient+R*(gain-COP*cool))


def settle(load,pv,cool,trade,planned,price,p,dt):
    net=load+cool-pv-trade
    buy=np.maximum(net,0); sell=np.maximum(-net,0)
    deviation=net-planned
    energy_bill=dt*np.sum(price*buy+p['quadratic_import_AUD_per_kw2h']*buy**2-p['sell_AUD_per_kwh']*sell)
    fee=dt*p['balancing_surcharge_AUD_per_kwh']*np.abs(deviation).sum()
    return dict(net=net,buy=buy,sell=sell,deviation=deviation,bill=float(energy_bill),fee=float(fee))


class Home:
    def __init__(self,n,cfg,S):
        self.n=n; self.cfg=cfg; p=cfg['controller']; th=cfg['thermal']
        self.H=H=p['horizon']; self.S=S; self.dt=dt=cfg['dt_hours']; self.D=1+(H-1)*S
        self.R=th['R_C_per_kw'][n]; self.C=th['C_kwh_per_C'][n]; self.COP=th['COP'][n]
        self.gain=th['internal_gain_kw_thermal'][n]; self.a=np.exp(-dt/(self.R*self.C))
        self.coolmax=th['cooling_max_kw_electric'][n]
        self.load=cp.Parameter((S,H),nonneg=True); self.pv=cp.Parameter((S,H),nonneg=True)
        self.ambient=cp.Parameter((S,H)); self.initial=cp.Parameter(); self.previous=cp.Parameter(nonneg=True)
        self.price=cp.Parameter(H,nonneg=True); self.balancing_constant=cp.Parameter(nonneg=True)
        self.cool=cp.Variable((S,H),nonneg=True); self.buy=cp.Variable((S,H),nonneg=True)
        self.sell=cp.Variable((S,H),nonneg=True); self.g=cp.Variable((S,H)); self.temp=cp.Variable((S,H+1))
        self.low=cp.Variable((S,H),nonneg=True); self.high=cp.Variable((S,H),nonneg=True)
        self.trade=cp.hstack([self.g[0,0:1],cp.reshape(self.g[:,1:],((H-1)*S,),order='C')])
        self.constraints=[self.temp[:,0]==self.initial,
            self.temp[:,1:]==self.a*self.temp[:,:-1]+(1-self.a)*(self.ambient+self.R*(self.gain-self.COP*self.cool)),
            self.cool<=self.coolmax,self.buy<=p['grid_import_max_kw'],self.sell<=p['grid_export_max_kw'],
            self.g>=-p['trade_max_kw'],self.g<=p['trade_max_kw'],
            self.load+self.cool-self.pv==self.buy-self.sell+self.g,
            self.temp[:,1:]+self.low>=th['comfort_low_C'],self.temp[:,1:]-self.high<=th['comfort_high_C']]
        if S>1:
            self.constraints += [self.cool[1:,0]==self.cool[0,0],self.g[1:,0]==self.g[0,0]]
        self.bill=dt/S*cp.sum(cp.multiply(self.price,self.buy)+p['quadratic_import_AUD_per_kw2h']*cp.square(self.buy)-p['sell_AUD_per_kwh']*self.sell)
        self.comfort=dt*p['comfort_tracking_weight']/S*cp.sum_squares(self.temp[:,1:]-th['setpoint_C'])
        self.slack=dt*p['comfort_slack_AUD_per_degree_hour']/S*cp.sum(self.low+self.high)
        self.smooth=p['smoothing_weight']/S*(cp.sum_squares(self.cool[:,0]-self.previous)+cp.sum_squares(self.cool[:,1:]-self.cool[:,:-1]))
        self.terminal=p['terminal_weight']/S*cp.sum_squares(self.temp[:,-1]-th['setpoint_C'])
        self.cost=self.bill+self.comfort+self.slack+self.smooth+self.terminal+self.balancing_constant
        self.target=cp.Parameter(self.D); self.old=cp.Parameter(self.D)
        self.problem=cp.Problem(cp.Minimize(self.cost+p['rho']/2*cp.sum_squares(self.trade-self.target)+p['beta']/2*cp.sum_squares(self.trade-self.old)),self.constraints)

    def set(self,f,initial,previous,price):
        n=self.n
        self.load.value=f['load'][n]; self.pv.value=f['pv'][n]; self.ambient.value=f['ambient']
        self.initial.value=float(initial); self.previous.value=float(max(0,previous)); self.price.value=price
        residual=f['load'][n]-f['pv'][n]-(f['point_load'][n]-f['point_pv'][n])[None,:]
        self.balancing_constant.value=float(self.dt*self.cfg['controller']['balancing_surcharge_AUD_per_kwh']*np.abs(residual).sum()/self.S)

    def snapshot(self):
        return {k:np.array(getattr(self,k).value,copy=True) for k in ['cool','buy','sell','g','temp','low','high']}

    def cost_check(self,v):
        p=self.cfg['controller']; th=self.cfg['thermal']; dt=self.dt
        bill=dt*np.mean(np.sum(self.price.value*v['buy']+p['quadratic_import_AUD_per_kw2h']*v['buy']**2-p['sell_AUD_per_kwh']*v['sell'],axis=1))
        comfort=dt*p['comfort_tracking_weight']*np.mean(np.sum((v['temp'][:,1:]-th['setpoint_C'])**2,axis=1))
        slack=dt*p['comfort_slack_AUD_per_degree_hour']*np.mean(np.sum(v['low']+v['high'],axis=1))
        smooth=p['smoothing_weight']*np.mean((v['cool'][:,0]-self.previous.value)**2+np.sum(np.diff(v['cool'],axis=1)**2,axis=1))
        terminal=p['terminal_weight']*np.mean((v['temp'][:,-1]-th['setpoint_C'])**2)
        return float(bill+comfort+slack+smooth+terminal+self.balancing_constant.value)


def run(key='sydney_utc10_end',steps=144,n=3,scenario_count=None,start_label='2014-01-14',warmup=48,rho_override=None):
    cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text())
    if rho_override is not None:
        cfg['controller']['rho']=float(rho_override)
    data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'))
    assert str(data['time'][-1])[:10]=='2014-01-16'
    p=cfg['controller']; th=cfg['thermal']; dt=cfg['dt_hours']; H=p['horizon']
    S=scenario_count or p['scenarios']; D=1+(H-1)*S
    homes=[Home(i,cfg,S) for i in range(n)]
    center=cp.Problem(cp.Minimize(sum(h.cost for h in homes)),sum([h.constraints for h in homes],[])+[sum(h.trade for h in homes)==0])
    initial=np.full(n,th['initial_C']); previous=np.zeros(n)
    z=np.zeros((n,D)); u=z.copy(); old=z.copy(); trace=[]; records=[]
    R=np.array(th['R_C_per_kw'][:n]); C=np.array(th['C_kwh_per_C'][:n]); COP=np.array(th['COP'][:n]); gain=np.array(th['internal_gain_kw_thermal'][:n])
    cap=np.array(cfg['pv']['ac_kw'][:n]); start=int(np.searchsorted(data['time'],np.datetime64(start_label)))
    assert start>=60*48 and start+steps<=len(data['time'])
    started=time.perf_counter()
    for step,t in enumerate(range(start,start+steps)):
        f=forecast(data,t,key,n,H,S,cap)
        prices=price_at(np.arange(t,t+H),p)
        for i,h in enumerate(homes): h.set(f,initial[i],previous[i],prices)
        center.solve(**SOLVER)
        if center.status!='optimal': raise RuntimeError(('central',step,center.status))
        optimum=float(center.value)
        assert abs(sum(h.cost_check(h.snapshot()) for h in homes)-optimum)<1e-5
        for iteration in range(1,1001):
            candidates=[]
            for i,h in enumerate(homes):
                h.target.value=z[i]-u[i]; h.old.value=old[i]
                h.problem.solve(warm_start=True,**SOLVER)
                if h.problem.status!='optimal': raise RuntimeError(('local',step,i,h.problem.status))
                candidates.append(h.trade.value.copy())
            gamma=np.array(candidates); v=gamma+u; zn=v-v.mean(axis=0,keepdims=True)
            primal=float(np.linalg.norm(gamma-zn)); dual=float(p['rho']*np.linalg.norm(zn-z))
            proximal=float(p['beta']*np.linalg.norm(gamma-old))
            u+=gamma-zn; old=gamma.copy(); z=zn
            ep=1e-4*np.sqrt(n*D)+1e-6*max(np.linalg.norm(gamma),np.linalg.norm(z))
            ed=1e-6*np.sqrt(n*D)+1e-6*np.linalg.norm(p['rho']*u)
            if step==warmup: trace.append([iteration,primal,dual,proximal])
            if primal<=ep and dual<=ed and proximal<=ed and np.abs(gamma.sum(axis=0)).max()<=MARKET_TOL_KW: break
        else:
            context=dict(step=step,index=t,initial=initial.tolist(),previous=previous.tolist(),
                forecast={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items()},
                cfg=cfg,gamma=gamma.tolist(),z=z.tolist(),u=u.tolist(),primal=primal,dual=dual)
            (ROOT/f'results/failure_context_{key}_rho{p["rho"]}.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
            raise RuntimeError(('ADMM iteration limit',step,primal,dual,proximal))
        raw_snaps=[h.snapshot() for h in homes]
        raw_objective=sum(h.cost_check(v) for h,v in zip(homes,raw_snaps))
        raw_clearing=float(np.abs(gamma.sum(axis=0)).max())
        snaps=[cleared_snapshot(h,v,z[i]) for i,(h,v) in enumerate(zip(homes,raw_snaps))]
        repair_max=float(max(np.abs(v['g']-r['g']).max() for v,r in zip(snaps,raw_snaps)))
        objective=sum(h.cost_check(v) for h,v in zip(homes,snaps))
        gap=abs(objective-optimum)/max(1,abs(optimum))
        assert objective>=optimum-1e-6*max(1,abs(optimum))
        # Actions fixed HERE, before realized interval load/PV/weather is read.
        cool=np.array([v['cool'][0,0] for v in snaps]); trade=np.array([v['g'][0,0] for v in snaps])
        planned=f['point_load'][:,0]+cool-f['point_pv'][:,0]-trade
        predicted_temp=thermal_step(initial,f['point_ambient'][0],cool,R,C,COP,gain,dt)
        actual_load=data['base_kw'][t,:n]; actual_pv=data[key+'_pv'][t,:n]; actual_ambient=float(data[key+'_ambient'][t])
        actual=settle(actual_load,actual_pv,cool,trade,planned,prices[0],p,dt)
        next_temp=thermal_step(initial,actual_ambient,cool,R,C,COP,gain,dt)
        # Independent thermal energy equation and explicit prediction-error identity.
        a=np.exp(-dt/(R*C)); eq=actual_ambient+R*(gain-COP*cool)
        independent=eq+(initial-eq)*a
        thermal_error=float(np.max(abs(next_temp-independent)))
        thermal_residual=next_temp-predicted_temp
        assert np.max(abs(thermal_residual-(1-a)*(actual_ambient-f['point_ambient'][0])))<1e-9
        balance=actual_load+cool-actual_pv-actual['buy']+actual['sell']-trade
        scheduled_residual=actual_load+cool-actual_pv-planned-trade
        assert np.max(abs(actual['deviation']-scheduled_residual))<1e-9
        assert np.max(abs(actual['deviation']-((actual_load-f['point_load'][:,0])-(actual_pv-f['point_pv'][:,0]))))<1e-9
        e_in=dt*actual['buy']; e_out=dt*actual['sell']
        independent_bill=float(np.sum(prices[0]*e_in+p['quadratic_import_AUD_per_kw2h']/dt*e_in**2-p['sell_AUD_per_kwh']*e_out))
        comfort_low=np.maximum(th['comfort_low_C']-next_temp,0); comfort_high=np.maximum(next_temp-th['comfort_high_C'],0)
        nonanticip=max(float(np.max(abs(v[k][:,0]-v[k][0,0]))) for v in snaps for k in ['cool','g'])
        constraint_residual=max(float(np.max(c.violation())) for h in homes for c in h.constraints)
        cap_violation=float(max(0,actual['buy'].max()-p['grid_import_max_kw'],actual['sell'].max()-p['grid_export_max_kw']))
        scenario_simultaneous=max(float(np.minimum(v['buy'],v['sell']).max()) for v in snaps)
        assert thermal_error<1e-9 and np.max(abs(balance))<1e-8
        assert abs(trade.sum())<1e-4 and gap<1e-4 and nonanticip<1e-7 and constraint_residual<1e-6
        assert cap_violation<1e-7 and scenario_simultaneous<1e-5 and abs(independent_bill-actual['bill'])<1e-9
        row=dict(step=step,label=str(data['time'][t]),warmup=step<warmup,
            temperature_before_C=initial.tolist(),temperature_after_C=next_temp.tolist(),predicted_temperature_C=predicted_temp.tolist(),
            ambient_actual_C=actual_ambient,ambient_forecast_C=float(f['point_ambient'][0]),
            actual_load_kw=actual_load.tolist(),forecast_load_kw=f['point_load'][:,0].tolist(),
            actual_pv_kw=actual_pv.tolist(),forecast_pv_kw=f['point_pv'][:,0].tolist(),
            cooling_kw=cool.tolist(),p2p_kw=trade.tolist(),planned_grid_kw=planned.tolist(),
            actual_grid_kw=actual['net'].tolist(),balancing_kw=actual['deviation'].tolist(),
            grid_bill=actual['bill'],balancing_fee=actual['fee'],
            comfort_low_C=comfort_low.tolist(),comfort_high_C=comfort_high.tolist(),
            comfort_tracking_cost=float(dt*p['comfort_tracking_weight']*np.sum((next_temp-th['setpoint_C'])**2)),
            comfort_slack_cost=float(dt*p['comfort_slack_AUD_per_degree_hour']*(comfort_low+comfort_high).sum()),
            smooth_cost=float(p['smoothing_weight']*np.sum((cool-previous)**2)),
            iterations=iteration,primal=primal,dual=dual,proximal=proximal,relative_objective_gap=gap,
            raw_local_objective=raw_objective,raw_horizon_clearing_max_kw=raw_clearing,
            trade_repair_max_kw=repair_max,raw_first_step_p2p_kw=[float(v['g'][0,0]) for v in raw_snaps],
            central_objective=optimum,distributed_objective=objective,balance_max_kw=float(np.max(abs(balance))),
            market_clearing_kw=float(abs(trade.sum())),thermal_equation_error_C=thermal_error,
            nonanticipativity=nonanticip,horizon_constraint_error=constraint_residual,
            actual_grid_capacity_violation_kw=cap_violation,max_forecast_source_index=f['max_source_index'])
        records.append(row); initial=next_temp; previous=cool
        if step%24==0: print(key,step,'iterations',iteration,'T',next_temp.round(2),'balance deviation',actual['deviation'].round(3),flush=True)
    evaluated=[r for r in records if not r['warmup']]
    assert evaluated
    temps=np.array([r['temperature_after_C'] for r in evaluated])
    errs=lambda a,b: np.array([r[a] for r in evaluated])-np.array([r[b] for r in evaluated])
    summary=dict(code_checks_passed=True,steps=steps,warmup_steps=warmup,evaluated_steps=len(evaluated),
        temperature_min_C=float(temps.min()),temperature_max_C=float(temps.max()),
        comfort_lower_degree_hours=float(dt*np.array([r['comfort_low_C'] for r in evaluated]).sum()),
        comfort_upper_degree_hours=float(dt*np.array([r['comfort_high_C'] for r in evaluated]).sum()),
        max_cold_violation_C=float(np.array([r['comfort_low_C'] for r in evaluated]).max()),
        max_hot_violation_C=float(np.array([r['comfort_high_C'] for r in evaluated]).max()),
        load_forecast_mae_kw=float(np.abs(errs('actual_load_kw','forecast_load_kw')).mean()),
        pv_forecast_mae_kw=float(np.abs(errs('actual_pv_kw','forecast_pv_kw')).mean()),
        ambient_forecast_mae_C=float(np.abs(errs('ambient_actual_C','ambient_forecast_C')).mean()),
        indoor_one_step_forecast_mae_C=float(np.abs(errs('temperature_after_C','predicted_temperature_C')).mean()),
        balancing_energy_kwh=float(dt*np.abs(np.array([r['balancing_kw'] for r in evaluated])).sum()),
        hvac_energy_kwh=float(dt*np.array([r['cooling_kw'] for r in evaluated]).sum()),
        total_grid_bill=sum(r['grid_bill'] for r in evaluated),total_balancing_fee=sum(r['balancing_fee'] for r in evaluated),
        max_balance_error_kw=max(r['balance_max_kw'] for r in records),
        max_clearing_error_kw=max(r['market_clearing_kw'] for r in records),
        max_thermal_error_C=max(r['thermal_equation_error_C'] for r in records),
        max_objective_gap=max(r['relative_objective_gap'] for r in records),max_iterations=max(r['iterations'] for r in records),
        max_raw_horizon_clearing_kw=max(r['raw_horizon_clearing_max_kw'] for r in records),
        max_trade_repair_kw=max(r['trade_repair_max_kw'] for r in records),
        elapsed_seconds=time.perf_counter()-started)
    payload=dict(case=key,household_ids=cfg['household_ids'][:n],scenarios=S,rho_used=p['rho'],beta_used=p['beta'],
        validation_override=dict(rho=rho_override),
        parameter_sha256=hashlib.sha256((ROOT/'formal_parameters_v1.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(python=platform.python_version(),cvxpy=cp.__version__,solver_settings=SOLVER),full_configuration=cfg,
        numerical_acceptance=dict(market_pre_repair_tolerance_kw=MARKET_TOL_KW,primal_abs_tolerance_kw=1e-4,
            primal_relative_tolerance=1e-6,dual_abs_relative_tolerance=1e-6,
            final_trade='Explicit coordinator projection plus scenario grid recourse; repaired feasibility and objective independently checked'),
        status='validation only; final Jan17-29 test not run; serial synchronous PJ-ADMM, not parallel timing',
        summary=summary,records=records,first_evaluated_step_convergence=trace)
    suffix='' if rho_override is None else '_rho'+str(rho_override).replace('.','p')
    path=ROOT/f'results/causal_{key}_n{n}_s{S}_{steps}{suffix}_balanced.json'
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    return payload


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--case',default='sydney_utc10_end'); ap.add_argument('--steps',type=int,default=144)
    ap.add_argument('--homes',type=int,default=3); ap.add_argument('--scenarios',type=int); ap.add_argument('--warmup',type=int,default=48)
    ap.add_argument('--start',default='2014-01-14'); ap.add_argument('--rho',type=float); args=ap.parse_args()
    try:
        run(args.case,args.steps,args.homes,args.scenarios,args.start,args.warmup,args.rho)
    except (RuntimeError,AssertionError) as error:
        fail=ROOT/f'results/causal_failure_{args.case}_rho{args.rho}.json'
        fail.write_text(json.dumps(dict(arguments=vars(args),error=repr(error)),indent=2),encoding='utf-8')
        raise
