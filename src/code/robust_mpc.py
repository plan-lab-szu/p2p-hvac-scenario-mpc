"""Mean versus community min-max, shared empirical hull and two-stage recourse.

Only first cooling/trading decisions are non-anticipative, exactly as the existing
scenario MPC. This is NOT a causal multistage robust policy or a field guarantee.
Worst is max_s sum_n J_ns, never sum_n max_s J_ns. Balancing fees are scenario
specific inside the maximum, even though their mean is a dispatch-independent
constant in expected-cost MPC. No base model files or parameters are modified.
"""
import time
import numpy as np
import cvxpy as cp
from forecast_closed_loop import Home,SOLVER,cleared_snapshot
from uncached_central_optimizer import DirectProblem

RISK_SOLVER=dict(SOLVER,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9)


def community_risk(costs,risk):
    community=np.asarray(costs).sum(axis=0)
    if risk=='mean':return float(community.mean())
    if risk=='worst':return float(community.max())
    raise ValueError(risk)


def scenario_costs(h,v,balancing):
    p=h.cfg['controller'];th=h.cfg['thermal'];dt=h.dt
    return (dt*np.sum(h.price.value*v['buy']+p['quadratic_import_AUD_per_kw2h']*v['buy']**2-p['sell_AUD_per_kwh']*v['sell'],axis=1)
        +dt*p['comfort_tracking_weight']*np.sum((v['temp'][:,1:]-th['setpoint_C'])**2,axis=1)
        +dt*p['comfort_slack_AUD_per_degree_hour']*np.sum(v['low']+v['high'],axis=1)
        +p['smoothing_weight']*((v['cool'][:,0]-h.previous.value)**2+np.sum(np.diff(v['cool'],axis=1)**2,axis=1))
        +p['terminal_weight']*(v['temp'][:,-1]-th['setpoint_C'])**2+balancing)


class RiskOptimizer:
    def __init__(self,cfg,n,S,method='central',risk='worst',save_certificate=False):
        if method!='central' or risk not in ('mean','worst'):raise ValueError((method,risk))
        self.cfg=cfg;self.risk=risk;self.save_certificate=save_certificate
        self.homes=[Home(i,cfg,S) for i in range(n)];self.fees=[];costs=[]
        p=cfg['controller'];th=cfg['thermal'];dt=cfg['dt_hours']
        for h in self.homes:
            fee=cp.Parameter(S,nonneg=True);self.fees.append(fee)
            cost=(dt*cp.sum(cp.multiply(h.price,h.buy)+p['quadratic_import_AUD_per_kw2h']*cp.square(h.buy)-p['sell_AUD_per_kwh']*h.sell,axis=1)
                +dt*p['comfort_tracking_weight']*cp.sum(cp.square(h.temp[:,1:]-th['setpoint_C']),axis=1)
                +dt*p['comfort_slack_AUD_per_degree_hour']*cp.sum(h.low+h.high,axis=1)
                +p['smoothing_weight']*(cp.square(h.cool[:,0]-h.previous)+cp.sum(cp.square(h.cool[:,1:]-h.cool[:,:-1]),axis=1))
                +p['terminal_weight']*cp.square(h.temp[:,-1]-th['setpoint_C'])+fee)
            costs.append(cost)
        self.costs=costs;self.community=sum(costs)
        # Fixed unit scaling, identical feasible set and optimizer in AUD.
        self.objective_scale=float(n*cfg['controller']['horizon'])
        constraints=sum([h.constraints for h in self.homes],[])+[sum(h.trade for h in self.homes)==0]
        if risk=='worst':
            self.eta=cp.Variable();constraints+=[self.community/self.objective_scale<=self.eta];objective=self.eta
        else:objective=cp.sum(self.community)/(S*self.objective_scale)
        self.problem=DirectProblem(cp.Minimize(objective),constraints)
        assert self.problem.is_dcp()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter();p=self.cfg['controller'];dt=self.cfg['dt_hours'];th=self.cfg['thermal']
        for i,(h,fee) in enumerate(zip(self.homes,self.fees)):
            h.set(f,temp[i],previous[i],prices)
            residual=f['load'][i]-f['pv'][i]-(f['point_load'][i]-f['point_pv'][i])[None,:]
            fee.value=dt*p['balancing_surcharge_AUD_per_kwh']*np.abs(residual).sum(axis=1)
        self.problem.solve(**RISK_SOLVER)
        if self.problem.status!='optimal':raise RuntimeError(('risk optimizer status',self.risk,self.problem.status))
        optimum=float(self.problem.value)*self.objective_scale
        raw=[h.snapshot() for h in self.homes]
        raw_violation=max(float(np.max(c.violation())) for h in self.homes for c in h.constraints)
        assert raw_violation<1e-6
        raw_cost=np.array([scenario_costs(h,v,b.value) for h,v,b in zip(self.homes,raw,self.fees)])
        np.testing.assert_allclose(raw_cost,np.array([x.value for x in self.costs]),atol=1e-7,rtol=1e-9)
        gamma=np.array([h.trade.value for h in self.homes]);z=gamma-gamma.mean(axis=0,keepdims=True)
        snapshots=[]
        for i,(h,v) in enumerate(zip(self.homes,raw)):
            v=cleared_snapshot(h,v,z[i])
            # Epigraph non-worst scenarios need not have tight auxiliaries.
            # Keep all controls/states; use physical net settlement/minimal slacks.
            v['low']=np.maximum(th['comfort_low_C']-v['temp'][:,1:],0)
            v['high']=np.maximum(v['temp'][:,1:]-th['comfort_high_C'],0)
            assert max(float(abs(v[k][:,0]-v[k][0,0]).max()) for k in ('cool','g'))<1e-7
            snapshots.append(v)
        costs=np.array([scenario_costs(h,v,b.value) for h,v,b in zip(self.homes,snapshots,self.fees)])
        community=costs.sum(axis=0);value=community_risk(costs,self.risk)
        gap=abs(value-optimum)/max(1,abs(optimum))
        assert gap<1e-6 and value>=optimum-1e-6*max(1,abs(optimum))
        assert abs(sum(v['g'] for v in snapshots)).max()<1e-8
        self.last=dict(f=f,snapshots=snapshots,costs=costs,raw_costs=raw_cost,objective=value)
        stats=dict(iterations=0,primal=0.,dual=0.,proximal=0.,central_objective=optimum,feasible_objective=value,
            relative_objective_gap=gap,raw_horizon_clearing_kw=float(abs(gamma.sum(axis=0)).max()),
            max_trade_repair_kw=max(float(abs(v['g']-a['g']).max()) for v,a in zip(snapshots,raw)),
            raw_first_trade_kw=[float(v['g'][0,0]) for v in raw],center_reference_seconds=time.perf_counter()-start,
            algorithm_and_repair_seconds=0.,risk=self.risk,expected_horizon_cost=float(community.mean()),
            worst_horizon_cost=float(community.max()),scenario_community_costs=community.tolist(),
            raw_constraint_violation=raw_violation,
            objective_scaling_divisor=self.objective_scale,
            risk_solver_settings=RISK_SOLVER,
            timing_scope='Central mean/min-max solve, not a distributed reference or speed comparison.')
        if self.save_certificate:
            stats['risk_certificate']=[{k:a.tolist() for k,a in v.items()} for v in snapshots]
        return np.array([v['cool'][0,0] for v in snapshots]),np.array([v['g'][0,0] for v in snapshots]),stats
