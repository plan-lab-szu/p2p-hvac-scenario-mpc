"""Exact common-control mean model with repeated controls/states eliminated.

No risk or information-structure change. Mean temperature plus known scenario
offsets gives exactly the original per-scenario RC trajectory. Only nonnegative
grid purchase and discomfort epigraphs remain scenario-expanded variables.
"""
import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from robust_mpc import RISK_SOLVER
from uncached_central_optimizer import DirectProblem
from compact_open_loop import expand_common


class ReducedMeanQP:
    def __init__(self,cfg,n,S,method='central'):
        if method!='central':raise ValueError(method)
        self.cfg=cfg;self.n=n;self.S=S;self.H=H=cfg['controller']['horizon'];p=cfg['controller'];th=cfg['thermal'];dt=cfg['dt_hours']
        repeat=sp.csr_matrix((np.ones(n*S),(np.arange(n*S),np.repeat(np.arange(n),S))),shape=(n*S,n))
        self.base=cp.Parameter((n*S,H));self.offset=cp.Parameter((n*S,H));self.meanambient=cp.Parameter(H)
        self.netlo=cp.Parameter((n,H));self.nethi=cp.Parameter((n,H));self.initial=cp.Parameter(n);self.previous=cp.Parameter(n)
        self.spread=cp.Parameter(H,nonneg=True)
        self.cool=cp.Variable((n,H),nonneg=True);self.trade=cp.Variable((n,H));self.meanT=cp.Variable((n,H+1))
        buy=cp.Variable((n*S,H),nonneg=True);slack=cp.Variable((n*S,H),nonneg=True)
        R=np.array(th['R_C_per_kw'][:n])[:,None];C=np.array(th['C_kwh_per_C'][:n])[:,None];a=np.exp(-dt/(R*C))
        cop=np.array(th['COP'][:n])[:,None];gain=np.array(th['internal_gain_kw_thermal'][:n])[:,None]
        net=self.base+repeat@(self.cool-self.trade);T=repeat@self.meanT[:,1:]+self.offset
        constraints=[self.meanT[:,0]==self.initial,
            self.meanT[:,1:]==cp.multiply(a,self.meanT[:,:-1])+cp.multiply(1-a,self.meanambient+cp.multiply(R,gain-cp.multiply(cop,self.cool))),
            self.cool<=np.array(th['cooling_max_kw_electric'][:n])[:,None],cp.sum(self.trade,axis=0)==0,
            self.trade<=p['trade_max_kw'],self.trade>=-p['trade_max_kw'],
            self.nethi+self.cool-self.trade<=p['grid_import_max_kw'],self.netlo+self.cool-self.trade>=-p['grid_export_max_kw'],
            buy>=net,slack>=th['comfort_low_C']-th['setpoint_C']-T,slack>=T-(th['comfort_high_C']-th['setpoint_C'])]
        bill=dt/S*(p['sell_AUD_per_kwh']*cp.sum(net)+cp.sum(cp.multiply(self.spread,buy))+p['quadratic_import_AUD_per_kw2h']*cp.sum_squares(buy))
        comfort=dt*(p['comfort_tracking_weight']*cp.sum_squares(self.meanT[:,1:])+p['comfort_slack_AUD_per_degree_hour']/S*cp.sum(slack))
        smooth=p['smoothing_weight']*(cp.sum_squares(self.cool[:,0]-self.previous)+cp.sum_squares(self.cool[:,1:]-self.cool[:,:-1]))
        terminal=p['terminal_weight']*cp.sum_squares(self.meanT[:,-1])
        self.problem=DirectProblem(cp.Minimize((bill+comfort+smooth+terminal)/(n*H)),constraints)
        assert self.problem.is_qp()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter();cfg=self.cfg;th=cfg['thermal'];p=cfg['controller'];dt=cfg['dt_hours'];n=self.n;S=self.S;H=self.H
        mean=f['ambient'].mean(axis=0);delta=np.zeros((n,S,H+1));R=np.array(th['R_C_per_kw'][:n]);C=np.array(th['C_kwh_per_C'][:n]);a=np.exp(-dt/(R*C))
        for k in range(H):delta[:,:,k+1]=a[:,None]*delta[:,:,k]+(1-a[:,None])*(f['ambient'][:,k]-mean[k])[None,:]
        assert abs(delta.mean(axis=1)).max()<1e-10
        netbase=f['load']-f['pv'];nominal=f['point_load']-f['point_pv']
        fees=dt*p['balancing_surcharge_AUD_per_kwh']*abs(netbase-nominal[:,None,:]).sum(axis=(0,2))
        variance=dt*p['comfort_tracking_weight']*np.sum(np.mean(delta[:,:,1:]**2,axis=1))+p['terminal_weight']*np.sum(np.mean(delta[:,:,-1]**2,axis=1))
        self.base.value=netbase.reshape(n*S,H);self.offset.value=delta[:,:,1:].reshape(n*S,H)
        self.meanambient.value=mean-th['setpoint_C'];self.netlo.value=netbase.min(axis=1);self.nethi.value=netbase.max(axis=1)
        self.initial.value=temp-th['setpoint_C'];self.previous.value=np.maximum(previous,0);self.spread.value=prices-p['sell_AUD_per_kwh']
        self.problem.solve(**RISK_SOLVER)
        if self.problem.status!='optimal':raise RuntimeError(('reduced mean QP',self.problem.status))
        raw=self.trade.value.copy();trade=raw-raw.mean(axis=0,keepdims=True);cool=self.cool.value.copy()
        snapshots=expand_common(cfg,f,temp,cool,trade);costs=fees.copy()
        for j,v in enumerate(snapshots):
            assert v['buy'].max()<p['grid_import_max_kw']+1e-6 and v['sell'].max()<p['grid_export_max_kw']+1e-6
            assert abs(v['g']).max()<p['trade_max_kw']+1e-6
            assert cool[j].min()>=-1e-7 and cool[j].max()<th['cooling_max_kw_electric'][j]+1e-7
            np.testing.assert_allclose(v['temp'].mean(axis=0),self.meanT.value[j]+th['setpoint_C'],atol=1e-7,rtol=0)
            costs+=dt*np.sum(prices*v['buy']+p['quadratic_import_AUD_per_kw2h']*v['buy']**2-p['sell_AUD_per_kwh']*v['sell'],axis=1)
            costs+=dt*(p['comfort_tracking_weight']*((v['temp'][:,1:]-th['setpoint_C'])**2).sum(axis=1)+p['comfort_slack_AUD_per_degree_hour']*(v['low']+v['high']).sum(axis=1))
            costs+=p['smoothing_weight']*((cool[j,0]-previous[j])**2+np.diff(cool[j])@np.diff(cool[j]))
            costs+=p['terminal_weight']*(v['temp'][:,-1]-th['setpoint_C'])**2
        optimum=float(self.problem.value)*n*H+float(variance+fees.mean());value=float(costs.mean());gap=abs(value-optimum)/max(1,abs(optimum));assert gap<1e-6
        return cool[:,0],trade[:,0],dict(iterations=0,primal=0.,dual=0.,proximal=0.,central_objective=optimum,feasible_objective=value,
            relative_objective_gap=gap,raw_horizon_clearing_kw=float(abs(raw.sum(axis=0)).max()),max_trade_repair_kw=float(abs(trade-raw).max()),
            raw_first_trade_kw=raw[:,0].tolist(),center_reference_seconds=time.perf_counter()-start,algorithm_and_repair_seconds=0.,
            risk='mean',expected_horizon_cost=value,worst_horizon_cost=float(costs.max()),scenario_community_costs=costs.tolist(),
            common_control_certificate=dict(cooling_kw=cool.tolist(),p2p_kw=trade.tolist()),
            formulation='Exact reduced common-control mean QP; known temperature variance and balancing constants restored in objective',
            objective_scaling_divisor=float(n*H),risk_solver_settings=RISK_SOLVER)
