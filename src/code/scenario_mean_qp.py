"""Vectorized exact mean two-stage scenario QP; only first controls common."""
import time
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from robust_mpc import RISK_SOLVER
from uncached_central_optimizer import DirectProblem


class ScenarioMeanQP:
    def __init__(self,cfg,n,S,method='central'):
        if method!='central':raise ValueError(method)
        self.cfg=cfg;self.n=n;self.S=S;self.H=H=cfg['controller']['horizon'];p=cfg['controller'];th=cfg['thermal'];dt=cfg['dt_hours'];K=n*S
        self.base=cp.Parameter((K,H));self.ambient=cp.Parameter((K,H));self.initial=cp.Parameter(K);self.previous=cp.Parameter(K);self.spread=cp.Parameter(H,nonneg=True)
        self.cool=cp.Variable((K,H),nonneg=True);self.trade=cp.Variable((K,H));self.T=cp.Variable((K,H+1));buy=cp.Variable((K,H),nonneg=True);slack=cp.Variable((K,H),nonneg=True)
        R=np.repeat(th['R_C_per_kw'][:n],S)[:,None];C=np.repeat(th['C_kwh_per_C'][:n],S)[:,None];a=np.exp(-dt/(R*C));cop=np.repeat(th['COP'][:n],S)[:,None];gain=np.repeat(th['internal_gain_kw_thermal'][:n],S)[:,None]
        aggregate=sp.csr_matrix((np.ones(K),(np.tile(np.arange(S),n),np.arange(K))),shape=(S,K));net=self.base+self.cool-self.trade
        cons=[self.T[:,0]==self.initial,self.T[:,1:]==cp.multiply(a,self.T[:,:-1])+cp.multiply(1-a,self.ambient+cp.multiply(R,gain-cp.multiply(cop,self.cool))),
            self.cool<=np.repeat(th['cooling_max_kw_electric'][:n],S)[:,None],self.trade<=p['trade_max_kw'],self.trade>=-p['trade_max_kw'],
            net<=p['grid_import_max_kw'],net>=-p['grid_export_max_kw'],buy>=net,
            slack>=th['comfort_low_C']-th['setpoint_C']-self.T[:,1:],slack>=self.T[:,1:]-(th['comfort_high_C']-th['setpoint_C']),
            cp.sum(self.trade[::S,0])==0,aggregate@self.trade[:,1:]==0]
        if S>1:
            index=np.concatenate([np.arange(j*S+1,(j+1)*S) for j in range(n)]);first=np.repeat(np.arange(n)*S,S-1)
            cons+=[self.cool[index,0]==self.cool[first,0],self.trade[index,0]==self.trade[first,0]]
        bill=dt/S*(p['sell_AUD_per_kwh']*cp.sum(net)+cp.sum(cp.multiply(self.spread,buy))+p['quadratic_import_AUD_per_kw2h']*cp.sum_squares(buy))
        comfort=dt/S*(p['comfort_tracking_weight']*cp.sum_squares(self.T[:,1:])+p['comfort_slack_AUD_per_degree_hour']*cp.sum(slack))
        smooth=p['smoothing_weight']/S*(cp.sum_squares(self.cool[:,0]-self.previous)+cp.sum_squares(self.cool[:,1:]-self.cool[:,:-1]))
        terminal=p['terminal_weight']/S*cp.sum_squares(self.T[:,-1])
        self.problem=DirectProblem(cp.Minimize((bill+comfort+smooth+terminal)/(n*H)),cons);assert self.problem.is_qp()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter();n=self.n;S=self.S;H=self.H;cfg=self.cfg;th=cfg['thermal'];p=cfg['controller'];dt=cfg['dt_hours']
        base=f['load']-f['pv'];self.base.value=base.reshape(n*S,H);self.ambient.value=np.tile(f['ambient']-th['setpoint_C'],(n,1))
        self.initial.value=np.repeat(temp-th['setpoint_C'],S);self.previous.value=np.repeat(np.maximum(previous,0),S);self.spread.value=prices-p['sell_AUD_per_kwh']
        self.problem.solve(**RISK_SOLVER)
        if self.problem.status!='optimal':raise RuntimeError(('scenario mean QP',self.problem.status))
        cool=self.cool.value.reshape(n,S,H).copy();raw=self.trade.value.reshape(n,S,H).copy()
        cool[:,:,0]=cool[:,0:1,0];trade=raw.copy();trade[:,:,0]=raw[:,0:1,0];trade-=trade.mean(axis=0,keepdims=True)
        T=np.empty((n,S,H+1));T[:,:,0]=temp[:,None];R=np.array(th['R_C_per_kw'][:n]);a=np.exp(-dt/(R*np.array(th['C_kwh_per_C'][:n])))
        for k in range(H):T[:,:,k+1]=a[:,None]*T[:,:,k]+(1-a[:,None])*(f['ambient'][None,:,k]+R[:,None]*(np.array(th['internal_gain_kw_thermal'][:n])[:,None]-np.array(th['COP'][:n])[:,None]*cool[:,:,k]))
        np.testing.assert_allclose(T.reshape(n*S,H+1),self.T.value+th['setpoint_C'],atol=1e-7,rtol=0)
        net=base+cool-trade;buy=np.maximum(net,0);sell=np.maximum(-net,0)
        assert net.max()<p['grid_import_max_kw']+1e-6 and net.min()>-p['grid_export_max_kw']-1e-6 and abs(trade).max()<p['trade_max_kw']+1e-6
        assert cool.min()>=-1e-7 and (cool-np.array(th['cooling_max_kw_electric'][:n])[:,None,None]).max()<1e-7
        fees=dt*p['balancing_surcharge_AUD_per_kwh']*abs(base-(f['point_load']-f['point_pv'])[:,None,:]).sum(axis=(0,2))
        costs=fees+dt*(prices*buy+p['quadratic_import_AUD_per_kw2h']*buy**2-p['sell_AUD_per_kwh']*sell).sum(axis=(0,2))
        costs+=dt*p['comfort_tracking_weight']*((T[:,:,1:]-th['setpoint_C'])**2).sum(axis=(0,2))
        costs+=dt*p['comfort_slack_AUD_per_degree_hour']*(np.maximum(th['comfort_low_C']-T[:,:,1:],0)+np.maximum(T[:,:,1:]-th['comfort_high_C'],0)).sum(axis=(0,2))
        costs+=p['smoothing_weight']*(((cool[:,:,0]-previous[:,None])**2).sum(axis=0)+(np.diff(cool,axis=2)**2).sum(axis=(0,2)))
        costs+=p['terminal_weight']*((T[:,:,-1]-th['setpoint_C'])**2).sum(axis=0)
        value=float(costs.mean());optimum=float(self.problem.value)*n*H+float(fees.mean());gap=abs(value-optimum)/max(1,abs(optimum));assert gap<1e-6
        return cool[:,0,0],trade[:,0,0],dict(iterations=0,primal=0.,dual=0.,proximal=0.,central_objective=optimum,feasible_objective=value,relative_objective_gap=gap,
            raw_horizon_clearing_kw=float(abs(raw.sum(axis=0)).max()),max_trade_repair_kw=float(abs(raw-trade).max()),raw_first_trade_kw=raw[:,0,0].tolist(),
            center_reference_seconds=time.perf_counter()-start,algorithm_and_repair_seconds=0.,risk='mean',expected_horizon_cost=value,worst_horizon_cost=float(costs.max()),scenario_community_costs=costs.tolist(),
            scenario_control_certificate=dict(cooling_kw=cool.tolist(),p2p_kw=trade.tolist()),formulation='Exact vectorized mean two-stage scenario QP; full future scenario recourse, common first control only')
