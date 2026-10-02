"""Conservative box cost-upper-bound MPC with open-loop horizon controls.

Not exact adjustable min-max. All endpoint limits hold throughout the continuous
box for fixed controls. Sum of thermal endpoint maxima can be conservative.
"""
import time
import numpy as np
import cvxpy as cp
from robust_mpc import RISK_SOLVER
from uncached_central_optimizer import DirectProblem


def thermal_cost(T,th,p):
    return p['comfort_tracking_weight']*(T-th['setpoint_C'])**2+p['comfort_slack_AUD_per_degree_hour']*(np.maximum(th['comfort_low_C']-T,0)+np.maximum(T-th['comfort_high_C'],0))


def envelope_values(cfg,f,cool,trade,temp,previous,prices):
    n,H=cool.shape;th=cfg['thermal'];p=cfg['controller'];dt=cfg['dt_hours'];lo=f['box_low'];hi=f['box_high']
    R=np.array(th['R_C_per_kw'][:n]);C=np.array(th['C_kwh_per_C'][:n]);cop=np.array(th['COP'][:n]);gain=np.array(th['internal_gain_kw_thermal'][:n]);a=np.exp(-dt/(R*C))
    Tlo=np.empty((n,H+1));Thi=np.empty_like(Tlo);Tlo[:,0]=Thi[:,0]=temp
    for k in range(H):
        Tlo[:,k+1]=a*Tlo[:,k]+(1-a)*(lo[-1,k]+R*(gain-cop*cool[:,k]))
        Thi[:,k+1]=a*Thi[:,k]+(1-a)*(hi[-1,k]+R*(gain-cop*cool[:,k]))
    dlow=lo[:n]-hi[n:2*n];dhigh=hi[:n]-lo[n:2*n]
    gridlo=dlow+cool-trade;gridhi=dhigh+cool-trade
    positive=np.maximum(gridhi,0)
    bill=dt*np.sum(p['sell_AUD_per_kwh']*gridhi+(prices-p['sell_AUD_per_kwh'])*positive+p['quadratic_import_AUD_per_kw2h']*positive**2)
    nominal=f['point_load']-f['point_pv']
    fee=dt*p['balancing_surcharge_AUD_per_kwh']*np.maximum(abs(dlow-nominal),abs(dhigh-nominal)).sum()
    comfort=dt*np.maximum(thermal_cost(Tlo[:,1:],th,p),thermal_cost(Thi[:,1:],th,p)).sum()
    smooth=p['smoothing_weight']*(np.sum((cool[:,0]-previous)**2)+np.sum(np.diff(cool,axis=1)**2))
    terminal=p['terminal_weight']*np.maximum((Tlo[:,-1]-th['setpoint_C'])**2,(Thi[:,-1]-th['setpoint_C'])**2).sum()
    return dict(Tlo=Tlo,Thi=Thi,gridlo=gridlo,gridhi=gridhi,bill_upper=float(bill),fee_upper=float(fee),
        comfort_upper=float(comfort),smoothing=float(smooth),terminal_upper=float(terminal),total=float(bill+fee+comfort+smooth+terminal))


class BoxEnvelopeOptimizer:
    def __init__(self,cfg,n,S=30,method='central'):
        if method!='central':raise ValueError(method)
        self.cfg=cfg;self.n=n;H=cfg['controller']['horizon'];th=cfg['thermal'];p=cfg['controller'];dt=cfg['dt_hours']
        self.low=cp.Parameter((2*n+1,H));self.high=cp.Parameter((2*n+1,H));self.initial=cp.Parameter(n)
        self.previous=cp.Parameter(n);self.prices=cp.Parameter(H,nonneg=True);self.fee=cp.Parameter(nonneg=True)
        self.cool=cp.Variable((n,H),nonneg=True);self.trade=cp.Variable((n,H));self.Tlo=cp.Variable((n,H+1));self.Thi=cp.Variable((n,H+1))
        cap=np.array(th['cooling_max_kw_electric'][:n])[:,None]
        R=np.array(th['R_C_per_kw'][:n])[:,None];C=np.array(th['C_kwh_per_C'][:n])[:,None];a=np.exp(-dt/(R*C))
        cop=np.array(th['COP'][:n])[:,None];gain=np.array(th['internal_gain_kw_thermal'][:n])[:,None]
        constraints=[self.cool<=cap,self.trade>=-p['trade_max_kw'],self.trade<=p['trade_max_kw'],cp.sum(self.trade,axis=0)==0,
            self.Tlo[:,0]==self.initial,self.Thi[:,0]==self.initial]
        for T,ambient in ((self.Tlo,self.low[-1,:]),(self.Thi,self.high[-1,:])):
            # Solver temperatures are deviations from setpoint (same model).
            constraints += [T[:,1:]==cp.multiply(a,T[:,:-1])+cp.multiply(1-a,ambient-th['setpoint_C']+cp.multiply(R,gain-cp.multiply(cop,self.cool)))]
        gl=self.low[:n,:]-self.high[n:2*n,:]+self.cool-self.trade
        gh=self.high[:n,:]-self.low[n:2*n,:]+self.cool-self.trade
        constraints += [gh<=p['grid_import_max_kw'],gl>=-p['grid_export_max_kw']]
        # Prices minus sell must be nonnegative for the convex representation.
        self.spread=cp.Parameter(H,nonneg=True)
        bill=dt*cp.sum(p['sell_AUD_per_kwh']*gh+cp.multiply(self.spread,cp.pos(gh))+p['quadratic_import_AUD_per_kw2h']*cp.square(cp.pos(gh)))
        def discomfort(T):
            return p['comfort_tracking_weight']*cp.square(T)+p['comfort_slack_AUD_per_degree_hour']*(cp.pos(th['comfort_low_C']-th['setpoint_C']-T)+cp.pos(T-(th['comfort_high_C']-th['setpoint_C'])))
        comfort=dt*cp.sum(cp.maximum(discomfort(self.Tlo[:,1:]),discomfort(self.Thi[:,1:])))
        smooth=p['smoothing_weight']*(cp.sum_squares(self.cool[:,0]-self.previous)+cp.sum_squares(self.cool[:,1:]-self.cool[:,:-1]))
        terminal=p['terminal_weight']*cp.sum(cp.maximum(cp.square(self.Tlo[:,-1]),cp.square(self.Thi[:,-1])))
        self.problem=DirectProblem(cp.Minimize((bill+comfort+smooth+terminal+self.fee)/(n*H)),constraints)
        assert self.problem.is_dcp()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter();cfg=self.cfg;p=cfg['controller'];n=self.n;H=cfg['controller']['horizon'];dt=cfg['dt_hours']
        lo=f['box_low'];hi=f['box_high'];assert np.all(lo<=hi)
        self.low.value=lo;self.high.value=hi;self.initial.value=temp-cfg['thermal']['setpoint_C'];self.previous.value=previous;self.spread.value=prices-p['sell_AUD_per_kwh']
        nominal=f['point_load']-f['point_pv'];dl=lo[:n]-hi[n:2*n];dh=hi[:n]-lo[n:2*n]
        self.fee.value=float(dt*p['balancing_surcharge_AUD_per_kwh']*np.maximum(abs(dl-nominal),abs(dh-nominal)).sum())
        self.problem.solve(**RISK_SOLVER)
        if self.problem.status!='optimal':raise RuntimeError(('box envelope status',self.problem.status))
        raw=self.trade.value.copy();trade=raw-raw.mean(axis=0,keepdims=True);cool=self.cool.value.copy()
        cert=envelope_values(cfg,f,cool,trade,temp,previous,prices)
        assert cool.min()>=-1e-7 and np.max(cool-np.array(cfg['thermal']['cooling_max_kw_electric'][:n])[:,None])<1e-7
        assert abs(trade).max()<p['trade_max_kw']+1e-6 and abs(trade.sum(axis=0)).max()<1e-8
        assert cert['gridhi'].max()<p['grid_import_max_kw']+1e-6 and cert['gridlo'].min()>-p['grid_export_max_kw']-1e-6
        np.testing.assert_allclose(cert['Tlo'],self.Tlo.value+cfg['thermal']['setpoint_C'],atol=1e-7,rtol=0);np.testing.assert_allclose(cert['Thi'],self.Thi.value+cfg['thermal']['setpoint_C'],atol=1e-7,rtol=0)
        optimum=float(self.problem.value)*n*H;gap=abs(cert['total']-optimum)/max(1,abs(optimum));assert gap<1e-6
        certificate={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cert.items()}
        certificate.update(cooling_kw=cool.tolist(),p2p_kw=trade.tolist(),low=lo.tolist(),high=hi.tolist())
        return cool[:,0],trade[:,0],dict(iterations=0,primal=0.,dual=0.,proximal=0.,central_objective=optimum,
            feasible_objective=cert['total'],relative_objective_gap=gap,raw_horizon_clearing_kw=float(abs(raw.sum(axis=0)).max()),
            max_trade_repair_kw=float(abs(trade-raw).max()),raw_first_trade_kw=raw[:,0].tolist(),
            center_reference_seconds=time.perf_counter()-start,algorithm_and_repair_seconds=0.,box_certificate=certificate,
            risk='box_cost_upper_bound',timing_scope='Central conservative envelope solve; not exact adjustable min-max.')
