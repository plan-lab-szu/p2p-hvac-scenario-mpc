"""Exactly the same interval-cost envelope, expressed as a QP (no SOC).

Both endpoint thermal costs share w*m^2. Their remainder is max of six
affine expressions, since low and high comfort hinges cannot both be positive.
"""
import numpy as np
import cvxpy as cp
from box_envelope_mpc import BoxEnvelopeOptimizer
from uncached_central_optimizer import DirectProblem


class BoxEnvelopeQP(BoxEnvelopeOptimizer):
    def __init__(self,cfg,n,S=30,method='central'):
        if method!='central':raise ValueError(method)
        self.cfg=cfg;self.n=n;th=cfg['thermal'];p=cfg['controller'];dt=cfg['dt_hours'];H=p['horizon']
        self.low=cp.Parameter((2*n+1,H));self.high=cp.Parameter((2*n+1,H));self.initial=cp.Parameter(n)
        self.previous=cp.Parameter(n);self.spread=cp.Parameter(H,nonneg=True);self.fee=cp.Parameter(nonneg=True)
        self.width=cp.Parameter((n,H+1),nonneg=True);self.mid=cp.Variable((n,H+1))
        self.Tlo=self.mid-self.width;self.Thi=self.mid+self.width
        self.cool=cp.Variable((n,H),nonneg=True);self.trade=cp.Variable((n,H));buy=cp.Variable((n,H),nonneg=True)
        R=np.array(th['R_C_per_kw'][:n])[:,None];C=np.array(th['C_kwh_per_C'][:n])[:,None];a=np.exp(-dt/(R*C))
        cop=np.array(th['COP'][:n])[:,None];gain=np.array(th['internal_gain_kw_thermal'][:n])[:,None]
        ambient=(self.low[-1,:]+self.high[-1,:])/2-th['setpoint_C']
        constraints=[self.mid[:,0]==self.initial,
            self.mid[:,1:]==cp.multiply(a,self.mid[:,:-1])+cp.multiply(1-a,ambient+cp.multiply(R,gain-cp.multiply(cop,self.cool))),
            self.cool<=np.array(th['cooling_max_kw_electric'][:n])[:,None],self.trade<=p['trade_max_kw'],self.trade>=-p['trade_max_kw'],cp.sum(self.trade,axis=0)==0]
        gh=self.high[:n,:]-self.low[n:2*n,:]+self.cool-self.trade
        gl=self.low[:n,:]-self.high[n:2*n,:]+self.cool-self.trade
        constraints += [gh<=p['grid_import_max_kw'],gl>=-p['grid_export_max_kw'],buy>=gh]
        bill=dt*(p['sell_AUD_per_kwh']*cp.sum(gh)+cp.sum(cp.multiply(self.spread,buy))+p['quadratic_import_AUD_per_kw2h']*cp.sum_squares(buy))
        w=p['comfort_tracking_weight'];kappa=p['comfort_slack_AUD_per_degree_hour'];m=self.mid[:,1:];d=self.width[:,1:]
        epigraph=cp.Variable((n,H))
        for sign in (-1,1):
            shift=sign*d;base=2*w*cp.multiply(shift,m)+w*cp.square(shift)
            for hinge in (0.,th['comfort_low_C']-th['setpoint_C']-m-shift,m+shift-(th['comfort_high_C']-th['setpoint_C'])):
                constraints += [epigraph>=base+kappa*hinge]
        comfort=dt*(w*cp.sum_squares(m)+cp.sum(epigraph))
        smooth=p['smoothing_weight']*(cp.sum_squares(self.cool[:,0]-self.previous)+cp.sum_squares(self.cool[:,1:]-self.cool[:,:-1]))
        terminal=p['terminal_weight']*(cp.sum_squares(self.mid[:,-1])+cp.sum_squares(self.width[:,-1])+2*cp.sum(cp.multiply(self.width[:,-1],cp.abs(self.mid[:,-1]))))
        self.problem=DirectProblem(cp.Minimize((bill+comfort+smooth+terminal+self.fee)/(n*H)),constraints)
        assert self.problem.is_qp() and self.problem.is_dcp()

    def solve(self,f,temp,previous,prices):
        n=self.n;H=self.cfg['controller']['horizon'];th=self.cfg['thermal'];dt=self.cfg['dt_hours']
        a=np.exp(-dt/(np.array(th['R_C_per_kw'][:n])*np.array(th['C_kwh_per_C'][:n])))
        width=np.zeros((n,H+1))
        for k in range(H):width[:,k+1]=a*width[:,k]+(1-a)*(f['box_high'][-1,k]-f['box_low'][-1,k])/2
        self.width.value=width
        cool,trade,stats=super().solve(f,temp,previous,prices)
        stats['formulation']='Exact QP reformulation of the same conservative interval-cost upper bound'
        return cool,trade,stats
