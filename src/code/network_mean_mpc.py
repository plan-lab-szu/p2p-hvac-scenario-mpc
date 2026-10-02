"""Nominal LinDistFlow extension; AC realized security is checked separately."""
import numpy as np
import cvxpy as cp
from reduced_mean_qp import ReducedMeanQP
from network33 import matrices,P_KW,Q_KVAR,BASE_MVA
from uncached_central_optimizer import DirectProblem


class NetworkMean(ReducedMeanQP):
    def __init__(self,cfg,n,S,method='central',background=1.):
        super().__init__(cfg,n,S,method);H=self.H;self.background=background;m=matrices(n)
        self.nominal_load=cp.Parameter((n,H));self.nominal_pv=cp.Parameter((n,H))
        P=(background*P_KW[:,None]+m['M']@(self.nominal_load+self.cool-self.nominal_pv))/1000
        Q=(background*Q_KVAR[:,None]+m['pf_tan']*m['M']@(self.nominal_load+self.cool))/1000
        self.lineP=m['D']@P;self.lineQ=m['D']@Q
        self.v2=1-2*m['A']@(cp.multiply(m['z'].real[:,None],self.lineP/BASE_MVA)+cp.multiply(m['z'].imag[:,None],self.lineQ/BASE_MVA))
        self.lower=self.v2[1:]>=.9**2;self.upper=self.v2[1:]<=1.1**2
        self.line_limits=cp.norm(cp.vstack([cp.reshape(self.lineP,(1,32*H),order='C'),cp.reshape(self.lineQ,(1,32*H),order='C')]),axis=0)<=np.repeat(m['ratings_mva'],H)
        self.transformer=cp.norm(cp.vstack([self.lineP[0],self.lineQ[0]]),axis=0)<=5.
        self.network_constraints=[self.lower,self.upper,self.line_limits,self.transformer]
        self.market=self.problem.constraints[3]
        self.problem=DirectProblem(self.problem.objective,self.problem.constraints+self.network_constraints)

    def solve(self,f,temp,previous,prices):
        self.nominal_load.value=f['point_load'];self.nominal_pv.value=f['point_pv']
        cool,trade,s=super().solve(f,temp,previous,prices)
        violation=max(float(np.asarray(c.violation()).max()) for c in self.network_constraints);assert violation<1e-7
        scale=self.n*self.H
        s['network_certificate']=dict(background=self.background,voltage_squared=self.v2.value.tolist(),branch_P_mw=self.lineP.value.tolist(),
            branch_Q_mvar=self.lineQ.value.tolist(),max_constraint_violation=violation,
            voltage_lower_dual_AUD_per_pu2=(self.lower.dual_value*scale).tolist(),voltage_upper_dual_AUD_per_pu2=(self.upper.dual_value*scale).tolist(),
            line_dual_AUD_per_MVA=(self.line_limits.dual_value*scale).tolist(),transformer_dual_AUD_per_MVA=(self.transformer.dual_value*scale).tolist(),
            market_shadow_AUD_per_kwh=(self.market.dual_value*scale/self.cfg['dt_hours']).tolist(),
            scope='Nominal linear constraints only; not scenario-wise or AC-realization security; market shadow is not an implemented tariff.')
        return cool,trade,s
