"""Exactly the same nominal network extension, with four bus cooling sums."""
import numpy as np
import cvxpy as cp
from reduced_mean_qp import ReducedMeanQP
from network33 import matrices,P_KW,Q_KVAR,BASE_MVA
from uncached_central_optimizer import DirectProblem
from network_mean_mpc import NetworkMean


class AggregatedNetworkMean(NetworkMean):
    def __init__(self,cfg,n,S,method='central',background=1.):
        ReducedMeanQP.__init__(self,cfg,n,S,method);H=self.H;self.background=background;m=matrices(n)
        nodes=np.where(m['M'].sum(axis=1)>0)[0];K=len(nodes);select=np.eye(33)[:,nodes]
        self.nominal_load=cp.Parameter((n,H));self.nominal_pv=cp.Parameter((n,H))
        self.busP=cp.Parameter((33,H));self.busQ=cp.Parameter((33,H));group=cp.Variable((K,H))
        aggregation=group==m['M'][nodes]@self.cool
        P=self.busP+select@group/1000;Q=self.busQ+m['pf_tan']*select@group/1000
        self.lineP=m['D']@P;self.lineQ=m['D']@Q
        self.v2=1-2*m['A']@(cp.multiply(m['z'].real[:,None],self.lineP/BASE_MVA)+cp.multiply(m['z'].imag[:,None],self.lineQ/BASE_MVA))
        self.lower=self.v2[1:]>=.9**2;self.upper=self.v2[1:]<=1.1**2
        self.line_limits=cp.norm(cp.vstack([cp.reshape(self.lineP,(1,32*H),order='C'),cp.reshape(self.lineQ,(1,32*H),order='C')]),axis=0)<=np.repeat(m['ratings_mva'],H)
        self.transformer=cp.norm(cp.vstack([self.lineP[0],self.lineQ[0]]),axis=0)<=5.
        self.network_constraints=[self.lower,self.upper,self.line_limits,self.transformer,aggregation]
        self.market=self.problem.constraints[3]
        self.problem=DirectProblem(self.problem.objective,self.problem.constraints+self.network_constraints)

    def solve(self,f,temp,previous,prices):
        m=matrices(self.n)
        self.busP.value=(self.background*P_KW[:,None]+m['M']@(f['point_load']-f['point_pv']))/1000
        self.busQ.value=(self.background*Q_KVAR[:,None]+m['pf_tan']*m['M']@f['point_load'])/1000
        return super().solve(f,temp,previous,prices)
