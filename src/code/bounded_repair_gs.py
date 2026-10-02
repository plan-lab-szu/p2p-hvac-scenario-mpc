"""Cyclic proximal multi-block ADMM, not serial execution of Jacobi updates."""
import time
import numpy as np
from forecast_closed_loop import Home,SOLVER,cleared_snapshot,MARKET_TOL_KW
from uncached_central_optimizer import DirectProblem
import cvxpy as cp
from bounded_trade_repair import project_trade_box_zero_sum


class BoundedRepairGS:
    def __init__(self,cfg,n,S):
        self.cfg=cfg;self.homes=[Home(i,cfg,S) for i in range(n)];self.n=n;D=self.homes[0].D
        self.gamma=np.zeros((n,D));self.u=np.zeros(D);self.history=[]

    def solve(self,f,temp,previous,prices):
        self.last_clearing_certificate=None
        begin=time.perf_counter();p=self.cfg['controller'];n,D=self.gamma.shape
        for j,h in enumerate(self.homes):h.set(f,temp[j],previous[j],prices)
        rejections=[]
        for iteration in range(1,1001):
            old=self.gamma.copy();total=self.gamma.sum(axis=0)
            for j,h in enumerate(self.homes):
                others=total-self.gamma[j]
                h.target.value=-others-self.u;h.old.value=old[j]
                h.problem.solve(warm_start=True,**SOLVER)
                if h.problem.status!='optimal':raise RuntimeError(('sequential local status',iteration,j,h.problem.status))
                new=h.trade.value.copy();total+=new-self.gamma[j];self.gamma[j]=new
            self.u+=total;delta=self.gamma-old
            later=np.cumsum(delta[::-1],axis=0)[::-1]-delta
            stationarity=p['rho']*later-p['beta']*delta
            dual=float(np.linalg.norm(stationarity));primal=float(np.linalg.norm(total));rawmax=float(abs(total).max())
            tol=1e-6*np.sqrt(n*D)+1e-6*p['rho']*np.sqrt(n)*np.linalg.norm(self.u)
            self.history.append(dict(iteration=iteration,primal=primal,max_clearing=rawmax,stationarity=dual,tolerance=float(tol)))
            if rawmax<=MARKET_TOL_KW and dual<=tol:
                plain=self.gamma-self.gamma.mean(axis=0,keepdims=True)
                z=project_trade_box_zero_sum(plain,p['trade_max_kw'])
                try:repaired=[cleared_snapshot(h,h.snapshot(),z[j]) for j,h in enumerate(self.homes)]
                except AssertionError as error:rejections.append(dict(iteration=iteration,error=str(error)));continue
                break
        else:raise RuntimeError(('sequential iteration limit',self.history[-1]))
        seconds=time.perf_counter()-begin;value=sum(h.cost_check(v) for h,v in zip(self.homes,repaired))
        self.last_clearing_certificate=dict(raw_gamma=self.gamma.copy(),consensus_z=plain.copy(),cleared_z=z.copy())
        # Central reference is created and solved only AFTER the stop.
        central=DirectProblem(cp.Minimize(sum(h.cost for h in self.homes)),sum([h.constraints for h in self.homes],[])+[sum(h.trade for h in self.homes)==0])
        start=time.perf_counter();central.solve(**SOLVER)
        if central.status!='optimal':raise RuntimeError(('central audit',central.status))
        optimum=float(central.value);gap=abs(value-optimum)/max(1,abs(optimum));assert gap<1e-4
        assert abs(sum(v['g'] for v in repaired)).max()<1e-8
        return dict(iterations=iteration,primal=primal,stationarity=dual,relative_gap=gap,feasible_objective=value,central_objective=optimum,
            algorithm_seconds=seconds,central_audit_seconds=time.perf_counter()-start,rejections=rejections,history=self.history,
            snapshots=[{k:a.tolist() for k,a in v.items()} for v in repaired])
