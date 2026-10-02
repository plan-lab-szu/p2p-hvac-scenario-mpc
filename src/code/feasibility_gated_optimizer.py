"""PJ stopping also requires the unchanged numerical clearing to be feasible.

No new raw household data, objective, projection, or physical constraint is
introduced. Each home checks its repair and acknowledges readiness; iterate
further until all are ready. Centralized objective is an ex-post audit only.
"""
import time
import numpy as np
from benchmark_validation import Optimizer
from forecast_closed_loop import SOLVER,MARKET_TOL_KW,cleared_snapshot


def repair_candidate(homes,z):
    raw=[h.snapshot() for h in homes]
    assert max(float(np.max(c.violation())) for h in homes for c in h.constraints)<1e-6
    repaired=[cleared_snapshot(h,v,z[i]) for i,(h,v) in enumerate(zip(homes,raw))]
    assert abs(sum(v['g'] for v in repaired)).max()<1e-8
    for h,v in zip(homes,repaired):
        assert abs(h.load.value+v['cool']-h.pv.value-v['buy']+v['sell']-v['g']).max()<1e-8
        assert np.minimum(v['buy'],v['sell']).max()<1e-6
    return raw,repaired


class FeasibilityGatedOptimizer(Optimizer):
    def solve(self,f,temp,previous,prices):
        if self.method=='central': return super().solve(f,temp,previous,prices)
        start=time.perf_counter();p=self.cfg['controller'];homes=self.homes
        for i,h in enumerate(homes): h.set(f,temp[i],previous[i],prices)
        self.center.solve(**SOLVER)
        if self.center.status!='optimal': raise RuntimeError(('central status',self.center.status))
        optimum=float(self.center.value)
        assert abs(sum(h.cost_check(h.snapshot()) for h in homes)-optimum)<1e-5
        center_seconds=time.perf_counter()-start;algorithm_start=time.perf_counter()
        n,D=self.z.shape;rejections=[]
        for iterations in range(1,1001):
            gamma=[]
            for i,h in enumerate(homes):
                h.target.value=self.z[i]-self.u[i];h.old.value=self.old[i]
                h.problem.solve(warm_start=True,**SOLVER)
                if h.problem.status!='optimal': raise RuntimeError(('local status',iterations,i,h.problem.status))
                gamma.append(h.trade.value.copy())
            gamma=np.array(gamma);v=gamma+self.u;z=v-v.mean(axis=0,keepdims=True)
            primal=float(np.linalg.norm(gamma-z))
            dual=float(p['rho']*np.linalg.norm(z-self.z))
            proximal=float(p['beta']*np.linalg.norm(gamma-self.old))
            self.u+=gamma-z;self.old=gamma.copy();self.z=z
            ep=1e-4*np.sqrt(n*D)+1e-6*max(np.linalg.norm(gamma),np.linalg.norm(z))
            ed=1e-6*np.sqrt(n*D)+1e-6*np.linalg.norm(p['rho']*self.u)
            if primal<=ep and dual<=ed and proximal<=ed and abs(gamma.sum(axis=0)).max()<=MARKET_TOL_KW:
                try: raw,repaired=repair_candidate(homes,z)
                except AssertionError as error:
                    rejections.append(dict(iteration=iterations,reason=str(error) or 'Local post-clearing feasibility not met'))
                    continue
                break
        else: raise RuntimeError(('ADMM iteration limit with feasibility gate',iterations,primal,dual,proximal,rejections[-3:]))
        # Audit against the centralized reference AFTER stopping; never use its
        # objective to determine the distributed iteration count or actions.
        objective=sum(h.cost_check(v) for h,v in zip(homes,repaired))
        gap=abs(objective-optimum)/max(1,abs(optimum))
        assert objective>=optimum-1e-6*max(1,abs(optimum)) and gap<1e-4
        cool=np.array([v['cool'][0,0] for v in repaired]);trade=np.array([v['g'][0,0] for v in repaired])
        assert max(float(abs(v[k][:,0]-v[k][0,0]).max()) for v in repaired for k in ('cool','g'))<1e-7
        return cool,trade,dict(iterations=iterations,primal=primal,dual=dual,proximal=proximal,
            central_objective=optimum,feasible_objective=objective,relative_objective_gap=gap,
            raw_horizon_clearing_kw=float(abs(gamma.sum(axis=0)).max()),
            max_trade_repair_kw=max(float(abs(a['g']-b['g']).max()) for a,b in zip(repaired,raw)),
            raw_first_trade_kw=[float(v['g'][0,0]) for v in raw],
            center_reference_seconds=center_seconds,algorithm_and_repair_seconds=time.perf_counter()-algorithm_start,
            rejected_stop_candidates=rejections,feasibility_gate_rejections=len(rejections))
