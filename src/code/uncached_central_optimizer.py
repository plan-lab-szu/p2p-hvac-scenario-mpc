"""Avoid dense parameter-program construction in large centralized references."""
import cvxpy as cp
from feasibility_gated_optimizer import FeasibilityGatedOptimizer


class DirectProblem(cp.Problem):
    def solve(self,*args,**kwargs):
        kwargs['ignore_dpp']=True
        return super().solve(*args,**kwargs)


class MemorySafeOptimizer(FeasibilityGatedOptimizer):
    def __init__(self,cfg,n,S,method):
        super().__init__(cfg,n,S,method)
        # Exactly the existing objective and constraints; only canonicalization
        # caching is disabled. Local QPs retain their original parameter caching.
        self.center=DirectProblem(self.center.objective,self.center.constraints)
