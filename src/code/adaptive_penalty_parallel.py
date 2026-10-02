"""Finite residual balancing; inherit original iteration, stop and certificates."""
import copy
import multiprocessing as mp
import time
import numpy as np
from adaptive_penalty_workers import worker_main,local_config
from uncached_central_optimizer import MemorySafeOptimizer
from bounded_failure_diagnostic import ObservedBoundedParallel
from parallel_worker_optimizer import ParallelOptimizer


def next_penalty(rho,iteration,metrics):
    # Called before the NEXT iteration, never after a stopping certificate.
    if iteration%25 or iteration>750:return rho
    primal=max(metrics['primal']/metrics['ep'],metrics['raw_max_kw']/.001)
    dual=max(metrics['dual'],metrics['proximal'])/metrics['ed']
    if primal>5*max(1.,dual):return min(.16,rho*2)
    if dual>5*max(1.,primal):return max(.0025,rho/2)
    return rho


class AdaptivePenaltyParallel(ObservedBoundedParallel):
    def __init__(self,cfg,n,S,method,workers=4):
        if method!='pj':raise ValueError('Only synchronous PJ is supported')
        cfg=copy.deepcopy(cfg)
        MemorySafeOptimizer.__init__(self,cfg,n,S,method)
        self.groups=[x.tolist() for x in np.array_split(np.arange(n),min(n,workers))]
        self.pipes=[];self.processes=[];self.closed=False
        self.penalty_events=[];self.origin_initial_penalty=None
        begin=time.perf_counter();context=mp.get_context('spawn')
        try:
            for indices in self.groups:
                parent,child=context.Pipe()
                process=context.Process(target=worker_main,args=(child,local_config(cfg,indices),S),daemon=True)
                process.start();child.close();self.pipes.append(parent);self.processes.append(process)
            replies=self.collect();self.worker_pids=[r['pid'] for r in replies]
            assert len(set(self.worker_pids))==len(self.groups)
        except BaseException:
            self.close();raise
        self.startup_seconds=time.perf_counter()-begin

    def dispatch(self,command,payloads):
        p=self.cfg['controller']
        if command=='prepare':
            self.penalty_events=[];self.origin_initial_penalty=p['rho']
        if command=='iterate' and self.trace:
            old_rho=p['rho'];new_rho=next_penalty(old_rho,len(self.trace),self.trace[-1])
            if new_rho!=old_rho:
                unscaled=old_rho*self.u.copy()
                # Update all workers before sending any new iteration targets.
                replies=ParallelOptimizer.dispatch(self,'penalty',[(new_rho,.1*new_rho) for _ in self.groups])
                assert all(r['rho']==new_rho and r['beta']==.1*new_rho for r in replies)
                self.u*=old_rho/new_rho
                p.update(rho=new_rho,beta=.1*new_rho)
                error=float(abs(new_rho*self.u-unscaled).max())
                assert error<1e-12
                self.penalty_events.append(dict(after_iteration=len(self.trace),old_rho=old_rho,
                    new_rho=new_rho,unscaled_dual_preservation_error=error))
                payloads=[(self.z[ix]-self.u[ix],self.old[ix]) for ix in self.groups]
        replies=super().dispatch(command,payloads)
        if command=='iterate':
            self.trace[-1].update(rho=p['rho'],beta=p['beta'])
        return replies
