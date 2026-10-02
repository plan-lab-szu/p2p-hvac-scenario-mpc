"""Unchanged PJ iterations; final numerical repair also respects public trade box.

The iteration consensus z and dual updates are not replaced. Only certificate
and dispatched trades use bounded cleared_z. All grid checks remain mandatory.
"""
import time
import numpy as np
from parallel_worker_optimizer import ParallelOptimizer,local_forecast
from forecast_closed_loop import SOLVER,MARKET_TOL_KW
from bounded_trade_repair import project_trade_box_zero_sum


class BoundedRepairParallel(ParallelOptimizer):
    def solve(self,f,temp,previous,prices):
        self.last_clearing_certificate=None
        p=self.cfg['controller'];homes=self.homes
        algorithm_start=time.perf_counter()
        self.dispatch('prepare',[(local_forecast(f,ix),temp[ix],previous[ix],prices) for ix in self.groups])
        preparation_seconds=time.perf_counter()-algorithm_start
        n,D=self.z.shape;rejections=[];overlap_rounds=0;overlap_seconds=0.
        for iterations in range(1,1001):
            replies=self.dispatch('iterate',[(self.z[ix]-self.u[ix],self.old[ix]) for ix in self.groups])
            gamma=np.concatenate([r['gamma'] for r in replies],axis=0)
            overlap=max(0,min(r['end_ns'] for r in replies)-max(r['begin_ns'] for r in replies))/1e9
            overlap_rounds+=int(overlap>0);overlap_seconds+=overlap
            v=gamma+self.u;z=v-v.mean(axis=0,keepdims=True)
            primal=float(np.linalg.norm(gamma-z));dual=float(p['rho']*np.linalg.norm(z-self.z))
            proximal=float(p['beta']*np.linalg.norm(gamma-self.old))
            self.u+=gamma-z;self.old=gamma.copy();self.z=z
            ep=1e-4*np.sqrt(n*D)+1e-6*max(np.linalg.norm(gamma),np.linalg.norm(z))
            ed=1e-6*np.sqrt(n*D)+1e-6*np.linalg.norm(p['rho']*self.u)
            if primal<=ep and dual<=ed and proximal<=ed and abs(gamma.sum(axis=0)).max()<=MARKET_TOL_KW:
                cleared=project_trade_box_zero_sum(z,p['trade_max_kw'])
                certificates=self.dispatch('certificate',[cleared[ix] for ix in self.groups])
                if not all(r['ready'] for r in certificates):
                    rejections.append(dict(iteration=iterations,reason=[r.get('reason') for r in certificates if not r['ready']]))
                    continue
                break
        else:raise RuntimeError(('Parallel PJ iteration limit',iterations,primal,dual,proximal,rejections[-3:]))
        algorithm_seconds=time.perf_counter()-algorithm_start
        records=[row for r in certificates for row in r['records']]
        objective=sum(r['objective'] for r in records)
        start=time.perf_counter()
        for i,h in enumerate(homes):h.set(f,temp[i],previous[i],prices)
        self.center.solve(**SOLVER)
        if self.center.status!='optimal':raise RuntimeError(('central status',self.center.status))
        optimum=float(self.center.value)
        assert abs(sum(h.cost_check(h.snapshot()) for h in homes)-optimum)<1e-5
        center_seconds=time.perf_counter()-start
        gap=abs(objective-optimum)/max(1,abs(optimum))
        # The centralized reference is used only after the distributed stop.
        assert objective>=optimum-1e-6*max(1,abs(optimum)) and gap<1e-4
        assert abs(cleared.sum(axis=0)).max()<1e-8
        self.last_clearing_certificate=dict(raw_gamma=gamma.copy(),consensus_z=z.copy(),cleared_z=cleared.copy())
        return np.array([r['cool'] for r in records]),np.array([r['trade'] for r in records]),dict(
            iterations=iterations,primal=primal,dual=dual,proximal=proximal,
            repair_method='public_trade_box_zero_sum_projection',
            bounded_projection_adjustment_kw=float(abs(cleared-z).max()),
            bounded_clearing_kw=float(abs(cleared.sum(axis=0)).max()),
            bounded_trade_excess_kw=float(max(0,abs(cleared).max()-p['trade_max_kw'])),
            central_objective=optimum,feasible_objective=objective,relative_objective_gap=gap,
            raw_horizon_clearing_kw=float(abs(gamma.sum(axis=0)).max()),
            max_trade_repair_kw=max(r['repair'] for r in records),raw_first_trade_kw=[r['raw_first_trade'] for r in records],
            center_reference_seconds=center_seconds,algorithm_and_repair_seconds=algorithm_seconds,
            simulator_input_injection_seconds=preparation_seconds,worker_startup_seconds=self.startup_seconds,
            worker_pids=self.worker_pids,parallel_workers=len(self.groups),
            all_worker_overlap_rounds=overlap_rounds,all_worker_overlap_seconds=overlap_seconds,
            rejected_stop_candidates=rejections,feasibility_gate_rejections=len(rejections),
            timing_scope='Includes input IPC and validation certificates; excludes worker startup and centralized reference. Not deployment network latency.')


