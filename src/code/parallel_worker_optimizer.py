"""Persistent process workers for synchronous PJ updates; validation harness.

Step-input injection, local dispatch collection and scalar objective certificates
belong to the simulator/auditor, not to a claimed deployed market wire protocol.
Only prior-iteration targets/old trades enter each parallel update. No central
solution or centralized objective is sent to the workers.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import copy
import multiprocessing as mp
import time
import traceback
import numpy as np

from forecast_closed_loop import Home,SOLVER,MARKET_TOL_KW,cleared_snapshot
from uncached_central_optimizer import MemorySafeOptimizer


def local_config(cfg,indices):
    result=copy.deepcopy(cfg);total=len(cfg['household_ids'])
    result['household_ids']=[cfg['household_ids'][i] for i in indices]
    for section in ('pv','thermal'):
        for key,value in cfg[section].items():
            if isinstance(value,list) and len(value)==total:
                result[section][key]=[value[i] for i in indices]
    return result


def local_forecast(f,indices):
    result=dict(f)
    for key in ('load','pv','point_load','point_pv'):result[key]=f[key][indices]
    return result


def worker_main(pipe,cfg,S):
    homes=[Home(i,cfg,S) for i in range(len(cfg['household_ids']))]
    pipe.send(dict(ok=True,pid=os.getpid()))
    while True:
        try:command,payload=pipe.recv()
        except EOFError:break
        if command=='close':break
        try:
            begin=time.perf_counter_ns()
            if command=='prepare':
                f,temp,previous,prices=payload
                for i,h in enumerate(homes):h.set(f,temp[i],previous[i],prices)
                response={}
            elif command=='iterate':
                targets,old=payload;gamma=[]
                for i,h in enumerate(homes):
                    h.target.value=targets[i];h.old.value=old[i]
                    h.problem.solve(warm_start=True,**SOLVER)
                    if h.problem.status!='optimal':raise RuntimeError(('local status',i,h.problem.status))
                    gamma.append(h.trade.value.copy())
                response=dict(gamma=np.array(gamma))
            elif command=='certificate':
                records=[]
                for i,h in enumerate(homes):
                    raw=h.snapshot()
                    violation=max(float(np.max(c.violation())) for c in h.constraints)
                    assert violation<1e-6
                    p=cfg['controller']
                    raw_cost=h.cost_check(raw)
                    augmented=raw_cost+p['rho']/2*np.sum((h.trade.value-h.target.value)**2)+p['beta']/2*np.sum((h.trade.value-h.old.value)**2)
                    assert abs(augmented-h.problem.value)<1e-5
                    v=cleared_snapshot(h,raw,payload[i])
                    assert abs(h.load.value+v['cool']-h.pv.value-v['buy']+v['sell']-v['g']).max()<1e-8
                    assert np.minimum(v['buy'],v['sell']).max()<1e-6
                    assert max(float(abs(v[k][:,0]-v[k][0,0]).max()) for k in ('cool','g'))<1e-7
                    records.append(dict(cool=float(v['cool'][0,0]),trade=float(v['g'][0,0]),
                        raw_first_trade=float(raw['g'][0,0]),objective=h.cost_check(v),
                        repair=float(abs(v['g']-raw['g']).max())))
                response=dict(ready=True,records=records)
            else:raise ValueError(command)
            response.update(ok=True,pid=os.getpid(),begin_ns=begin,end_ns=time.perf_counter_ns())
            pipe.send(response)
        except AssertionError as error:
            if command=='certificate':pipe.send(dict(ok=True,ready=False,reason=str(error) or 'Local certificate failed',pid=os.getpid()))
            else:pipe.send(dict(ok=False,error=traceback.format_exc(),pid=os.getpid()))
        except Exception:pipe.send(dict(ok=False,error=traceback.format_exc(),pid=os.getpid()))
    pipe.close()


class ParallelOptimizer(MemorySafeOptimizer):
    def __init__(self,cfg,n,S,method,workers=4):
        if method!='pj':raise ValueError('ParallelOptimizer only implements synchronous PJ')
        super().__init__(cfg,n,S,method)
        self.groups=[x.tolist() for x in np.array_split(np.arange(n),min(n,workers))]
        self.pipes=[];self.processes=[];self.closed=False
        begin=time.perf_counter();context=mp.get_context('spawn')
        try:
            for indices in self.groups:
                parent,child=context.Pipe()
                process=context.Process(target=worker_main,args=(child,local_config(cfg,indices),S),daemon=True)
                process.start();child.close();self.pipes.append(parent);self.processes.append(process)
            replies=self.collect()
            self.worker_pids=[r['pid'] for r in replies]
            assert len(set(self.worker_pids))==len(self.groups)
        except BaseException:
            self.close();raise
        self.startup_seconds=time.perf_counter()-begin

    def collect(self,timeout=180):
        deadline=time.monotonic()+timeout;replies=[]
        for pipe in self.pipes:
            if not pipe.poll(max(0,deadline-time.monotonic())):raise TimeoutError('Parallel worker response deadline')
            reply=pipe.recv()
            if not reply['ok']:raise RuntimeError(reply['error'])
            replies.append(reply)
        return replies

    def dispatch(self,command,payloads):
        assert len(payloads)==len(self.pipes)
        # Send to every worker BEFORE receiving any result (Jacobi update).
        for pipe,payload in zip(self.pipes,payloads):pipe.send((command,payload))
        return self.collect()

    def close(self):
        if self.closed:return
        self.closed=True
        for pipe in self.pipes:
            try:pipe.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
        for pipe in self.pipes:pipe.close()

    def solve(self,f,temp,previous,prices):
        start=time.perf_counter();p=self.cfg['controller'];homes=self.homes
        for i,h in enumerate(homes):h.set(f,temp[i],previous[i],prices)
        self.center.solve(**SOLVER)
        if self.center.status!='optimal':raise RuntimeError(('central status',self.center.status))
        optimum=float(self.center.value)
        assert abs(sum(h.cost_check(h.snapshot()) for h in homes)-optimum)<1e-5
        center_seconds=time.perf_counter()-start
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
                certificates=self.dispatch('certificate',[z[ix] for ix in self.groups])
                if not all(r['ready'] for r in certificates):
                    rejections.append(dict(iteration=iterations,reason=[r.get('reason') for r in certificates if not r['ready']]))
                    continue
                break
        else:raise RuntimeError(('Parallel PJ iteration limit',iterations,primal,dual,proximal,rejections[-3:]))
        algorithm_seconds=time.perf_counter()-algorithm_start
        records=[row for r in certificates for row in r['records']]
        objective=sum(r['objective'] for r in records)
        gap=abs(objective-optimum)/max(1,abs(optimum))
        # The centralized reference is used only after the distributed stop.
        assert objective>=optimum-1e-6*max(1,abs(optimum)) and gap<1e-4
        assert abs(z.sum(axis=0)).max()<1e-8
        return np.array([r['cool'] for r in records]),np.array([r['trade'] for r in records]),dict(
            iterations=iterations,primal=primal,dual=dual,proximal=proximal,
            central_objective=optimum,feasible_objective=objective,relative_objective_gap=gap,
            raw_horizon_clearing_kw=float(abs(gamma.sum(axis=0)).max()),
            max_trade_repair_kw=max(r['repair'] for r in records),raw_first_trade_kw=[r['raw_first_trade'] for r in records],
            center_reference_seconds=center_seconds,algorithm_and_repair_seconds=algorithm_seconds,
            simulator_input_injection_seconds=preparation_seconds,worker_startup_seconds=self.startup_seconds,
            worker_pids=self.worker_pids,parallel_workers=len(self.groups),
            all_worker_overlap_rounds=overlap_rounds,all_worker_overlap_seconds=overlap_seconds,
            rejected_stop_candidates=rejections,feasibility_gate_rejections=len(rejections),
            timing_scope='Includes input IPC and validation certificates; excludes worker startup and centralized reference. Not deployment network latency.')
