"""Persistent process workers for synchronous PJ updates; validation harness.

Step-input injection, local dispatch collection and scalar objective certificates
belong to the simulator/auditor, not to a claimed deployed market wire protocol.
Only prior-iteration targets/old trades enter each parallel update. No central
solution or centralized objective is sent to the workers.
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import cvxpy as cp
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
            elif command=='penalty':
                rho,beta=payload
                assert rho>0 and beta>0
                cfg['controller'].update(rho=rho,beta=beta)
                for h in homes:
                    h.problem=cp.Problem(cp.Minimize(h.cost+rho/2*cp.sum_squares(h.trade-h.target)+beta/2*cp.sum_squares(h.trade-h.old)),h.constraints)
                response=dict(rho=rho,beta=beta)
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
