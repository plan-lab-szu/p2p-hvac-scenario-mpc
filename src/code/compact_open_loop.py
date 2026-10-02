"""Unchanged open-loop mean QP; store only its nonduplicated common controls."""
import numpy as np
from diagnose_robust_heat import OpenLoopRiskOptimizer
from robust_mpc import scenario_costs


def expand_common(cfg,f,temp,cool,trade):
    n,H=cool.shape;S=f['load'].shape[1];th=cfg['thermal'];dt=cfg['dt_hours'];snapshots=[]
    for j in range(n):
        c=np.broadcast_to(cool[j],(S,H)).copy();g=np.broadcast_to(trade[j],(S,H)).copy()
        T=np.empty((S,H+1));T[:,0]=temp[j];R=th['R_C_per_kw'][j];a=np.exp(-dt/(R*th['C_kwh_per_C'][j]))
        for k in range(H):T[:,k+1]=a*T[:,k]+(1-a)*(f['ambient'][:,k]+R*(th['internal_gain_kw_thermal'][j]-th['COP'][j]*c[:,k]))
        net=f['load'][j]+c-f['pv'][j]-g
        snapshots.append(dict(cool=c,g=g,temp=T,buy=np.maximum(net,0),sell=np.maximum(-net,0),
            low=np.maximum(th['comfort_low_C']-T[:,1:],0),high=np.maximum(T[:,1:]-th['comfort_high_C'],0)))
    return snapshots


class CompactOpenLoop(OpenLoopRiskOptimizer):
    def __init__(self,cfg,n,S,method='central'):
        super().__init__(cfg,n,S,method,risk='mean',save_certificate=False)

    def solve(self,f,temp,previous,prices):
        _,_,stats=super().solve(f,temp,previous,prices)
        old=self.last['snapshots'];cool=np.array([v['cool'][0] for v in old]);trade=np.array([v['g'][0] for v in old])
        trade-=trade.mean(axis=0,keepdims=True)
        compact=expand_common(self.cfg,f,temp,cool,trade);p=self.cfg['controller']
        for a,b in zip(old,compact):
            for key in ('cool','g','temp'):np.testing.assert_allclose(a[key],b[key],atol=1e-7,rtol=0)
            assert b['buy'].max()<=p['grid_import_max_kw']+1e-6 and b['sell'].max()<=p['grid_export_max_kw']+1e-6
        costs=np.array([scenario_costs(h,v,b.value) for h,v,b in zip(self.homes,compact,self.fees)]).sum(axis=0)
        val=float(costs.mean());gap=abs(val-stats['central_objective'])/max(1.,abs(stats['central_objective']));assert gap<1e-6
        stats.update(feasible_objective=val,relative_objective_gap=gap,scenario_community_costs=costs.tolist(),
            expected_horizon_cost=val,worst_horizon_cost=float(costs.max()),
            common_control_certificate=dict(cooling_kw=cool.tolist(),p2p_kw=trade.tolist()),
            certificate_note='Common controls only; scenario states and net settlement independently reconstructible from frozen causal forecasts.')
        return cool[:,0],trade[:,0],stats
