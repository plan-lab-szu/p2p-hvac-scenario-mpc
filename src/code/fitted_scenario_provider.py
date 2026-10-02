"""Fixed fitting-pool forecast with explicit seed; legacy seed-202 unchanged."""
import numpy as np
from forecast_closed_loop import forecast
from calibrated_box import KEY


def provider(data,t,key,n,H,S,ac_cap,mode='scenario',seed=202):
    if key!=KEY or H!=8:raise ValueError('Frozen case/horizon')
    point=forecast(data,t,key,n,H,1,ac_cap,mode='deterministic')
    if mode=='deterministic':return point
    days=np.random.default_rng(seed).choice(np.arange(1,39),size=S,replace=False)
    loads=[];pvs=[];amb=[]
    for day in days:
        ix=day*48+t%48+np.arange(H)
        assert ix.max()<40*48 and ix.max()<t
        loads.append(np.maximum(0,point['point_load']+(data['base_kw'][ix,:n]-data['base_kw'][ix-48,:n]).T))
        pvs.append(np.clip(point['point_pv']+(data[key+'_pv'][ix,:n]-data[key+'_pv'][ix-48,:n]).T,0,ac_cap[:,None]))
        amb.append(point['point_ambient']+data[key+'_ambient'][ix]-data[key+'_ambient'][ix-48])
    return dict(point,load=np.stack(loads,axis=1),pv=np.stack(pvs,axis=1),ambient=np.array(amb),
        max_source_index=max(point['max_source_index'],int(days.max()*48+t%48+H-1)))
