"""Training-only daily joint calibration and restricted residual provider."""
import json
import math
import numpy as np
from benchmark_validation import ROOT,digest
from forecast_closed_loop import forecast

KEY='sydney_utc10_end'


def feature_values(data,n):
    return np.concatenate([data['base_kw'][:,:n],data[KEY+'_pv'][:,:n],data[KEY+'_ambient'][:,None]],axis=1)


def fit_box(data,n):
    # Explicit slice prevents calibration from inspecting any validation values.
    fit={k:v[:60*48] for k,v in data.items()};x=feature_values(fit,n)
    assert len(x)==60*48
    errors=x[48:40*48]-x[:39*48]
    scale=np.maximum(np.quantile(abs(errors),.9,axis=0),np.r_[np.full(2*n,.1),1.])
    scores=[]
    for day in range(40,59):
        origins=day*48+np.arange(48);ix=origins[:,None]+np.arange(8)
        assert ix.max()<60*48
        scores.append(float((abs(x[ix]-x[ix-48])/scale).max()))
    rank=math.ceil((len(scores)+1)*.90)
    q=float(np.sort(scores)[rank-1])
    return dict(n=n,scale=scale.tolist(),radius=q,rank=rank,daily_scores=scores,
        fit_stop_exclusive=40*48,calibration_max_index=59*48+6,
        target=.90,probability_guarantee=False)


def bounds(point,calibration,cap,multiplier=1.):
    n=len(cap);scale=np.array(calibration['scale']);r=calibration['radius']*multiplier
    center=np.vstack([point['point_load'],point['point_pv'],point['point_ambient'][None,:]])
    low=center-r*scale[:,None];high=center+r*scale[:,None]
    low[:n]=np.maximum(low[:n],0)
    low[n:2*n]=np.clip(low[n:2*n],0,cap[:,None]);high[n:2*n]=np.clip(high[n:2*n],0,cap[:,None])
    return low,high


def provider(data,t,key,n,H,S,ac_cap,mode='scenario',calibration=None,multiplier=1.):
    if key!=KEY or H!=8:raise ValueError('Frozen calibration case/horizon')
    point=forecast(data,t,key,n,H,1,ac_cap,mode='deterministic')
    if mode=='deterministic':f=point
    else:
        days=np.random.default_rng(202).choice(np.arange(1,39),size=S,replace=False)
        load=[];pv=[];amb=[]
        for day in days:
            ix=day*48+t%48+np.arange(H);assert ix.max()<40*48 and ix.max()<t
            load.append(np.maximum(0,point['point_load']+(data['base_kw'][ix,:n]-data['base_kw'][ix-48,:n]).T))
            pv.append(np.clip(point['point_pv']+(data[key+'_pv'][ix,:n]-data[key+'_pv'][ix-48,:n]).T,0,ac_cap[:,None]))
            amb.append(point['point_ambient']+data[key+'_ambient'][ix]-data[key+'_ambient'][ix-48])
        f=dict(point,load=np.stack(load,axis=1),pv=np.stack(pv,axis=1),ambient=np.array(amb))
        f['max_source_index']=max(point['max_source_index'],int(days.max()*48+t%48+H-1))
    if calibration is not None:
        f['box_low'],f['box_high']=bounds(point,calibration,ac_cap,multiplier)
    return f


def main():
    data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'));cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text())
    folder=ROOT/'results/box_calibration_20260914_v1';folder.mkdir(parents=True,exist_ok=True)
    out=folder/'calibration.json'
    if out.exists():raise FileExistsError(out)
    calibration={str(n):fit_box(data,n) for n in (3,10,50,100)}
    frozen=dict(populations=calibration,parameter_sha256=digest(ROOT/'formal_parameters_v1.json'),
        input_sha256=digest(ROOT/'data/formal_validation_inputs.npz'),source_sha256=digest(__file__),
        protocol_sha256=digest(ROOT/'CALIBRATED_BOX_PROTOCOL_2026-09-14.md'),formal_test_run=False)
    out.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    # All calibration values persisted BEFORE inspecting validation coverage.
    results=[]
    for n in (3,10,50,100):
        x=feature_values(data,n);cap=np.array(cfg['pv']['ac_kw'][:n]);cal=calibration[str(n)]
        for factor in (.8,1.,1.2):
            first=[];full=[]
            for t in range(60*48,len(x)):
                point=forecast(data,t,KEY,n,8,1,cap,mode='deterministic');lo,hi=bounds(point,cal,cap,factor)
                first.append(bool(np.all(x[t]>=lo[:,0]-1e-8) and np.all(x[t]<=hi[:,0]+1e-8)))
                if t+8<=len(x):full.append(bool(np.all(x[t:t+8].T>=lo-1e-8) and np.all(x[t:t+8].T<=hi+1e-8)))
            days=[all(full[i:i+48]) for i in range(0,len(full)-47,48)]
            results.append(dict(n=n,multiplier=factor,radius=cal['radius'],first_inside=sum(first),first_total=len(first),
                horizon_inside=sum(full),horizon_total=len(full),complete_day_inside=sum(days),complete_day_total=len(days),
                final_partial_day_horizons=len(full)%48))
    (folder/'coverage.json').write_text(json.dumps(dict(results=results,formal_test_run=False,
        calibration_sha256=digest(out),interpretation='Empirical coverage only; correlated days, limited calibration and validation. Main factor remains 1.0.'),indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
