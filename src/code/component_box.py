"""Componentwise asymmetric calibration; no validation-dependent selection."""
import json
import numpy as np
from benchmark_validation import ROOT,digest
from calibrated_box import feature_values,KEY,provider as original_provider

FOLDER=ROOT/'results/component_box_20260914_v1'
PROTOCOL=ROOT/'COMPONENT_BOX_PROTOCOL_2026-09-14.md'


def fit_components(data,n):
    x=feature_values({k:v[:60*48] for k,v in data.items()},n)
    assert len(x)==60*48 and np.isfinite(x).all()
    floor=np.r_[np.full(2*n,.1),1.]
    scale=np.maximum(np.quantile(abs(x[48:1920]-x[:1872]),.9,axis=0),floor)
    low=[];high=[]
    for day in range(40,59):
        start=day*48;err=x[start:start+55]-x[start-48:start+7]
        low.append(err.min(axis=0));high.append(err.max(axis=0))
    low=np.array(low);high=np.array(high);candidates=[]
    for dropped in range(19):
        keep=np.arange(19)!=dropped
        lo=np.minimum(low[keep].min(axis=0),-floor);hi=np.maximum(high[keep].max(axis=0),floor)
        logs=np.log((hi-lo)/(2*scale))
        score=float((logs[:n].mean()+logs[n:2*n].mean()+logs[-1])/3)
        candidates.append(dict(dropped_day=40+dropped,score=score,low=lo.tolist(),high=hi.tolist()))
    chosen=min(candidates,key=lambda row:(row['score'],row['dropped_day']))
    lo=np.array(chosen['low']);hi=np.array(chosen['high'])
    included=np.all(low>=lo-1e-12,axis=1)&np.all(high<=hi+1e-12,axis=1)
    return dict(n=n,residual_low=lo.tolist(),residual_high=hi.tolist(),fit_scale=scale.tolist(),
        daily_min=low.tolist(),daily_max=high.tolist(),candidates=candidates,
        selected_dropped_day=chosen['dropped_day'],selected_score=chosen['score'],
        calibration_days_inside=int(included.sum()),included_days=(40+np.where(included)[0]).tolist(),
        support_min=x.min(axis=0).tolist(),support_max=x.max(axis=0).tolist(),
        fit_stop_exclusive=1920,calibration_origin_days=list(range(40,59)),
        support_stop_exclusive=2880,probability_guarantee=False)


def component_bounds(point,cal,cap,variant='component_support'):
    n=len(cap);center=np.vstack([point['point_load'],point['point_pv'],point['point_ambient'][None,:]])
    lo=center+np.array(cal['residual_low'])[:,None];hi=center+np.array(cal['residual_high'])[:,None]
    lo[:n]=np.maximum(lo[:n],0)
    lo[n:2*n]=np.clip(lo[n:2*n],0,cap[:,None]);hi[n:2*n]=np.clip(hi[n:2*n],0,cap[:,None])
    if variant=='component_support':
        hi[:n]=np.minimum(hi[:n],np.array(cal['support_max'])[:n,None])
        lo[-1]=np.maximum(lo[-1],cal['support_min'][-1]);hi[-1]=np.minimum(hi[-1],cal['support_max'][-1])
    elif variant!='component_residual':raise ValueError(variant)
    return lo,hi


def provider(data,t,key,n,H,S,ac_cap,mode='scenario',calibration=None,variant='component_support'):
    f=original_provider(data,t,key,n,H,S,ac_cap,mode=mode)
    if calibration is None:raise ValueError('Frozen component calibration required')
    f['box_low'],f['box_high']=component_bounds(f,calibration,ac_cap,variant)
    if np.any(f['box_low']>f['box_high']):raise ValueError('Empty empirical-support intersection; no clipping fallback')
    return f


def electrical_certificate(lo,hi,cfg):
    """Exact common-trade feasibility with zero cooling and independent grid recourse."""
    n=(len(lo)-1)//2;p=cfg['controller']
    minimum=np.maximum(-p['trade_max_kw'],hi[:n]-lo[n:2*n]-p['grid_import_max_kw'])
    maximum=np.minimum(p['trade_max_kw'],lo[:n]-hi[n:2*n]+p['grid_export_max_kw'])
    violations=np.array([float((lo-hi).max()),float((minimum-maximum).max()),
        float(minimum.sum(axis=0).max()),float((-maximum.sum(axis=0)).max())])
    return dict(feasible=bool(violations.max()<=1e-9),max_violation_kw=float(max(0.,violations.max())),
        trade_low=minimum,trade_high=maximum)


def inspect(data,cfg,cal,variant):
    n=cal['n'];cap=np.array(cfg['pv']['ac_kw'][:n]);x=feature_values(data,n)
    first=[];full=[];group=[[],[],[]];empty=[];bad=[];worst=0.;witness=None;ambient_min=np.inf;ambient_max=-np.inf
    from forecast_closed_loop import forecast
    for t in range(2880,len(x)):
        f=forecast(data,t,KEY,n,8,1,cap,mode='deterministic');lo,hi=component_bounds(f,cal,cap,variant)
        is_empty=bool(np.any(lo>hi));empty.append(is_empty)
        cert=electrical_certificate(lo,hi,cfg);bad.append(not cert['feasible']);worst=max(worst,cert['max_violation_kw'])
        if not cert['feasible'] and witness is None:witness=dict(index=t,label=str(data['time'][t]),violation_kw=cert['max_violation_kw'])
        ambient_min=min(ambient_min,float(lo[-1].min()));ambient_max=max(ambient_max,float(hi[-1].max()))
        first.append(bool(np.all(x[t]>=lo[:,0]-1e-8) and np.all(x[t]<=hi[:,0]+1e-8)))
        if t+8<=len(x):
            mask=(x[t:t+8].T>=lo-1e-8)&(x[t:t+8].T<=hi+1e-8)
            full.append(bool(mask.all()))
            for j,sl in enumerate((slice(0,n),slice(n,2*n),slice(2*n,2*n+1))):group[j].append(bool(mask[sl].all()))
    days=[all(full[i:i+48]) for i in range(0,len(full)-47,48)]
    return dict(n=n,variant=variant,first_inside=sum(first),first_total=len(first),horizon_inside=sum(full),horizon_total=len(full),
        complete_day_inside=sum(days),complete_day_total=len(days),partial_day_horizons=len(full)%48,
        group_horizon_inside=dict(zip(('load','pv','ambient'),map(sum,group))),
        empty_origins=sum(empty),zero_cooling_infeasible_origins=sum(bad),max_certificate_violation_kw=worst,witness=witness,
        ambient_interval_union_C=[ambient_min,ambient_max],checks_count=len(first))


def main():
    data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'));cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text())
    FOLDER.mkdir(parents=True,exist_ok=True);path=FOLDER/'calibration.json'
    if path.exists():raise FileExistsError('Preserve frozen calibration')
    calibration={str(n):fit_components(data,n) for n in (3,10,50,100)}
    path.write_text(json.dumps(dict(populations=calibration,source_sha256=digest(__file__),protocol_sha256=digest(PROTOCOL),
        parameter_sha256=digest(ROOT/'formal_parameters_v1.json'),input_sha256=digest(ROOT/'data/formal_validation_inputs.npz'),
        formal_test_run=False),indent=2),encoding='utf-8')
    # This immutable calibration is persisted before ANY validation inspection.
    rows=[inspect(data,cfg,calibration[str(n)],v) for n in (3,10,50,100) for v in ('component_residual','component_support')]
    (FOLDER/'screening.json').write_text(json.dumps(dict(results=rows,calibration_sha256=digest(path),source_sha256=digest(__file__),
        formal_test_run=False,probability_guarantee=False),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
