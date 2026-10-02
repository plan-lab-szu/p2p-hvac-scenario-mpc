"""Predeclared validation-only sampling diagnostic; no device/weight tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_validation import simulate, digest
from forecast_closed_loop import ROOT, forecast


def sample_forecast(data,t,key,n,H,S,ac_cap,mode='scenario',strategy='resampled',seed=0):
    point=forecast(data,t,key,n,H,1,ac_cap,mode='deterministic')
    if mode=='deterministic':
        return point
    train_end=int(np.searchsorted(data['time'],np.datetime64('2013-12-31')))
    slot=t%48
    # Days 1..58: all 48 origins have H=8 future samples entirely in training.
    days=np.arange(1,(train_end-H)//48)
    if S>len(days) or S<1:
        raise ValueError('Insufficient complete training days')
    if strategy=='resampled':
        selected=np.random.default_rng(seed+t).choice(days,size=S,replace=False)
    elif strategy=='fixed_days':
        # Same selected historical days across all origins, homes and variables.
        # Historical trajectories advance coherently within each day. There is a
        # midnight reset; this is not a conditioned weather model or scenario tree.
        selected=np.random.default_rng(seed).choice(days,size=S,replace=False)
    else:
        raise ValueError(strategy)
    load=[]; pv=[]; ambient=[]
    for day in selected:
        ix=day*48+slot+np.arange(H)
        assert ix.max()<train_end and ix.min()>=48 and ix.max()<t
        load.append(np.maximum(0,point['point_load']+(data['base_kw'][ix,:n]-data['base_kw'][ix-48,:n]).T))
        pv.append(np.clip(point['point_pv']+(data[key+'_pv'][ix,:n]-data[key+'_pv'][ix-48,:n]).T,0,ac_cap[:,None]))
        ambient.append(point['point_ambient']+data[key+'_ambient'][ix]-data[key+'_ambient'][ix-48])
    return dict(load=np.stack(load,axis=1),pv=np.stack(pv,axis=1),ambient=np.array(ambient),
        point_load=point['point_load'],point_pv=point['point_pv'],point_ambient=point['point_ambient'],
        max_source_index=int(max(point['max_source_index'],selected.max()*48+slot+H-1)))


def study(folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(start='2013-12-31',steps=17*48,warmup=48,evaluation='2014-01-01 through 2014-01-16, all dates',
        homes=3,case='sydney_utc10_end',method='central',counts=[3,10,30],seeds=[101,202,303],
        strategies=['resampled','fixed_days'],parameter_sha256=digest(ROOT/'formal_parameters_v1.json'),
        script_sha256=digest(__file__),test_dates_used=False,
        purpose='Diagnose seed/scenario-count stability; no automatic parameter selection or benefit guarantee',
        unchanged='All physical parameters, weights, point forecasts, constraints and settlement rules',
        training_pool='Same 58 complete day-origin trajectories for all configurations; no marginal residual shuffling')
    manifest=folder/'protocol.json'
    if manifest.exists():
        assert json.loads(manifest.read_text())==protocol, 'Protocol changed; use a new output directory'
    else:
        manifest.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs=[('deterministic',1,0)]+[(a,s,seed) for a in protocol['strategies'] for s in protocol['counts'] for seed in protocol['seeds']]
    results=[]
    for strategy,S,seed in jobs:
        sub=folder/f'{strategy}_s{S}_seed{seed}'
        existing=list(sub.glob('*.json')) if sub.exists() else []
        complete=[p for p in existing if not p.name.endswith('.checkpoint.json')]
        if complete:
            r=json.loads(complete[0].read_text())
            assert r['status']=='passed', 'Keep failures, do not silently skip them'
        else:
            mode='deterministic' if strategy=='deterministic' else 'scenario'
            def provider(*args,**kwargs):
                return sample_forecast(*args,**kwargs,strategy=strategy,seed=seed)
            r=simulate(sub,mode,'central',n=3,start=protocol['start'],steps=protocol['steps'],warmup=48,
                scenario_count=S,forecast_fn=provider,experiment=dict(strategy=strategy,seed=seed,S=S,
                    protocol_sha256=digest(manifest),provider_sha256=digest(__file__)))
        results.append(dict(strategy=strategy,S=S,seed=seed,summary=r['summary']))
        (folder/'progress.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    base=results[0]['summary']; grouped=[]
    for strategy in protocol['strategies']:
        for S in protocol['counts']:
            g=[r['summary'] for r in results if r['strategy']==strategy and r['S']==S]
            item=dict(strategy=strategy,S=S)
            for metric in ('monetary_bill','realized_stage_cost','cold_degree_hours','hot_degree_hours','smoothing_cost'):
                values=[r[metric] for r in g]
                item[metric]=dict(mean=float(np.mean(values)),min=min(values),max=max(values))
            item['stage_cost_change_percent']=100*(item['realized_stage_cost']['mean']/base['realized_stage_cost']-1)
            grouped.append(item)
    summary=dict(protocol=protocol,completed=len(results),deterministic=base,groups=grouped,runs=results)
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('STABILITY SUMMARY',json.dumps(grouped),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,required=True)
    study(ap.parse_args().outdir)
