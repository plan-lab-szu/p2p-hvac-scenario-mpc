"""Explicit alternate-configuration harness; legacy simulator/data remain intact."""
import json
from pathlib import Path
import time
import numpy as np
from benchmark_validation import ROOT,digest
from forecast_closed_loop import thermal_step,price_at,settle
from calibrated_box import KEY
from fitted_scenario_provider import provider
from scenario_mean_qp import ScenarioMeanQP
from reduced_mean_qp import ReducedMeanQP
from box_envelope_qp import BoxEnvelopeQP
from component_box import component_bounds
from balanced_tolerance_candidate import BalancedToleranceCandidate as AdaptivePenaltyParallel
from robust_mpc import RISK_SOLVER


def run(folder,cfg,data,experiment,n,mode='scenario',seed=202,start_label='2014-01-16',steps=672,formal=True):
    folder.mkdir(parents=True,exist_ok=True);path=folder/'run.json'
    if path.exists():raise FileExistsError(path)
    H=8;S=1 if mode=='deterministic' else 30;dt=cfg['dt_hours'];th=cfg['thermal'];p=cfg['controller']
    start=int(np.searchsorted(data['time'],np.datetime64(start_label)));assert start+steps<=len(data['time'])
    baseline=experiment['baseline'];method='pj' if baseline=='pj' else 'central'
    calibration=json.loads((ROOT/'results/component_box_20260914_v1/calibration.json').read_text())['populations'][str(n)]
    R=np.array(th['R_C_per_kw'][:n]);C=np.array(th['C_kwh_per_C'][:n]);cop=np.array(th['COP'][:n]);gain=np.array(th['internal_gain_kw_thermal'][:n]);cap=np.array(cfg['pv']['ac_kw'][:n])
    temp=np.full(n,th['initial_C']);prev=np.zeros(n);records=[]
    cls=ReducedMeanQP if baseline=='mean_open_loop' else BoxEnvelopeQP if baseline=='component_residual' else ScenarioMeanQP
    opt=AdaptivePenaltyParallel(cfg,n,S,'pj',workers=4) if method=='pj' else cls(cfg,n,S)
    optimizer_file={'mean_open_loop':'reduced_mean_qp.py','component_residual':'box_envelope_qp.py','pj':'balanced_tolerance_candidate.py'}.get(baseline,'scenario_mean_qp.py')
    out=dict(status='running',case=KEY,mode=mode,method=method,household_ids=cfg['household_ids'][:n],scenarios=S,
        start=start_label,steps=steps,warmup=48,full_configuration=cfg,experiment=experiment,formal_test_run=formal,
        base_parameter_sha256=digest(ROOT/'formal_parameters_v1.json'),base_input_sha256=digest(ROOT/experiment['input_path']),
        runner_sha256=digest(__file__),provider_sha256=digest(ROOT/'fitted_scenario_provider.py'),optimizer_sha256=digest(ROOT/optimizer_file),solver_settings=(__import__('forecast_closed_loop').SOLVER if method=='pj' else RISK_SOLVER),records=records)
    begun=time.perf_counter()
    try:
        for i,t in enumerate(range(start,start+steps)):
            f=provider(data,t,KEY,n,H,S,cap,mode=mode,seed=seed);prices=price_at(np.arange(t,t+H),p)
            if baseline=='component_residual':f['box_low'],f['box_high']=component_bounds(f,calibration,cap,'component_residual')
            initial_arrays={key:getattr(opt,key).copy() for key in ('z','u','old')} if method=='pj' else {}
            initial_rho=opt.cfg['controller']['rho'] if method=='pj' else None
            cool,trade,stats=opt.solve(f,temp,prev,prices)
            if method=='pj':
                cdir=folder/'clearing_certificates';cdir.mkdir(exist_ok=True);cp=cdir/f'step{i:04d}.npz'
                np.savez_compressed(cp,**dict(opt.last_arrays,**opt.last_clearing_certificate),
                    **{'initial_'+key:value for key,value in initial_arrays.items()})
                stats.update(initial_rho=initial_rho,final_rho=opt.cfg['controller']['rho'],
                    penalty_events=opt.penalty_events,penalty_trace=opt.trace,
                    initialization_policy='carry_coordination_and_penalty',attempts=1)
                stats['bounded_trade_certificate_file']=dict(path=cp.relative_to(folder).as_posix(),sha256=digest(cp))
            for kind in ('common_control_certificate','scenario_control_certificate','box_certificate'):
                if kind in stats:
                    cert=stats.pop(kind);cdir=folder/'certificates';cdir.mkdir(exist_ok=True);cp=cdir/f'step{i:04d}.npz'
                    np.savez_compressed(cp,**{k:np.asarray(v) for k,v in cert.items()});stats['certificate_file']=dict(kind=kind,path=cp.relative_to(folder).as_posix(),sha256=digest(cp))
            planned=f['point_load'][:,0]+cool-f['point_pv'][:,0]-trade
            predicted=thermal_step(temp,f['point_ambient'][0],cool,R,C,cop,gain,dt)
            # No current realized input is accessed before the control solve.
            load=data['base_kw'][t,:n];pv=data[KEY+'_pv'][t,:n];amb=float(data[KEY+'_ambient'][t])
            actual=settle(load,pv,cool,trade,planned,prices[0],p,dt);T=thermal_step(temp,amb,cool,R,C,cop,gain,dt)
            violation=max(0.,float(actual['net'].max()-p['grid_import_max_kw']),float(-actual['net'].min()-p['grid_export_max_kw']))
            if violation>=1e-7 or abs(trade.sum())>=1e-8:
                out['failed_step_evidence']=dict(step=i,label=str(data['time'][t]),cooling_kw=cool.tolist(),p2p_kw=trade.tolist(),actual_grid_kw=actual['net'].tolist(),temperature_after_C=T.tolist(),capacity_violation_kw=violation,stats=stats)
                raise AssertionError('Actual electrical/clearing violation; no hidden recourse')
            low=np.maximum(th['comfort_low_C']-T,0);high=np.maximum(T-th['comfort_high_C'],0)
            records.append(dict(step=i,label=str(data['time'][t]),warmup=i<48,temperature_before_C=temp.tolist(),temperature_after_C=T.tolist(),predicted_temperature_C=predicted.tolist(),
                cooling_kw=cool.tolist(),p2p_kw=trade.tolist(),actual_load_kw=load.tolist(),actual_pv_kw=pv.tolist(),ambient_actual_C=amb,
                forecast_load_kw=f['point_load'][:,0].tolist(),forecast_pv_kw=f['point_pv'][:,0].tolist(),ambient_forecast_C=float(f['point_ambient'][0]),max_source_index=f['max_source_index'],
                actual_grid_kw=actual['net'].tolist(),planned_grid_kw=planned.tolist(),balancing_kw=actual['deviation'].tolist(),grid_bill=actual['bill'],balancing_fee=actual['fee'],
                comfort_low_C=low.tolist(),comfort_high_C=high.tolist(),comfort_penalty=float(dt*p['comfort_slack_AUD_per_degree_hour']*(low+high).sum()),
                tracking_cost=float(dt*p['comfort_tracking_weight']*((T-th['setpoint_C'])**2).sum()),smoothing_cost=float(p['smoothing_weight']*((cool-prev)**2).sum()),
                capacity_violation_kw=violation,balance_error_kw=float(abs(load+cool-pv-actual['net']-trade).max()),clearing_error_kw=float(abs(trade.sum())),**stats))
            temp=T;prev=cool
            if i%6==0:
                path.with_suffix('.checkpoint.json').write_text(json.dumps(out),encoding='utf-8')
                print(json.dumps(dict(n=n,baseline=baseline,seed=seed,step=i,formal=formal)),flush=True)
        ev=records[48:];arr=lambda k:np.array([r[k] for r in ev]);total=lambda k:float(arr(k).sum())
        summary={k:total(k) for k in ('grid_bill','balancing_fee','comfort_penalty','tracking_cost','smoothing_cost')}
        summary.update(monetary_bill=summary['grid_bill']+summary['balancing_fee'],realized_stage_cost=sum(summary.values()),
            cold_degree_hours=dt*total('comfort_low_C'),hot_degree_hours=dt*total('comfort_high_C'),hvac_kwh=dt*total('cooling_kw'),
            temperature_min_C=float(arr('temperature_after_C').min()),temperature_max_C=float(arr('temperature_after_C').max()),
            max_balance_error_kw=float(arr('balance_error_kw').max()),max_clearing_error_kw=float(arr('clearing_error_kw').max()),
            max_relative_gap=float(arr('relative_objective_gap').max()),evaluated_steps=steps-48)
        out.update(status='passed',summary=summary)
    except Exception as error:
        out.update(status='failed',failure_step=len(records),error=repr(error))
        if method=='pj':
            out['failed_solver_trace']=opt.trace
            out['failed_penalty_events']=opt.penalty_events
            out['failed_initial_rho']=initial_rho
            if opt.last_arrays is not None:
                fp=folder/'failed_solver_arrays.npz'
                np.savez_compressed(fp,**opt.last_arrays,**{'initial_'+key:value for key,value in initial_arrays.items()})
                out['failed_solver_arrays']=dict(path=fp.name,sha256=digest(fp))
    if method=='pj':opt.close();out['workers_closed']=all(not w.is_alive() for w in opt.processes)
    out['elapsed_seconds']=time.perf_counter()-begun;path.write_text(json.dumps(out,indent=2),encoding='utf-8')
    return out
