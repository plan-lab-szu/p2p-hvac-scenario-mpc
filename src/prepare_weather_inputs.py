"""Weather/PV preprocessing copied from the frozen preparation functions.
No download, original identifiers or raw household records are included.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def pv_power(gti,ambient,dc_kw):
    """Simplified NOCT + PVWatts-inspired DC equation; not full PVWatts v8."""
    cell=ambient+(49.-20.)/800.*gti
    dc=dc_kw*(gti[...,None]/1000.)*np.maximum(0,1-.0037*(cell[...,None]-25.))
    return np.clip(dc*.86*.96,0,dc_kw/1.2)


def align_weather(hourly,times,offset,end_label=True):
    """Hourly radiation is backward-looking mean, NOT an instantaneous point."""
    index=pd.DatetimeIndex(hourly['time'])
    temp=np.asarray(hourly['temperature_2m'],float)
    gti=np.asarray(hourly['global_tilted_irradiance'],float)
    assert np.isfinite(temp).all() and np.isfinite(gti).all() and (gti>=0).all()
    end=times-pd.Timedelta(hours=offset)+(pd.Timedelta(0) if end_label else pd.Timedelta(minutes=30))
    begin=end-pd.Timedelta(minutes=30)
    assert begin.min()>=index.min() and end.max()<=index.max()
    ambient=.5*(np.interp(begin.as_unit('ns').asi8,index.as_unit('ns').asi8,temp)
                +np.interp(end.as_unit('ns').asi8,index.as_unit('ns').asi8,temp))
    # Both half-hours in (h-1,h] use the mean labelled h, preserving energy.
    rad_index=index.get_indexer(end.ceil('h'))
    assert (rad_index>=0).all()
    return ambient,gti[rad_index]




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--demand', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('No overwrite')
    cfg = json.loads((Path(__file__).parent/'configuration_without_identifiers.json').read_text())
    with np.load(args.demand, allow_pickle=False) as src:
        times = pd.DatetimeIndex(src['time'])
        base = src['base_kw'].copy()
    if base.shape != (len(times), 100) or not np.isfinite(base).all() or (base < 0).any():
        raise ValueError('Expected complete nonnegative 100-column base demand, kW')
    if not np.all(np.diff(times) == np.timedelta64(30, 'm')):
        raise ValueError('Time rows must be half-hourly and ordered')
    weather = json.loads(args.weather.read_text(encoding='utf-8'))
    w = weather[0] if isinstance(weather, list) else weather
    if w['utc_offset_seconds'] != 0:
        raise ValueError('Weather source must use UTC labels')
    ambient, gti = align_weather(w['hourly'], times, 10, True)
    pv = pv_power(gti, ambient, np.array(cfg['pv']['dc_kw']))
    np.savez_compressed(args.output, time=times.to_numpy(), base_kw=base,
        sydney_utc10_end_ambient=ambient, sydney_utc10_end_gti=gti,
        sydney_utc10_end_pv=pv)
    print(json.dumps(dict(rows=len(times), homes=100, original_ids_included=False)))


if __name__ == '__main__':
    main()
