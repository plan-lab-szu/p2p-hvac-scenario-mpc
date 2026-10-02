"""Replay the archived cohort refinement, without emitting household IDs.

Only use trusted local audit pickles. This validates cached refinement, not a
fresh scan of the upstream 344-million-row archive or metadata screening.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('No overwrite')
    folder = args.audit_dir
    initial = json.loads((folder/'preliminary.json').read_text())
    verified = json.loads((folder/'verified.json').read_text())
    final = json.loads((folder/'refined.json').read_text())
    raw = pd.read_pickle(folder/'captured200_raw.pkl')
    pool = set(verified['joint_conservative_warm_ids'])
    selected = []
    rejected = 0
    periods = [('2013-11-01','2014-01-29'), ('2013-07-01','2013-09-28')]
    for h in initial['capture_ids']:
        if h not in pool:
            continue
        g = raw.loc[raw.CUSTOMER_ID.eq(h)].sort_values('time')
        eligible = True
        for start,end in periods:
            times = pd.date_range(start,pd.Timestamp(end)+pd.Timedelta(hours=23,minutes=30),freq='30min')
            w = g.loc[g.time.between(times[0],times[-1])]
            np.testing.assert_array_equal(w.time.to_numpy(), times.to_numpy())
            energy = (w.GENERAL_SUPPLY_KWH+w.CONTROLLED_LOAD_KWH).to_numpy()
            if not np.isfinite(energy).all() or (energy < 0).any():
                raise ValueError('Invalid source energy')
            edges = np.r_[True,energy[1:]!=energy[:-1],True]
            eligible &= int(np.diff(np.flatnonzero(edges)).max()) < 24
        if eligible:
            selected.append(h)
        else:
            rejected += 1
    selected = selected[:100]
    if selected != final['provisional100_ids']:
        raise ValueError('Archived cohort ordering not reproduced')
    reference = pd.read_pickle(folder/'refined100_raw.pkl')
    replay = raw.loc[raw.CUSTOMER_ID.isin(selected)].copy()
    pd.testing.assert_frame_equal(replay.reset_index(drop=True),reference.reset_index(drop=True))
    for h,g in reference.groupby('CUSTOMER_ID'):
        digest = hashlib.sha256(g[['GENERAL_SUPPLY_KWH','CONTROLLED_LOAD_KWH']].to_numpy(dtype='<f8').tobytes()).hexdigest()
        if digest != final['selected_curve_hashes'][str(h)]:
            raise ValueError('Curve hash mismatch')
    result = dict(status='passed', homes=len(selected), source_rows=len(reference),
                  rejected_constant_runs=rejected, identical_order_and_raw_records=True,
                  all_curve_hashes_match=True, original_ids_emitted=False,
                  scope='Cached refinement replay only; upstream archive/metadata scan not repeated')
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
