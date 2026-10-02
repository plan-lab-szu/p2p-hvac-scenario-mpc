"""Euclidean projection onto public trade box intersected with community zero sum.

Only a numerical postprocessing candidate. No physical target, HVAC decision,
iteration update, private load bound or stopping threshold is changed here.
"""
import numpy as np


def project_trade_box_zero_sum(values,limit):
    v=np.asarray(values,dtype=float)
    if v.ndim!=2 or v.shape[0]<1 or not np.isfinite(v).all() or not np.isfinite(limit) or limit<=0:
        raise ValueError('Expected finite homes x components and positive common limit')
    # z_i=clip(v_i-lambda,-limit,limit), with a scalar multiplier per column.
    lo=np.min(v-limit,axis=0);hi=np.max(v+limit,axis=0)
    for _ in range(80):
        mid=(lo+hi)/2;z=np.clip(v-mid,-limit,limit);positive=z.sum(axis=0)>0
        lo=np.where(positive,mid,lo);hi=np.where(positive,hi,mid)
    z=np.clip(v-(lo+hi)/2,-limit,limit)
    if abs(z.sum(axis=0)).max()>1e-9 or abs(z).max()>limit+1e-12:
        raise ArithmeticError('Bounded clearing projection did not meet numerical accuracy')
    return z
