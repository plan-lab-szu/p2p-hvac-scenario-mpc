"""Observe the frozen bounded solver without changing its iteration or stop."""
import numpy as np
from bounded_repair_parallel import BoundedRepairParallel
from forecast_closed_loop import MARKET_TOL_KW


def observe_iteration(gamma, z_before, u_before, old, rho, beta):
    n, D = gamma.shape
    v = gamma + u_before
    z = v - v.mean(axis=0, keepdims=True)
    u = u_before + gamma - z
    primal = float(np.linalg.norm(gamma - z))
    dual = float(rho * np.linalg.norm(z - z_before))
    proximal = float(beta * np.linalg.norm(gamma - old))
    ep = float(1e-4*np.sqrt(n*D) + 1e-6*max(np.linalg.norm(gamma), np.linalg.norm(z)))
    ed = float(1e-6*np.sqrt(n*D) + 1e-6*np.linalg.norm(rho*u))
    sums = gamma.sum(axis=0); index = int(np.argmax(abs(sums)))
    raw = float(abs(sums[index]))
    metrics = dict(primal=primal, dual=dual, proximal=proximal, ep=ep, ed=ed,
        raw_max_kw=raw, raw_first_kw=float(abs(sums[0])), raw_coordinate=index,
        primal_ok=primal<=ep, dual_ok=dual<=ed, proximal_ok=proximal<=ed,
        raw_clearing_ok=raw<=MARKET_TOL_KW)
    return metrics, dict(raw_gamma=gamma.copy(), consensus_z=z, u_after=u,
                         z_before=z_before.copy(), u_before=u_before.copy(), old_before=old.copy())


class ObservedBoundedParallel(BoundedRepairParallel):
    # Inherit solve unchanged. Observation does not mutate inputs or responses.
    def dispatch(self, command, payloads):
        if command == 'prepare':
            self.trace = []; self.last_arrays = None
        replies = super().dispatch(command, payloads)
        if command == 'iterate':
            gamma = np.concatenate([r['gamma'] for r in replies], axis=0)
            p = self.cfg['controller']
            metrics, self.last_arrays = observe_iteration(gamma, self.z, self.u, self.old, p['rho'], p['beta'])
            self.trace.append(dict(iteration=len(self.trace)+1, **metrics))
        return replies
