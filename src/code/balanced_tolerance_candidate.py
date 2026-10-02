"""Diagnostic candidate only: respond to unmet clearing with settled duals."""
from adaptive_penalty_parallel import AdaptivePenaltyParallel, next_penalty
from parallel_worker_optimizer import ParallelOptimizer
from bounded_failure_diagnostic import ObservedBoundedParallel


def candidate_penalty(rho, iteration, metrics):
    if iteration % 25 or iteration > 750:
        return rho
    primal = max(metrics['primal']/metrics['ep'], metrics['raw_max_kw']/.001)
    dual = max(metrics['dual'], metrics['proximal'])/metrics['ed']
    if primal > 1 and dual <= 1:
        return min(.16, 2*rho)
    if primal <= 1 and dual > 1:
        return max(.0025, rho/2)
    return next_penalty(rho, iteration, metrics)


class BalancedToleranceCandidate(AdaptivePenaltyParallel):
    def dispatch(self, command, payloads):
        p = self.cfg['controller']
        if command == 'prepare':
            self.penalty_events = []
            self.origin_initial_penalty = p['rho']
        if command == 'iterate' and self.trace:
            old_rho = p['rho']
            new_rho = candidate_penalty(old_rho, len(self.trace), self.trace[-1])
            if new_rho != old_rho:
                unscaled = old_rho*self.u.copy()
                replies = ParallelOptimizer.dispatch(self, 'penalty', [(new_rho,.1*new_rho) for _ in self.groups])
                assert all(r['rho'] == new_rho and r['beta'] == .1*new_rho for r in replies)
                self.u *= old_rho/new_rho
                p.update(rho=new_rho, beta=.1*new_rho)
                error = float(abs(new_rho*self.u-unscaled).max())
                assert error < 1e-12
                self.penalty_events.append(dict(after_iteration=len(self.trace),old_rho=old_rho,
                    new_rho=new_rho,unscaled_dual_preservation_error=error))
                payloads = [(self.z[ix]-self.u[ix], self.old[ix]) for ix in self.groups]
        # Bypass the legacy adaptation layer, keep the same observed iteration.
        replies = ObservedBoundedParallel.dispatch(self, command, payloads)
        if command == 'iterate':
            self.trace[-1].update(rho=p['rho'], beta=p['beta'])
        return replies
