import copy
import json
import unittest
import numpy as np
from benchmark_validation import ROOT
from calibrated_box import provider,KEY
from forecast_closed_loop import price_at
from scenario_mean_qp import ScenarioMeanQP
from robust_mpc import RiskOptimizer


class ScenarioMeanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((ROOT/'formal_parameters_v1.json').read_text());cls.data=dict(np.load(ROOT/'data/formal_validation_inputs.npz'))

    def pair(self,n,S,changed=False):
        cfg=copy.deepcopy(self.cfg);t=75*48+24
        if changed:cfg['thermal']['C_kwh_per_C']=(np.array(cfg['thermal']['C_kwh_per_C'])*1.2).tolist()
        f=provider(self.data,t,KEY,n,8,S,np.array(cfg['pv']['ac_kw'][:n]));rng=np.random.default_rng(1467);temp=rng.uniform(21,25,n);prev=rng.uniform(0,1,n);prices=price_at(np.arange(t,t+8),cfg['controller'])
        a=ScenarioMeanQP(cfg,n,S).solve(f,temp,prev,prices);b=RiskOptimizer(cfg,n,S,risk='mean').solve(f,temp,prev,prices)
        np.testing.assert_allclose(a[2]['feasible_objective'],b[2]['feasible_objective'],atol=1e-6,rtol=1e-7)
        np.testing.assert_allclose(a[0],b[0],atol=2e-4,rtol=0)
        c=np.array(a[2]['scenario_control_certificate']['cooling_kw']);g=np.array(a[2]['scenario_control_certificate']['p2p_kw'])
        self.assertLess(abs(c[:,:,0]-c[:,0:1,0]).max(),1e-8);self.assertLess(abs(g.sum(axis=0)).max(),1e-8)

    def test_single_scenario(self):self.pair(3,1)
    def test_full_scenario_recourse(self):self.pair(10,30)
    def test_changed_parameter_case(self):self.pair(3,3,True)


if __name__=='__main__':unittest.main(verbosity=2)
