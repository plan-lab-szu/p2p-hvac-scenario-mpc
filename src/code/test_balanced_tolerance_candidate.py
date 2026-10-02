import unittest
from balanced_tolerance_candidate import candidate_penalty


class BalancedTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(primal=.00000255,ep=.01038,raw_max_kw=.000018,dual=.000148,proximal=.0000148,ed=.0001034)
    def test_reduce_settled_primal(self): self.assertEqual(candidate_penalty(.16,750,self.m),.08)
    def test_increase_settled_dual(self):
        self.m.update(raw_max_kw=.004394,dual=4.6e-8,proximal=4.6e-9)
        self.assertEqual(candidate_penalty(.01,500,self.m),.02)
    def test_no_change_when_passed(self):
        self.m['dual']=1e-5
        self.assertEqual(candidate_penalty(.16,500,self.m),.16)
    def test_cutoff(self): self.assertEqual(candidate_penalty(.16,775,self.m),.16)
    def test_cadence(self): self.assertEqual(candidate_penalty(.16,751,self.m),.16)
    def test_lower_bound(self): self.assertEqual(candidate_penalty(.0025,500,self.m),.0025)
    def test_upper_bound(self):
        self.m.update(raw_max_kw=.005,dual=1e-6,proximal=1e-7)
        self.assertEqual(candidate_penalty(.16,500,self.m),.16)


if __name__=='__main__': unittest.main(verbosity=2)
