import unittest
from .reuse import reusable_prediction
class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.selected=dict(arm='MAG_ONLY',source='ettm1',seed=81552,sha256='same-adapter')
        self.prior=dict(arm='MAG_ONLY',source='ettm1',seed=[81552,81552,81551],panel='ettm1',kind='standard',checkpoint_sha256='same-adapter')
    def match(self,r):return reusable_prediction(r,self.selected,'ettm1','standard')
    def test_same_adapter_different_B0_rejected(self):
        self.assertFalse(self.match(dict(self.prior,seed=[81551,81552,81551])))
    def test_same_B0_same_adapter_accepted(self):
        self.assertTrue(self.match(self.prior));self.assertTrue(self.match(dict(self.prior,seed=81552)))
    def test_unknown_or_other_source_rejected(self):
        for source in [None,'electricity']:self.assertFalse(self.match(dict(self.prior,source=source)))
    def test_wrong_view_or_hash_rejected(self):
        for key,value in [('panel','electricity_transfer'),('kind','shape'),('checkpoint_sha256','other'),('arm','C3'),('seed',None)]:self.assertFalse(self.match(dict(self.prior,**{key:value})))
    def test_alias_only_applies_to_same_implemented_model(self):
        for alias,canonical in [('PLAIN','C2'),('B0','C0')]:
            row=dict(self.selected,arm=alias)
            self.assertTrue(reusable_prediction(dict(self.prior,arm=canonical),row,'ettm1','standard'))
if __name__=='__main__':unittest.main()
