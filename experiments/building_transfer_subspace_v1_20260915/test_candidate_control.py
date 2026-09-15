import unittest
from run import candidate_gate
class CandidateControls(unittest.TestCase):
    def test_fixed120_is_in_strongest_reported_control(self):
        rows=[]
        for b in range(8):
            for h in [3,14]:
                for arm,score in [('F0',1),('AFFINE',1),('LOCAL',1),('LOCAL_FIXED120',.5),('POOLED',1),('POOLED_AFFINE',1),('WARM',1),('COEFFICIENT',.9)]:
                    rows.append(dict(role='dev',episode=f'{b}_H{h}',building=str(b),days=h,arm=arm,scaled_RMSE=score))
        result=candidate_gate(rows,'POOLED')
        self.assertFalse(result['pass']);self.assertEqual(result['best_reported_control'],'LOCAL_FIXED120')
if __name__=='__main__':unittest.main()
