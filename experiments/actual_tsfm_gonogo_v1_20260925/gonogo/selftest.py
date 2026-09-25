import unittest
import numpy as np
import torch
from torch import nn
from .lora import ScaleLoRALinear
from .candidate_a import downsample, upsample
from .candidate_c import mse


class Tests(unittest.TestCase):
    def test_independent_lora_zero_b_preserves(self):
        b = nn.Linear(5, 4)
        m = ScaleLoRALinear(b, 3, 2, 4, 'independent')
        x = torch.randn(3, 5)
        self.assertTrue(torch.equal(m(x), b(x)))

    def test_shared_lora_zero_b_preserves(self):
        b = nn.Linear(5, 4)
        m = ScaleLoRALinear(b, 3, 2, 4, 'gated_shared')
        x = torch.randn(3, 5)
        self.assertTrue(torch.equal(m(x), b(x)))

    def test_scale_selection_changes_independent_params(self):
        b = nn.Linear(3, 2, bias=False)
        m = ScaleLoRALinear(b, 2, 1, 1, 'independent')
        with torch.no_grad():
            m.B[0].fill_(1); m.B[1].fill_(2)
            m.A.fill_(1)
        x = torch.ones(1, 3)
        m.active = 0; a = m(x)
        m.active = 1; c = m(x)
        self.assertFalse(torch.equal(a, c))

    def test_down_up_shape(self):
        x = torch.arange(24.0).reshape(1, 24)
        y = downsample(x, 4)
        self.assertEqual(y.shape[-1], 6)
        self.assertEqual(upsample(y, 4, 24).shape[-1], 24)

    def test_c_mse(self):
        P = np.zeros((2, 3, 1, 2))
        Y = np.ones((2, 1, 2))
        w = np.array([1, 0, 0])
        self.assertAlmostEqual(mse(P, Y, w), 1.0)
