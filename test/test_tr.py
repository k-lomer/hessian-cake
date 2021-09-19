import unittest
import tensorflow as tf
import numpy as np
from optimizers.trust_region import TrustRegion


class TestAutogradFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Enable eager execution
        tf.compat.v1.enable_eager_execution()

    def setUp(self):
        self.tr = TrustRegion(radius=1.0)

    def test_dogleg(self):
        v1 = [tf.Variable(x_i, dtype=tf.float32) for x_i in (0.5, 0)]
        v2 = [tf.Variable(x_i, dtype=tf.float32) for x_i in (1, 1)]
        p = self.tr.dogleg_point(v1, v2)
        self.assertTrue(np.allclose(p, [0.8, 0.6]))

    # minimize (x-1)^2 + (y-1)^2 from (0,0)
    def test_2d_subspace(self):
        x = 0
        y = 0
        g = np.array([2*(x - 1), 2*(y - 1)])
        B = np.array([[2, 0], [0, 2]])
        p = self.tr.subspace_2D_minimization(B, g)
        self.assertTrue(np.allclose(p, [1/np.sqrt(2), 1/np.sqrt(2)]))
