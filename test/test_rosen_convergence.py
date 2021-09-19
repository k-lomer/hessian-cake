import tensorflow as tf
import unittest
from models.rosenbrock import Rosenbrock
from optimizers.inexact_newton import InexactNewton
from optimizers.sgd import StochasticGradientDescent
from optimizers.adam import Adam
from operations import multi_dim_dot
import numpy as np

class TestConvergance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Enable eager execution
        tf.compat.v1.enable_eager_execution()

    def setUp(self):
        self.r = Rosenbrock([-1, -1])

        # get params for inexact newton tests
        self.params = InexactNewton.get_default_params()
        self.params["regularization"] = 0
        self.params["subsamples"] = None

    def run_test(self, opt, max_iter):
        i = 0
        g = opt.gradient()

        while i < max_iter and multi_dim_dot(g, g) > 1e-8:
            opt.minimize()
            g = opt.gradient()
            i += 1

        print(i)
        opt_vars = [v.numpy() for v in self.r.variables]
        self.assertTrue(np.allclose(opt_vars, [1.0, 1.0], atol=1e-3))

    @unittest.skip("slow test")
    def test_sgd(self):
        opt = StochasticGradientDescent(0.01, self.r.variables, self.r)
        self.run_test(opt, 5000)

    @unittest.skip("slow test")
    def test_adam(self):
        opt = Adam(0.1, self.r.variables, self.r)
        self.run_test(opt, 5000)

    def test_inexact_newton_bt(self):
        self.params["globalization"] = 'bt'
        opt = InexactNewton(1.0, self.r.variables, self.r, self.params)
        self.run_test(opt, 100)

    def test_inexact_newton_tr_step(self):
        self.params["globalization"] = 'tr'
        self.params["tr_type"] = 'step'
        opt = InexactNewton(1.0, self.r.variables, self.r, self.params)
        self.run_test(opt, 100)

    def test_inexact_newton_tr_dogleg(self):
        self.params["globalization"] = 'tr'
        self.params["tr_type"] = 'dogleg'
        opt = InexactNewton(1.0, self.r.variables, self.r, self.params)
        self.run_test(opt, 100)

    def test_inexact_newton_tr_2d_subspace(self):
        self.params["globalization"] = 'tr'
        self.params["tr_type"] = 'subspace_2d'
        opt = InexactNewton(1.0, self.r.variables, self.r, self.params)
        self.run_test(opt, 100)


    def test_bt_momentum(self):
        self.params["globalization"] = 'bt'
        self.params["momentum"] = True
        opt = InexactNewton(1.0, self.r.variables, self.r, self.params)
        self.run_test(opt, 100)

if __name__ == "__main__":
    # Run Tests
    unittest.main()
