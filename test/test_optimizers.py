import tensorflow as tf
import numpy as np
import unittest
from optimizers.optimizer import Optimizer
from optimizers.inexact_newton import InexactNewton
from models.rosenbrock import Rosenbrock

class QuadSum:
    def __init__(self, x):
        assert len(x) >= 4
        self.variables = [tf.Variable(x_i, dtype=tf.float32) for x_i in x]

    def __call__(self):
        x = self.variables
        return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2

class QuadSumCoeffs:
    def __init__(self, x, coeffs):
        assert len(x) >= 4
        assert len(x) == len(coeffs)
        self.c = coeffs
        self.variables = [tf.Variable(x_i, dtype=tf.float32) for x_i in x]

    def __call__(self):
        x = self.variables
        return sum(cx[0]*cx[1]**2 for cx in zip(self.c, x))


class TestAutogradFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Enable eager execution
        tf.compat.v1.enable_eager_execution()


    def test_grad(self):

        r = Rosenbrock([0, 0])
        opt = Optimizer(0.01, r.variables, r)
        g = opt.gradient()
        self.assertTrue(np.array_equal(g, [-2, 0]))

        r = Rosenbrock([1, 1])
        opt = Optimizer(0.01, r.variables, r)
        g = opt.gradient()
        self.assertTrue(np.array_equal(g, [0, 0]))

        r = Rosenbrock([1, 0])
        opt = Optimizer(0.01, r.variables, r)
        g = opt.gradient()
        self.assertTrue(np.array_equal(g, [40, -20]))

        q = QuadSum([1, 1, 1, 1])
        opt = Optimizer(0.01, q.variables, q)
        g = opt.gradient()
        self.assertTrue(np.array_equal(g, [2, 2, 2, 2]))

        q = QuadSum([1, 2, 3, 4])
        opt = Optimizer(0.01, q.variables, q)
        g = opt.gradient()
        self.assertTrue(np.array_equal(g, [2, 4, 6, 8]))

        q = QuadSum([1, 1, 1, 1])
        opt = Optimizer(0.01, q.variables, q, regularization=0.1)
        g = opt.gradient()
        self.assertTrue(np.allclose(g, [2.1, 2.1, 2.1, 2.1]))

        r = Rosenbrock([1, 1])
        opt = Optimizer(0.01, r.variables, r, regularization=0.1)
        g = opt.gradient()
        self.assertTrue(np.allclose(g, [0.1, 0.1]))

    def test_H_v_prod(self):

        r = Rosenbrock([0, 0])
        opt = Optimizer(0.01, r.variables, r)
        hv = opt.hessian_v_prod(tf.constant([1, 1], dtype=tf.float32))
        self.assertTrue(np.array_equal(hv, [2, 20]))

        r = Rosenbrock([1, 0])
        opt = Optimizer(0.01, r.variables, r)
        hv = opt.hessian_v_prod(tf.constant([-1, 1], dtype=tf.float32))
        self.assertTrue(np.array_equal(hv, [-162, 60]))

        q = QuadSum([1, 1, 1, 1])
        opt = Optimizer(0.01, q.variables, q)
        hv = opt.hessian_v_prod(tf.constant([1, 1, 1, 1], dtype=tf.float32))
        self.assertTrue(np.array_equal(hv, [2, 2, 2, 2]))

        q = QuadSum([1, 2, 3, 4])
        opt = Optimizer(0.01, q.variables, q)
        hv = opt.hessian_v_prod(tf.constant([1, 2, 0, -1], dtype=tf.float32))
        self.assertTrue(np.array_equal(hv, [2, 4, 0, -2]))

        q = QuadSum([1, 1, 1, 1])
        opt_reg = Optimizer(0.01, q.variables, q, regularization=0.1)
        hv = opt_reg.hessian_v_prod(tf.constant([1, 1, 1, 1], dtype=tf.float32))
        self.assertTrue(np.allclose(hv, [2.1, 2.1, 2.1, 2.1]))


    def test_cg(self):
        q = QuadSum([1, 1, 1, 1])
        opt = InexactNewton(0.01, q.variables, q)
        x = opt.cg_solve(tf.constant([2, 4, 6, 8], dtype=tf.float32), 100, 1e-8)
        self.assertTrue(np.array_equal(x, [1, 2, 3, 4]))

    def test_lanczos(self):
        q = QuadSumCoeffs([1]*11, range(-5, 6))
        opt = InexactNewton(0.01, q.variables, q)
        evalues = opt.lanczos(k=10)
        self.assertTrue(np.isclose(max(evalues), 5*2))
        self.assertTrue(np.isclose(min(evalues), -5*2))



if __name__ == "__main__":
    # Run Tests
    unittest.main()

