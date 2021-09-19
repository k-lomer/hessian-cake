import tensorflow as tf
import numpy as np
from operations import multi_dim_dot
from scipy.linalg import cho_factor, cho_solve, LinAlgError

class TrustRegion:
    """
    Maintain the trust region radius and calculate intersections
    """
    def __init__(self, radius=0.1, eta1=0.25, eta2=0.75, gamma1=0.25, gamma2=2, verbose=False):
        """
        :param radius: float: the radius of the trust region ball
        :param eta1: float: lower threshold for updating trust region
        :param eta2: float: upper threshold for updating trust region
        :param gamma1: float: factor for shrinking trust region
        :param gamma2: float: factor for growing trust region
        :param verbose: bool: whether to print log messages
        """
        self.r = radius
        self.max_r = 100
        self.eta1 = eta1
        self.eta2 = eta2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.verbose = verbose

    def update(self, rho, on_boundary):
        """
        Update the trust region radius
        :param rho: float: the measurement of how good the quadratic approximation is
        :param on_boundary: bool: whether the step was on the boundary
        :return: bool: whether to keep the update or not
        """
        if self.verbose:
            print(f"r: {self.r}, rho: {rho},", end=" ")
        if rho < self.eta1:
            if self.verbose:
                print("reduce")
            self.r *= self.gamma1
            return False
        elif rho > self.eta2 and on_boundary:
                if self.verbose:
                    print("increase")
                self.r = min(self.gamma2 * self.r, self.max_r)
        else:
            if self.verbose:
                if not on_boundary:
                    print("no update - within boundary")
                else:
                    print("no update - average rho")
        return True

    def subproblem(self, p, grad, Hp):
        """
        evaluate the TR subproblem
        :param p: list(tf.Tensor): the step direction
        :param grad: list(tf.Tensor): the gradient
        :param Hp: list(tf.Tensor): the hessian vector product with p
        :return: tf.Tensor: the evaluation of the subproblem
        """
        return multi_dim_dot(grad, p) + 0.5 * multi_dim_dot(p, Hp)

    def tr_step(self, p):
        """
        Scale a step in a fixed direction, limited by the TR radius
        :param p: list(tf.Tensor): the direction to move in
        :return: list(tf.Tensor), bool: the correctly scaled step and whether or not it is on the boundary
        """
        p_dot_p = multi_dim_dot(p, p)
        # adjust stepsize if it is beyond the boundary
        if p_dot_p > self.r**2:
            on_boundary = True
            step_size = self.r / tf.sqrt(p_dot_p)
            p = [step_size * p_i for p_i in p]
        else:
            on_boundary = False

        return p, on_boundary

    def dogleg_point(self, v1, v2, v1_dot_v1=None, v2_dot_v2=None, v1_dot_v2=None):
        """
        Compute the dogleg point on the radius
        expect v1 (cauchy point) in trust region, v2 (newton point) out of trust region
        find 1D dogleg minimum, for 1 < tau < 2
        ||v1 + (tau - 1)(v2 - v1)||^2 = radius^2
        Dogleg point is v1 + (tau - 1)(v2 - v1)
        :param v1: list(tf.Tensor): cauchy point in the trust region
        :param v2: list(tf.Tensor): newton point out of the trust region
        :param v1_dot_v1: tf.Tensor: v1^T * v1, use if available to save computation
        :param v2_dot_v2: tf.Tensor: v2^T * v2, use if available to save computation
        :param v1_dot_v2: tf.Tensor: v1^T * v2, use if available to save computation
        :return: list(tf.Tensor): dogleg point
        """

        if v1_dot_v1 is None:
            v1_dot_v1 = multi_dim_dot(v1, v1)
        if v2_dot_v2 is None:
            v2_dot_v2 = multi_dim_dot(v2, v2)
        if v1_dot_v2 is None:
            v1_dot_v2 = multi_dim_dot(v1, v2)
        tau_a = v1_dot_v1 - 2 * v1_dot_v2 + v2_dot_v2
        tau_b = -4 * v1_dot_v1 + 6 * v1_dot_v2 - 2 * v2_dot_v2
        tau_c = 4 * v1_dot_v1 - 4 * v1_dot_v2 + v2_dot_v2 - self.r ** 2

        discriminant = tau_b ** 2 - 4 * tau_a * tau_c
        assert discriminant > 0
        tau = (-tau_b + tf.sqrt(discriminant)) / (2 * tau_a)
        if not (1 <= tau <= 2):
            tau = (-tau_b - tf.sqrt(discriminant)) / (2 * tau_a)

        assert 1 <= tau <= 2
        # compute dogleg point
        for i in range(len(v2)):
            v2[i] = v1[i] + (tau - 1) * (v2[i] - v1[i])

        return v2

    # method from https://nmayorov.wordpress.com/2015/07/01/2d-subspace-trust-region-method/
    def subspace_2D_minimization(self, B, g):
        """
        compute the value of p which minimizes p^tBp + p^tg subject to p^tp <= delta^2
        :param B: ndarray: shape (2, 2) symmetric matrix
        :param g: ndarray: shape (2,)
        :return: ndarray, bool: shape(2,) the minimizing vector p and whether it is on the boundary of the trust region
        """
        try:
            R, lower = cho_factor(B)
            p = -cho_solve((R, lower), g)
            if np.dot(p, p) < self.r ** 2:
                return p, False
        except LinAlgError:
            pass

        a = B[0, 0] * self.r ** 2
        b = B[0, 1] * self.r ** 2
        c = B[1, 1] * self.r ** 2

        d = g[0] * self.r
        f = g[1] * self.r

        coeffs = np.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
        try:
            t = np.roots(coeffs)  # Can handle leading zeros.
        except np.linalg.LinAlgError as e:
            print(coeffs)
            print(B)
            print(g)
            raise e
        t = np.real(t[np.isreal(t)])

        p = self.r * np.vstack((2 * t / (1 + t ** 2), (1 - t ** 2) / (1 + t ** 2)))
        value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)
        i = np.argmin(value)
        p = p[:, i]

        return p, True

