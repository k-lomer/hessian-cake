import numpy as np
import tensorflow as tf
from optimizers.inexact_newton import InexactNewton
from operations import multi_dim_dot, flatten, equal, reshape
from optimizers.trust_region import TrustRegion


class InexactNewtonTR(InexactNewton):
    """
    Inexact Newton Trust Region Optimizer
    Uses a CG solver to find the newton direction inexactly
    Uses a Trust Region for globalization
    """

    def __init__(self, learning_rate, variables, loss, tr_type= "dogleg", params={}, verbose={}):
        """
        :param learning_rate: float: The step size for the method
        :param variables: list(tf.Variable): the trainable variables
        :param loss_func: NN_Loss: class representing objective function
        :param tr_type: string: the method of solving the TR subproblem
        :param params: dict: the inexact newton parameters
        :param verbose: dict: the verbosity settings
        Note loss_func could be a class other than NN_Loss if it contains the required methods
        """

        super(InexactNewtonTR, self).__init__(learning_rate, variables, loss, params, verbose)

        self.tr = TrustRegion(learning_rate, verbose=verbose.get("tr_update", False))
        self.tr_type = tr_type
        self.reduce_reg = params.get("reduce_tr_reg", 0.9)

    def tr_update(self, p, on_boundary):
        """
        Update the variables by the given step based on the trust region metric
        :param p: list(tf.Tensor): the step to update the variables by
        :param on_boundary: bool: whether the step is on the boundary
        """
        current_loss = self.loss()

        m = self.tr.subproblem(p, self.g, self.hessian_v_prod(p))
        for i in range(len(self.variables)):
            self.variables[i].assign_add(p[i])
        new_loss = self.loss()
        rho = (current_loss - new_loss) / -m
        update = self.tr.update(rho, on_boundary)
        if not update:  # undo variable update
            for i in range(len(self.variables)):
                self.variables[i].assign_sub(p[i])
            if self.cg_tol == "ew":
                self.ew.do_update = False
        elif self.momentum_type is not None:
            self.momentum_update(p)

        if not on_boundary and update:
            self.reg *= self.reduce_reg


    def lowcost_tr_minimize(self):
        """
        Solve the TR subproblem with a lowcost method
        This could be the dogleg method or 2D subspace method
        :return: list(tf.Tensor), bool: the step direction and whether it is on the boundary
        """
        H_g = self.hessian_v_prod(self.g)
        cauchy_coeff = self.g_dot_g / multi_dim_dot(self.g, H_g)
        if cauchy_coeff < 0:
            # if regularization fails, ensure we move in the negative gradient direction
            print("cauchy coefficient negative")
            cauchy_coeff = 1
        cauchy_step_size = cauchy_coeff ** 2 * self.g_dot_g

        # If cauchy point is beyond trust region then only use the gradient direction
        if cauchy_step_size >= self.tr.r**2:
            step_size = self.tr.r / tf.sqrt(self.g_dot_g)
            if self.verbose["globalization"]:
                print("lowcost gradient update using cauchy point")
            p = [step_size * -g_i for g_i in self.g]
            on_boundary = True
        else:
            tol = self.forcing_term()
            cauchy_point = [cauchy_coeff * -g_i for g_i in self.g]
            # if the cauchy point satisfies EW we can use the gradient as the newton step direction
            if self.satisfy_tol(cauchy_point, tol, [cauchy_coeff * -Hg_i for Hg_i in H_g]):
                p = cauchy_point
                if self.verbose["globalization"]:
                    print("cauchy point satisfies EW")
                on_boundary = False
            else:
                p = self.cg_solve([-g_i for g_i in self.g], tol)
                p_dot_p = multi_dim_dot(p, p)

                # if newton step is within trust region use best of that and the cauchy
                if p_dot_p <= self.tr.r**2 and self.tr_type == "dogleg":
                    # try cauchy point
                    for i in range(len(self.variables)):
                        self.variables[i].assign_add(cauchy_point[i])
                    cp_loss = self.loss()
                    # try newton point
                    for i in range(len(self.variables)):
                        self.variables[i].assign_add(p[i] - cauchy_point[i])
                    n_loss = self.loss()
                    # undo variable update
                    for i in range(len(self.variables)):
                        self.variables[i].assign_sub(p[i])

                    # use cauchy point if it is better than newton point
                    if cp_loss < n_loss:
                        p = cauchy_point
                    if self.verbose["globalization"]:
                        print("newton step inside TR")
                    on_boundary = False
                else:
                    if self.tr_type == "dogleg":
                        p = self.tr.dogleg_point(cauchy_point, p, cauchy_step_size, p_dot_p)
                        on_boundary = True
                    else:  # 2D subspace minimization
                        p = self.subspace_min_2D(cauchy_point, p)
                        return p
        return p, on_boundary

    def subspace_min_2D(self, p1, p2):
        """
        Solve the TR subproblem with 2D subspace minimization across two search directions
        :param p1: list(tf.Tensor): the first search direction, typically the negative gradient
        :param p2: list(tf.Tensor): the second search direction, typically the newton direction
        :return: list(tf.Tensor), bool: the step direction and whether it is on the boundary
        """
        s = np.vstack([flatten(p1), flatten(p2)]).T
        q, r = np.linalg.qr(s)
        try:
            r_inv = np.linalg.inv(r)
        except np.linalg.LinAlgError as e:
            print(r)
            raise e
        q1 = reshape(q[:, 0], p1)
        q2 = reshape(q[:, 1], p2)
        H_q1 = self.hessian_v_prod(q1)
        H_q2 = self.hessian_v_prod(q2)
        q1_H_q1 = multi_dim_dot(q1, H_q1)
        q1_H_q2 = multi_dim_dot(q1, H_q2)
        q2_H_q2 = multi_dim_dot(q2, H_q2)
        B = np.array([[q1_H_q1, q1_H_q2], [q1_H_q2, q2_H_q2]])
        g = flatten(self.g).dot(q)
        alpha_beta_prime, on_boundary = self.tr.subspace_2D_minimization(B, g)
        alpha, beta = r_inv @ alpha_beta_prime

        for i in range(len(p1)):
            p1[i] = alpha * p1[i] + beta * p2[i]

        return p1, on_boundary


    def minimize(self):
        """
        Perform one iteration to minimize the loss function with respect to the variables
        """
        # two additional sweeps are performed the first time we do a hessian vector product
        self.init_minimize_iter()

        if self.tr_type == "dogleg" or self.tr_type == "lowcost_subspace_2d":
             p, on_boundary = self.lowcost_tr_minimize()
        else:
            neg_grad = [-g_i for g_i in self.g]
            tol = self.forcing_term()
            p = self.cg_solve(neg_grad, tol)

            # Update variables based on globalization method
            if self.tr_type == "subspace_2d":
                # if newton direction equals gradient then no need to solve 2D subspace problem
                if equal(neg_grad, p):
                    p, on_boundary = self.tr.tr_step(p)
                else:
                    p, on_boundary = self.subspace_min_2D(neg_grad, p)

            else:
                p, on_boundary = self.tr.tr_step(p)

        if self.momentum_type == "subspace_2D":
            if self.m is not None and not equal(p, self.m):
                p, on_boundary = self.subspace_min_2D(p, self.m)

        self.tr_update(p, on_boundary)


        if self.verbose["loss_breakdown"]:
            self.print_loss()
        # Free GradientTape memory
        self.gradient_tape_reset(evaluate=False)



