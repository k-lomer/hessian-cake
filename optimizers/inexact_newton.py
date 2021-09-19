import numpy as np
import tensorflow as tf
from optimizers.optimizer import Optimizer
from operations import multi_dim_dot, norm
from optimizers.eisenstat_walker import EisenstatWalker


class InexactNewton(Optimizer):
    """
    Inexact Newton Optimizer
    Uses a CG solver to find the newton direction inexactly
    Uses Backtracking Linesearch for globalization
    """
    _default_params = {"regularization": 0.1, "cg_tol": "ew", "precondition": True, "subsamples": None,
                       "bt_iters": 20, "momentum": 0, "momentum_type": "cg_warmup",
                       "lanczos": True, "hv_reuse": True, "ew_sg": False}

    @classmethod
    def get_param(cls, params, key):
        """
        Get the value of a parameter or the default
        :param params: dict: input parameters, take the value from here if it exists
        :param key: string: the key to look up
        :return: parameter value
        """
        return params.get(key, cls._default_params[key])

    @classmethod
    def get_default_params(cls):
        """
        Default params getter
        :return: dict: the default parameters
        """
        return cls._default_params.copy()

    def __init__(self, learning_rate, variables, loss, params={}, verbose={}):
        """
        :param learning_rate: float: The step size for the method
        :param variables: list(tf.Variable): the trainable variables
        :param loss_func: NN_Loss: class representing objective function
        :param params: dict: the inexact newton parameters
        :param verbose: dict: the verbosity settings
        Note loss_func could be a class other than NN_Loss if it contains the required methods
        """
        reg = self.get_param(params, "regularization")
        subsamples = self.get_param(params, "subsamples")
        super(InexactNewton, self).__init__(learning_rate, variables, loss, reg, subsamples)

        self.cg_iters = 0
        self.iteration = 0
        self.lanczos_update = self.get_param(params, "lanczos")
        self.cg_tol = self.get_param(params, "cg_tol")
        if self.cg_tol == "ew":
            use_safeguard = self.get_param(params, "ew_sg")
            self.ew = EisenstatWalker(sg=use_safeguard)
        else:
            self.cg_tol = float(self.cg_tol)**2
        if self.get_param(params, "precondition"):
            self.cg_solve = self.cg_solve_preconditioned
            self.M = None
        else:
            self.cg_solve = self.cg_solve_standard

        self.hv_reuse = self.get_param(params, "hv_reuse")

        if self.hv_reuse:
            self.gradient_tape = tf.GradientTape(persistent=True)
            self.hessian_v_prod = self.hessian_v_prod_h_reuse
        else:
            self.hessian_v_prod = self.hessian_v_prod_full

        self.max_bt_iter = self.get_param(params, "bt_iters")

        self.momentum_type = self.get_param(params, "momentum_type")
        if self.momentum_type not in ["fixed", "trial", "subspace_2D", "cg_warmup"]:
            self.momentum_type = None
        else:
            self.momentum = self.get_param(params, "momentum")
            self.m = None

        self.verbose = {
            "globalization": verbose.get("globalization", False),
            "cg": verbose.get("cg", False),
            "grad_size": verbose.get("grad_size", False),
            "loss_breakdown": verbose.get("loss_breakdown", False),
            "forcing_term": verbose.get("forcing_term", False),
            "lanczos": verbose.get("lanczos", False)
        }

    def finite_diff_hessian_v_prod(self, v, delta):
        scaled_v = [delta * v_i for v_i in v]
        for i in range(len(self.variables)):
            self.variables[i].assign_sub(scaled_v[i])
        g1 = self.gradient()
        for i in range(len(self.variables)):
            self.variables[i].assign_add(2 * scaled_v[i])
        g2 = self.gradient()
        # return to before
        for i in range(len(self.variables)):
            self.variables[i].assign_sub(scaled_v[i])
        return [(g1[i] - g2[i]) / delta for i in range(len(g1))]



    def hessian_v_prod_full(self, v):
        """
        Evaluate the Hessian vector product, including regularization by two back-propagations
        :param v: list(tf.Tensor): the vector to multiply, should have the same shape as the variables
        :return: list(tf.Tensor): the hessian vector product
        """
        self.sweeps += 3
        with tf.GradientTape() as t1:
            with tf.GradientTape() as t2:
                loss_value = self.loss(self.subsample)
                if self.reg != 0:
                    loss_value += self.reg * multi_dim_dot(self.variables, self.variables) / 2
            grad = t2.gradient(loss_value, self.variables)

        return t1.gradient(grad, self.variables, output_gradients=v)

    def hessian_v_prod_h_reuse(self, v):
        """
        Evaluate the Hessian vector product, including regularization by two back-propagations
        :param v: list(tf.Tensor): the vector to multiply, should have the same shape as the variables
        :return: list(tf.Tensor): the hessian vector product
        """
        self.sweeps += 2
        #TODO make this a single operation
        self.gradient_tape._push_tape()
        v_hat = multi_dim_dot(v, self.gt_grad)
        self.gradient_tape._pop_tape()
        hv = self.gradient_tape.gradient(v_hat, self.variables)
        return hv

    def forcing_term(self):
        """
        Get the forcing term
        :return: float: forcing term
        Note that this is actually the square of the forcing term to save repeated calculations
        """
        if self.cg_tol == "ew":
            ft = self.ew.eta
        else:
            ft = self.cg_tol
        if self.verbose["forcing_term"]:
            print(f"forcing term {ft}")
        return ft

    def backtracking_linesearch(self, p, reduction_factor=0.8, c=0.01):
        """
        Perform backtracking line search to find a suitable step size
        Note this also updates the variables with the scaled step
        :param p: list(tf.Tensorflow): the step direction
        :param reduction_factor: float: how much to reduce the step size at each iteration
        :param c: float: the armijo constant
        """
        step_size = self.lr
        original_loss = self.loss()
        grad_dot_p = multi_dim_dot(self.g, p)
        original_vars = [tf.identity(v) for v in self.variables]
        k = 0

        for i in range(len(p)):
            self.variables[i].assign_add(step_size * p[i])

        while self.loss() > original_loss + c * step_size * grad_dot_p and k < self.max_bt_iter:
            step_size *= reduction_factor
            # update variables with new step_size
            for i in range(len(p)):
                self.variables[i].assign(original_vars[i] + step_size * p[i])
            k += 1

        if self.verbose["globalization"]:
            if k == self.max_bt_iter:
                print(f"max backtracking iterations, {k}")
            else:
                print(f"{k} backtracking iterations")

        if self.momentum_type is not None:
            self.momentum_update(p)

    def momentum_update(self, p):
        """
        Update the variables with momentum
        Assumes that the variables have already been updated in the step direction p
        :param p: list(tf.Tensorflow): step direction
        """
        if self.m is None:
            self.m = p
        elif self.momentum_type == "fixed":
            # add momentum
            for i in range(len(p)):
                self.variables[i].assign_add(self.momentum * self.m[i])
                self.m[i] = p[i] + self.momentum * self.m[i]
        elif self.momentum_type == "trial":
            p_loss = self.loss()
            beta = min(self.momentum, norm(self.gradient()), norm(p))
            m_diff = [v_i - m_i for v_i, m_i in zip(self.variables, self.m)]
            self.m = [tf.identity(v_i) for v_i in self.variables]
            for i in range(len(p)):
                self.variables[i].assign_add(beta * m_diff[i])
            m_loss = self.loss()
            if p_loss < m_loss:
                # undo momentum
                for i in range(len(p)):
                    self.variables[i].assign_sub(beta * m_diff[i])
        elif self.momentum_type == "cg_warmup" or self.momentum_type == "subspace_2D":
            self.m = p

    def satisfy_tol(self, p, tol, H_p=None):
        """
        Check whether the search direction p satisfies the EW condition for a given tolerance
        :param p: list(tf.Tensor): the search direction
        :param tol: float: the tolerance (squared for convenience)
        :param H_p: Hessian product with p, optional
        :return: bool: whether p satisfies the EW condition
        """
        if H_p is None:
            H_p = self.hessian_v_prod(p)
        r = [Hp_i + g_i for Hp_i, g_i in zip(H_p, self.g)]
        r_dot_r = multi_dim_dot(r, r)
        return r_dot_r / self.g_dot_g <= tol


    def cg_get_x0(self):
        """
        Initialize x for the Conjugate Gradient method
        Either the previous step direction or the negative gradient direction
        :return: list(tf.Tensor)
        """
        if self.momentum_type == "cg_warmup" and self.m is not None:
            return self.m
        else:
            return [-g_i for g_i in self.g]

    def cg_solve_standard(self, b, tol):
        """
        Solve the equation Ax = b with the Conjugate Gradient method
        Let A be the hessian
        Does not use preconditioning
        :param b: list(tf.Tensor): the vector b, we use the negative gradient
        :param tol: float: the tolerance to compute x to
        :return: list(Tensor): the final value of x
        """
        x = self.cg_get_x0()
        r = [Ax_i - b_i for Ax_i, b_i in zip(self.hessian_v_prod(x), b)]
        p = [-r_i for r_i in r]
        k = 0
        r_dot_r = multi_dim_dot(r, r)
        b_dot_b = multi_dim_dot(b, b)

        max_iter = max(10, int(0.1 * self.iteration))
        while k < max_iter and r_dot_r / b_dot_b > tol:
            Ap = self.hessian_v_prod(p)
            pT_A_p = multi_dim_dot(p, Ap)
            if pT_A_p < 0:
                if self.verbose["cg"]:
                    print(f"Negative curvature on cg iter {k}")
                break
            alpha = r_dot_r / pT_A_p
            for i in range(len(x)):
                x[i] += alpha * p[i]
                r[i] += alpha * Ap[i]

            new_r_dot_r = multi_dim_dot(r, r)
            beta = new_r_dot_r / r_dot_r
            p = [-r_i + beta * p_i for r_i, p_i in zip(r, p)]

            k += 1
            r_dot_r = new_r_dot_r
        if self.verbose["cg"]:
            print(k, "cg iters")
        return x

    def gen_preconditioner_M(self):
        """
        Generate M, the diagonal Fisher Information Preconditioner
        """
        lam = max(self.reg, 1e-5)
        self.M = [tf.math.pow(tf.add(lam, tf.math.square(g_i)), 0.75) for g_i in self.g]

    def precondition(self, r):
        """
        Apply a diagonal preconditioner stored in self.M to the vector r
        :param r: list(tf.Tensor)
        :return: list(tf.Tensor): M^-1 * r
        """
        y = []
        for i in range(len(r)):
            y.append(tf.math.truediv(r[i], self.M[i]))
        return y

    def cg_solve_preconditioned(self, b, tol):
        """
        Solve the equation Ax = b with the Preconditioned Conjugate Gradient method
        Let A be the hessian
        Generates and uses the preconditioner self.M
        :param b: list(tf.Tensor): the vector b, we use the negative gradient
        :param tol: float: the tolerance to compute x to
        :return: list(Tensor): the final value of x
        """
        k = 0
        self.gen_preconditioner_M()

        x = self.cg_get_x0()
        r = [Ax_i - b_i for Ax_i, b_i in zip(self.hessian_v_prod(x), b)]
        y = self.precondition(r)
        p = [-y_i for y_i in y]

        r_dot_r = multi_dim_dot(r, r)
        r_dot_y = multi_dim_dot(r, y)
        b_dot_b = multi_dim_dot(b, b)

        max_iter = max(10, int(0.1 * self.iteration))
        while k < max_iter and r_dot_r / b_dot_b > tol:
            Ap = self.hessian_v_prod(p)
            pT_A_p = multi_dim_dot(p, Ap)
            if pT_A_p < 0:
                if self.verbose["cg"]:
                    print(f"Negative curvature on cg iter {k}")
                break
            alpha = r_dot_y / pT_A_p
            for i in range(len(x)):
                x[i] += alpha * p[i]
                r[i] += alpha * Ap[i]

            y = self.precondition(r)

            new_r_dot_y = multi_dim_dot(r, y)
            beta = new_r_dot_y / r_dot_y
            p = [-y_i + beta * p_i for y_i, p_i in zip(y, p)]

            k += 1
            r_dot_y = new_r_dot_y
            r_dot_r = multi_dim_dot(r, r)
        if self.verbose["cg"]:
            print(k, "cg iters")
        self.cg_iters = k
        return x

    def gradient_tape_reset(self, evaluate=True):
        if self.hv_reuse:
            self.sweeps += 2
            self.gradient_tape._tape = None
            if evaluate:
                self.gradient_tape._push_tape()
                loss_value = self.loss(self.subsample)
                if self.reg != 0:
                    loss_value += self.reg * multi_dim_dot(self.variables, self.variables) / 2
                self.gt_grad = self.gradient_tape.gradient(loss_value, self.variables)
                self.gradient_tape._pop_tape()

    def init_minimize_iter(self):
        self.iteration += 1

        if self.lanczos_update:
            self.gradient_tape_reset()
            self.lanczos_reg()
            self.lanczos_update = False

        self.g = self.gradient()
        self.g_dot_g = multi_dim_dot(self.g, self.g)

        if self.verbose["grad_size"]:
            print(f"gradient size: {self.g_dot_g:.6f}")
        if self.cg_tol == "ew":
            self.ew.update(self.g_dot_g)

        self.gradient_tape_reset()

    def minimize(self):
        """
        Perform one iteration to minimize the loss function with respect to the variables
        """
        self.init_minimize_iter()

        neg_grad = [-g_i for g_i in self.g]
        tol = self.forcing_term()
        p = self.cg_solve(neg_grad, tol)

        # Find step size and update variables
        self.backtracking_linesearch(p)

        if self.verbose["loss_breakdown"]:
            self.print_loss()

        # Free GradientTape memory
        self.gradient_tape_reset(evaluate=False)

    def lanczos(self, k=10):
        """
        Applies Lanczos algorithm to approximate the eigenvalues of the hessian
        :param k: number of iterations to complete
        :return: ndarray: shape(k,) the approximation of the eigenvalues
        """
        v_prev = [tf.zeros_like(var_i) for var_i in self.variables]
        v = [tf.random.uniform(shape=v_i.shape) for v_i in v_prev]
        v_norm = tf.sqrt(multi_dim_dot(v, v))
        v_next = [v_i / v_norm for v_i in v]
        beta = np.zeros(k + 1)
        alpha = np.zeros(k)

        V = []

        for i in range(0, k):
            w = self.hessian_v_prod(v_next)
            alpha[i] = multi_dim_dot(w, v_next)
            for j in range(len(w)):
                w[j] = w[j] - alpha[i] * v_next[j] - beta[i] * v_prev[j]

            # Orthogonalize:
            for t in range(i):
                tmpa = multi_dim_dot(w, V[t])
                for j in range(len(w)):
                    w[j] -= tmpa * V[t][j]

            beta[i + 1] = tf.sqrt(multi_dim_dot(w, w))
            v_prev = v_next
            v_next = [w_j / beta[i + 1] for w_j in w]
            V.append(list(v_prev))

        # Create tridiag matrix with size (k X k)
        tridiag = np.diag(alpha) + np.diag(beta[1:k], 1) + np.diag(beta[1:k], -1)

        # Get eigenvalues of this tridiagonal matrix.
        # eigenvectors may be obtained by multiplying V with its eigenvectors.
        d = np.linalg.eigvalsh(tridiag)

        return d

    def lanczos_reg(self):
        """
        Use Lanczos to calculate regularization for the hessian
        The approximation for the minimum eigenvalue is going to be smaller so increase by a factor of 1.2
        :param k: the number of iterations to complete
        """
        prev_reg = self.reg
        self.reg = 0

        d = self.lanczos()
        min_evalue = min(d)
        #print(min_evalue)
        # if not semipositive definite
        if min_evalue < 0:
            self.reg = -min_evalue * 1.2
        self.reg = max(self.reg, prev_reg)
        if self.verbose["lanczos"]:
            print(f"lanczos regularization {self.reg}")






