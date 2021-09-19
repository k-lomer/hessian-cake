import tensorflow as tf
from operations import multi_dim_dot

class EisenstatWalker:
    """
    Eisenstat Walker Forcing Term
    Implementation of choice 2 from EW paper "Choosing the Forcing Terms in an Inexact Newton Method"
    Using default values as recommended in the paper
    For computational efficiency we use g^tg instead of g, some values are adjusted accordingly (mu, eta_max, eta_sg)
    thus we use a squared value of eta here and in the CG method although we use the variable name eta internally
    """
    def __init__(self, eta0=1e-1, sg=True):
        """
        :param eta0: float: the inital forcing term
        """
        self.eta = eta0**2
        self.g_cur = None
        self.g_prev = None
        self.gamma = 1.0
        self.mu = (1 + 5**0.5)/2
        self.eta_max = 0.9**2
        self.sg = sg
        self.do_update = True

    def update(self, g_dot_g):
        """
        update the value of the forcing term eta
        :param g_dot_g: float: the norm of the gradient squared
        :param sg: bool: whether to use safe guard with previous tolerance
        """
        if not self.do_update:
            self.do_update = True
            return

        if self.g_prev is None:
            # not yet initialize so use default
            self.g_prev = self.g_cur
            self.g_cur = g_dot_g
        else:
            self.g_prev = self.g_cur
            self.g_cur = g_dot_g
            eta_new = self.gamma * (self.g_cur / self.g_prev) ** self.mu

            # Safeguards
            if self.sg:
                eta_sg = self.gamma * self.eta ** self.mu
                if eta_sg > 0.01:
                    eta_new = max(eta_new, eta_sg)
            self.eta = min(eta_new, self.eta_max)
