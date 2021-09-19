import tensorflow as tf
from optimizers.optimizer import Optimizer
from operations import multi_dim_dot

class Adam(Optimizer):
    """
    An Adam Optimizer, first order adaptive method using momentum
    """
    def __init__(self, learning_rate, vars, loss_func, b1=0.9, b2=0.999, eps=1e-8):
        """
        :param learning_rate: float: The step size for the method
        :param variables: list(tf.Variable): the trainable variables
        :param loss_func: NN_Loss: class representing objective function
        :param b1: float: method parameter for momentum, default recommended
        :param b2: float: method parameter for scaling, default recommended
        :param eps: float: method parameter for non-zero division, default recommended
        Note loss_func could be a class other than NN_Loss if it contains the required methods
        """
        super(Adam, self).__init__(learning_rate, vars, loss_func, regularization=0)

        self.b1 = b1
        self.b2 = b2
        self.eps = eps

        self.verbose = False

        self.t = 0

        self.m = []
        self.v = []
        for var in self.variables:
            self.m.append(tf.zeros_like(var))
            self.v.append(tf.zeros_like(var))

    def update_vars(self, vars):
        """
        Update the variables and ensure they are stored in a list
        Reset the adaptive stored values
        :param variables: tf.Variable or list(tf.Variable): the new variables
        """
        self.variables = list(vars)

        # reset parameters
        self.t = 0

        self.m = []
        self.v = []
        for var in self.variables:
            self.m.append(tf.zeros_like(var))
            self.v.append(tf.zeros_like(var))

    def minimize(self):
        """
        Perform one iteration to minimize the loss function with respect to the variables
        """
        self.t += 1
        grad = self.gradient()
        if self.verbose:
            g_dot_g = multi_dim_dot(grad, grad)
            print(f"gradient size: {g_dot_g:.6f}")
        alpha = self.lr * tf.sqrt(1 - self.b2 ** self.t) / (1 - self.b1**self.t)

        for i in range(len(grad)):

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad[i]
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * tf.square(grad[i])

            step = alpha * self.m[i] / (tf.sqrt(self.v[i]) + self.eps)
            self.variables[i].assign_sub(step)
