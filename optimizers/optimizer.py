import tensorflow as tf
from operations import multi_dim_dot

class Optimizer:
    """
    Base Optimizer Class contains common optimizer methods
    Children of this class must override the minimize() method
    """

    def __init__(self, learning_rate, variables, loss_func, regularization=0, subsamples=None):
        """
        :param learning_rate: float: The step size for the method
        :param variables: list(tf.Variable): the trainable variables
        :param loss_func: NN_Loss: class representing objective function
        :param regularization: float: the size of the regularization parameter, 0 for no regularization
        :param subsamples: int: the number of subsamples when subsampling the loss
        Note loss_func could be a class other than NN_Loss if it contains the required methods
        """
        self.lr = tf.constant(learning_rate)
        self.variables = list(variables)
        self.loss_func = loss_func
        self.reg = regularization
        self.subsample = subsamples is not None and subsamples > 0
        if self.subsample:
            self.loss_func.subsamples = subsamples
        self.sweeps = 0

    def loss(self, subsample=False):
        """
        Evaluate the loss function
        :param subsample: bool - whether to subsample the loss function at it's internal subsample size
        :return: tf.Tensor: the value of the loss function
        """
        self.sweeps += 1
        return self.loss_func.subsampled_loss() if subsample else self.loss_func()

    def gradient(self):
        """
        Evaluate the gradient of the loss function
        :return: list(tf.Tensor): the gradient of the loss function in the same shape as the variables.
        """
        self.sweeps += 1
        with tf.GradientTape() as tape:
            loss_value = self.loss()

        return tape.gradient(loss_value, self.variables)

    def update_vars(self, variables):
        """
        Update the variables and ensure they are stored in a list
        :param variables: tf.Variable or list(tf.Variable): the new variables
        """
        self.variables = list(variables)

    def minimize(self):
        """
        Abstract method, should override in child classes
        Perform one iteration to minimize the loss function with respect to the variables
        """
        pass

    def print_loss(self):
        """
        Print the current training loss and regularization
        """
        loss = self.loss_func()
        reg = self.reg * multi_dim_dot(self.variables, self.variables) / 2
        print(f"NN loss: {loss.numpy():.4f}, regularization: {reg.numpy():.6f}")
