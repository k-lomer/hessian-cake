from optimizers.optimizer import Optimizer


class StochasticGradientDescent(Optimizer):
    """
    A Stochastic Gradient Descent Optimizer, uses the local gradient and a fixed learning rate
    """
    def __init__(self, learning_rate, vars, loss):
        """
        :param learning_rate: float: The step size for the method
        :param variables: list(tf.Variable): the trainable variables
        :param loss_func: NN_Loss: class representing objective function
        Note loss_func could be a class other than NN_Loss if it contains the required methods
        """
        super(StochasticGradientDescent, self).__init__(learning_rate, vars, loss)

    def minimize(self):
        """
        Perform one iteration to minimize the loss function with respect to the variables
        """
        grad = self.gradient()
        for i in range(len(grad)):
            self.variables[i].assign_sub(self.lr * grad[i])
