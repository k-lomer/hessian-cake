import tensorflow as tf

class Rosenbrock:
    def __init__(self, x):
        self.variables = [tf.Variable(x_i, dtype=tf.float32) for x_i in x]

    def __call__(self):
        x = self.variables
        return 10*(x[1] - x[0]**2)**2 + (1 - x[0])**2