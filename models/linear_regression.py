import tensorflow as tf


class LinearRegression:
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([1]))
        self.b = tf.Variable(tf.random.normal([1]))
        self.trainable_variables = [self.w, self.b]

    def __call__(self, x):
        return self.w * x + self.b
