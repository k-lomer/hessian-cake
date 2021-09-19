import tensorflow as tf
import numpy as np


def multi_dim_dot(x, y):
    """
    Compute x^T * y, the dot product when the values are stored in lists of tensors
    x and y must have the same shape
    :param x: list(tf.Tensor)
    :param y: list(tf.Tensor)
    :return: tf.Tensor: the dot product of x and y
    """
    return tf.reduce_sum(input_tensor=[tf.reduce_sum(input_tensor=tf.math.multiply(x_i, y_i)) for x_i, y_i in zip(x, y)])

def norm(x):
    """
    compute the l2 norm of x
    :param x: list(tf.Tensor)
    :return: tf.Tensor l2-norm of x
    """
    return tf.sqrt(multi_dim_dot(x, x))

def flatten(x):
    """
    Flatten a list of tensors into a numpy vector
    :param x: list(tf.Tensor)
    :return: ndarray
    """
    return tf.concat([tf.reshape(x_i, [-1]) for x_i in x], 0).numpy()

def reshape(v, desired):
    """
    Reshape a ndarray into a list of tensors
    :param v: ndarray
    :param desired: list(tf.Tensor): a tensor with the desired shape
    :return: list(tf.Tensor): a tensor with the shape of desired and the values of v
    """
    reshaped = []
    count = 0
    for x_i in desired:
        shape = x_i.shape
        size = tf.size(input=x_i)
        reshaped.append(tf.constant(v[count: count + size], shape=shape))
        count += size
    return reshaped



def equal(x, y):
    for i in range(len(x)):
        if not tf.reduce_all(input_tensor=tf.equal(x[i], y[i])):
            return False
    return True


