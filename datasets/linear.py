import tensorflow as tf


def generate_linear_data(w, b, num_train=10000, num_test=1000):
    train_inputs = tf.random.normal(shape=[num_train])
    train_outputs = train_inputs * w + b + tf.random .normal(shape=[num_train]) * 0.1
    test_inputs = tf.random.normal(shape=[num_test])
    test_outputs = test_inputs * w + b + tf.random.normal(shape=[num_test]) * 0.1

    return (train_inputs, train_outputs), (test_inputs, test_outputs)
