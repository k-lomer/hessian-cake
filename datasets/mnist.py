import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

def processed_mnist(image_dim=32):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # one hot encode labels
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)



    # apply padding
    load_dim = x_train.shape[1]
    if load_dim != image_dim:
        p = (image_dim - load_dim) // 2
        assert image_dim == load_dim + 2*p

        x_train = np.pad(x_train, ((0, 0), (p, p), (p, p)), 'constant')
        x_test = np.pad(x_test, ((0, 0), (p, p), (p, p)), 'constant')

    # add 4th dimension (needed for CNNs)
    x_train = x_train.reshape((-1, image_dim, image_dim, 1))
    x_test = x_test.reshape((-1, image_dim, image_dim, 1))

    # Normalize to range [0,1]
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)

