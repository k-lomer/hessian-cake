import tensorflow as tf
from tensorflow.keras.preprocessing import sequence


def processed_imdb(max_features=20000, max_len=80):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)

    y_train = tf.dtypes.cast(y_train, tf.float32)
    y_test = tf.dtypes.cast(y_test, tf.float32)

    return (x_train, y_train), (x_test, y_test)

