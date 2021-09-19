from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.models import Model


def lstm(input_shape, max_features=20000):
    x_in = Input(input_shape)

    x = Embedding(max_features, 64)(x_in)
    x = LSTM(100)(x)
    x = Dense(1, activation='sigmoid')(x)

    m = Model(inputs=x_in, outputs=x)
    m.summary()
    return m