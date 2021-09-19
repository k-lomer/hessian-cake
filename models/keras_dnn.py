from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model

def dnn(input_shape=(28,28, 1), hidden_layer_dims=[64], output_size=10):
    x_in = Input(input_shape)
    x = Flatten()(x_in)

    # Hidden dense layers
    for layer_dim in hidden_layer_dims:
        x = Dense(units=layer_dim, activation='relu')(x)

    # Output - do not apply softmax, this is handles by the loss function
    x = Dense(units=output_size, activation=None)(x)

    m = Model(inputs=x_in, outputs=x)
    m.summary()
    return m
