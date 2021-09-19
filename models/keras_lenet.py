from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model

def lenet5(input_shape=(32,32, 1)):
    x_in = Input(input_shape)

    # Convolutional layer (32, 32, 1) -> (28, 28, 6) -> (14, 14, 6)
    x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1))(x_in)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    # Convolutional layer (14, 14, 6) -> (10, 10, 16) -> (5, 5, 16) -> (400)
    x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Dense
    x = Dense(units=120, activation='relu')(x)
    x = Dense(units=84, activation='relu')(x)
    # Output - do not apply softmax, this is handles by the loss function
    x = Dense(units=10, activation=None)(x)

    m = Model(inputs=x_in, outputs=x)
    m.summary()
    return m
