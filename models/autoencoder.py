from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model

def autoencoder(input_shape, n_filters, filter_sizes):
    filter_count = len(n_filters)
    # same number of filters as filter sizes
    assert filter_count == len(filter_sizes)

    x_in = Input(input_shape)
    x = x_in

    # encoder layers
    for i in range(filter_count):
        x = Conv2D(filters=n_filters[i], kernel_size=filter_sizes[i], strides=(2,2), padding='same', activation='softmax')(x)

    # reverse filters for decoding
    n_filters.reverse()
    filter_sizes.reverse()

    # decoder layers
    for i in range(filter_count - 1):
        x = Conv2DTranspose(filters=n_filters[i+1], kernel_size=filter_sizes[i], strides=(2,2), padding='same', activation='softmax')(x)
    # last layer uses identity activation
    x = Conv2DTranspose(filters=input_shape[2], kernel_size=filter_sizes[-1], strides=(2,2), padding='same', activation='linear')(x)

    m = Model(inputs=x_in, outputs=x)
    m.summary()
    return m