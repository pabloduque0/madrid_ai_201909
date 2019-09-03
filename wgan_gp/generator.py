from keras import models
from keras import layers


def create_model(input_shape):
    input_layer = layers.Input(input_shape)
    dense_1 = layers.Dense(input_shape * 4 * 4, activation="relu")(input_layer)
    reshaped = layers.Reshape((input_shape, 4, 4))(dense_1)
    res_out1 = residual_block(reshaped, (3, 3))
    res_out2 = residual_block(res_out1, (3, 3))
    res_out3 = residual_block(res_out2, (3, 3))

    last_conv = layers.Conv2D(2, (3, 3), activation="tanh")(res_out3)

    model = models.Model(inputs=input_layer, outputs=last_conv)
    model.summary()

    return model


def residual_block(input_layer, kernel_size):
    conv1 = layers.Conv2D(128, kernel_size)(input_layer)
    up_samp = layers.UpSampling2D((2, 2), interpolation="nearest")(conv1)
    conv2 = layers.Conv2D(128, kernel_size)(up_samp)
    return conv2
