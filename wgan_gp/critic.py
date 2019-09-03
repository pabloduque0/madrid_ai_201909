from keras import layers
from keras import models


def create_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    res_out_1 = residual_block(input_layer, 3, True)
    res_out_2 = residual_block(res_out_1, 3, True)
    res_out_3 = residual_block(res_out_2, 3, False)
    res_out_4 = residual_block(res_out_3, 3, True)
    flatten = layers.Flatten()(res_out_4)
    dense_1 = layers.Dense(1, kernel_initializer='he_normal')(flatten)

    model = models.Model(inputs=input_layer, outputs=dense_1)
    model.summary()

    return model


def residual_block(input_layer, k_sizes, down_sample):

    conv_1 = layers.Convolution2D(128, k_sizes, padding='same', activation="relu")(input_layer)
    conv_2 = layers.Convolution2D(128, k_sizes, padding='same', activation="relu")(conv_1)
    res = layers.Add()([conv_2, input_layer])
    res = layers.Activation("relu")(res)
    if down_sample:
        res = layers.AveragePooling2D(pool_size=(2, 2))(res)

    return res

