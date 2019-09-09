from keras import models
from keras import layers

def create_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    conv_1 = layers.Convolution2D(64, (5, 5), padding='same', input_shape=input_shape)(input_layer)
    lr_1 = layers.LeakyReLU()(conv_1)
    conv_2 = layers.Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2])(lr_1)
    bn_1 = layers.BatchNormalization()(conv_2)
    lr_2 = layers.LeakyReLU()(bn_1)
    conv_3 = layers.Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2])(lr_2)
    bn_2 = layers.BatchNormalization()(conv_3)
    lr_3 = layers.LeakyReLU()(bn_2)
    conv_4 = layers.Convolution2D(64, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2])(lr_3)
    bn_3 = layers.BatchNormalization()(conv_4)
    lr_4 = layers.LeakyReLU()(bn_3)
    flatten = layers.Flatten()(lr_4)
    dense_1 = layers.Dense(64, kernel_initializer='he_normal')(flatten)
    lr_5 = layers.LeakyReLU()(dense_1)
    dense_2 = layers.Dense(1, kernel_initializer='he_normal')(lr_5)

    model = models.Model(inputs=input_layer, outputs=dense_2)
    model.summary()

    return model

