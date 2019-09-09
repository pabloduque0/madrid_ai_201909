from keras import models
from keras import layers
import numpy as np

def create_model(input_shape, sub_start_shape=(6, 6, 128)):

    input_layer = layers.Input(shape=input_shape)

    conv_2 = layers.Dense(1024)(input_layer)
    lr_1 = layers.ReLU()(conv_2)
    dense_2 = layers.Dense(np.prod(list(sub_start_shape)))(lr_1)
    bn_1 = layers.BatchNormalization()(dense_2)
    lr_2 = layers.ReLU()(bn_1)
    reshaped = layers.Reshape(sub_start_shape, input_shape=(np.prod(list(sub_start_shape)),))(lr_2)
    bn_axis = -1
    conv_1 = layers.Conv2DTranspose(512, (5, 5), strides=2, padding='same')(reshaped)
    bn_2 = layers.BatchNormalization(axis=bn_axis)(conv_1)
    lr_3 = layers.ReLU()(bn_2)

    conv_2 = layers.Conv2DTranspose(256, (5, 5), strides=2, padding='same')(lr_3)
    bn_3 = layers.BatchNormalization(axis=bn_axis)(conv_2)
    lr_4 = layers.ReLU()(bn_3)

    padded = layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(lr_4)

    conv_3 = layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same')(padded)
    bn_5 = layers.BatchNormalization(axis=bn_axis)(conv_3)
    lr_6 = layers.ReLU()(bn_5)

    conv_4 = layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same')(lr_6)
    bn_6 = layers.BatchNormalization(axis=bn_axis)(conv_4)
    lr_7 = layers.ReLU()(bn_6)

    conv_5 = layers.Conv2DTranspose(2, (5, 5), strides=2, activation="tanh", padding='same')(lr_7)

    model = models.Model(inputs=input_layer, outputs=conv_5)
    model.summary()
    return model
