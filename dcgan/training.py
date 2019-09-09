from keras import layers
from keras import models
import numpy as np
from dcgan import discriminator
from dcgan import generator
from keras import optimizers
from keras import losses

discriminator_model = discriminator.create_model(img_shape)
discriminator_model.compile(optimizers.Adam(lr=0.00001),
                            loss=losses.binary_crossentropy)

generator_model = generator.create_model(noise_shape)

z = layers.Input(shape=noise_shape)
image = generator_model(z)

discriminator_model.trainable = False
valid = discriminator_model(image)

combined_model = models.Model(z, valid)
combined_model.compile(optimizers.Adam(lr=0.00001),
                       loss=losses.binary_crossentropy)


for i in range(n_iterations):

    for j in range(k_steps):

        idx = np.random.randint(0, images.shape[0], half_batch)
        batch_images = images[idx]
        noise = np.random.normal(0, 1, (half_batch, *noise_shape))
        generated_imgs = generator_model.predict(noise)

        discriminator_model.trainable = True
        d_loss_real = discriminator_model.train_on_batch(batch_images,
                                                        np.ones((half_batch, 1)))
        d_loss_fake = discriminator_model.train_on_batch(generated_imgs,
                                                        np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, *noise_shape))

    valid_y = np.array([1] * batch_size)
    discriminator_model.trainable = False

    g_loss = combined_model.train_on_batch(noise, valid_y)