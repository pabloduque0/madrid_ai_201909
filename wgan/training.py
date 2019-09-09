from wgan import critic
from wgan import generator

discriminator_model = discriminator.create_model(img_shape)
discriminator_model.compile(optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99),
                            loss=metrics.wasserstein_loss)

generator_model = wasserstein_generator.create_model(noise_shape)
z = layers.Input(shape=noise_shape)
image = generator_model(z)

discriminator_model.trainable = False
valid = discriminator_model(image)

combined_model = models.Model(z, valid)
combined_model.compile(optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99),
                       loss=metrics.wasserstein_loss)




for i in range(n_iterations):
    for k in range(k_steps):
        idx = np.random.randint(0, images.shape[0], half_batch)
        batch_images = images[batch_idx]

        noise = np.random.normal(0, 1, (half_batch, *self.noise_shape))
        generated_imgs = self.generator.predict(noise)

        self.discriminator.trainable = True
        d_loss_real = self.discriminator.train_on_batch(batch_images,
                                                        -np.ones((half_batch, 1)))
        d_loss_fake = self.discriminator.train_on_batch(generated_imgs,
                                                        np.ones((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        for l in self.discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
            l.set_weights(weights)

    noise = np.random.normal(0, 1, (batch_size, *self.noise_shape))

    valid_y = np.array([-1] * batch_size)
    self.discriminator.trainable = False
    g_loss = self.combined.train_on_batch(noise, valid_y)