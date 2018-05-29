import tensorflow as tf
from tensorflow import logging
import numpy as np
from cnnvae import CNNVAE
import utils


logging.set_verbosity(logging.INFO)
logging.info("read image data...")
# splitted_train_data, n_samples = utils.read_from_tfrecord()
splitted_train_data, n_samples = utils.read_from_jpg(partial=51200)
# n_samples = 25600
logging.info("read {} images".format(n_samples))
training_epochs = 1000
lr = 3e-7
latent_dim = 100
batch_size = 128

envvar = "#sample: {}, lr: {}, latent_dim: {}, batch_size: {}".format(n_samples, lr, latent_dim, batch_size)

logdir = '/tmp/vae-cnn/'

display_step = 10

vae = CNNVAE(lr=lr, latent_dim=latent_dim, logdir=logdir)
restored_global_step = vae.restore()

for epoch in range(restored_global_step, training_epochs):
    avg_cost = 0.
    avg_kl_divergence = 0.
    total_batch = int(n_samples / batch_size)

    for i in range(total_batch):
        batch_xs = splitted_train_data[np.random.randint(0, len(splitted_train_data))]
        cost, kl_divergence = vae.partial_fit(batch_xs)
        avg_cost += cost * batch_size / n_samples
        avg_kl_divergence += kl_divergence * batch_size / n_samples

    summary = vae.get_summary(batch_xs)
    vae.summary_writer.add_summary(summary, global_step=epoch)

    # Display logs per epoch step
    if epoch % display_step == 0:
        logging.info("Epoch: {}, Cost: {:.9f} KLD: {:.9f}"
                     .format(epoch + 1, avg_cost, avg_kl_divergence))

    vae.save(epoch=epoch)
# print("Total cost: " + str(vae.calc_total_cost(X_test)))
