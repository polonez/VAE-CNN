import os
from glob import glob
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow import logging
import numpy as np
import imageio
from cnnvae import CNNVAE
from tqdm import tqdm


def min_max_scale(d):
    scaler = prep.MinMaxScaler()
    train_shape = d.shape
    d = d.reshape((train_shape[0], -1))
    scaler.fit(d)
    d = scaler.transform(d)
    return d.reshape(train_shape)


dataset = glob(os.path.join("./", "data", "*.jpg"))  # 202599

n_samples = len(dataset)
X_train_data = dataset[:n_samples]
training_epochs = 1200
batch_size = 128
display_step = 10


logging.set_verbosity(logging.INFO)
logging.info("read image data...")
X_train = [imageio.imread(path) for path in tqdm(X_train_data)]
X_train = min_max_scale(np.array(X_train).astype(np.float32))
splitted_train_data = np.array_split(X_train, n_samples // batch_size)

logdir = '/tmp/cnnae/'
vae = CNNVAE(lr=3e-6, logdir=logdir)
saver = tf.train.Saver()

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = splitted_train_data[
            np.random.randint(0, len(splitted_train_data))]
        cost = vae.partial_fit(batch_xs, epoch)
        avg_cost += cost / n_samples

    # Display logs per epoch step
    if epoch % display_step == 0:
        logging.info("Epoch: {}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        saver.save(vae.sess, '{}cnnae'.format(logdir), global_step=epoch)
# print("Total cost: " + str(vae.calc_total_cost(X_test)))
