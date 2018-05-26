import os
from glob import glob
import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import imageio
from IPython.display import display
from cnnvae import CNNVAE
from tqdm import tqdm


dataset = glob(os.path.join("./", "data", "*.jpg"))  # 202599

n_samples = len(dataset)
training_epochs = 1200
batch_size = 128
display_step = 10

X_train_data = dataset[:n_samples]


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


def min_max_scale(d):
    scaler = prep.MinMaxScaler()
    train_shape = d.shape
    d = d.reshape((train_shape[0], -1))
    scaler.fit(d)
    d = scaler.transform(d)
    return d.reshape(train_shape)


print('read image data...')
X_train = [imageio.imread(path) for path in tqdm(X_train_data)]
X_train = min_max_scale(np.array(X_train).astype(np.float32))

logdir = '/tmp/cnnvae/'
vae = CNNVAE(lr=3e-6, logdir=logdir)
saver = tf.train.Saver()

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = vae.partial_fit(batch_xs, epoch)
        avg_cost += cost / n_samples

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch: {}, Cost: {:.9f}".format(epoch + 1, avg_cost))
        saver.save(vae.sess, '{}cnnvae'.format(logdir), global_step=epoch)
# print("Total cost: " + str(vae.calc_total_cost(X_test)))
