import tensorflow as tf
import numpy as np


def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CNNVAE(object):
    def __init__(self, lr=1e-4, latent_dim=100, logdir='/tmp/cnnvae/'):
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.lr = lr
        self.latent_dim = latent_dim
        self.logdir = logdir
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            net = tf.contrib.layers.conv2d(self.x,
                                           128,
                                           [5, 5],
                                           (2, 2),
                                           padding='SAME',
                                           activation_fn=lrelu,
                                           normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d(net,
                                           256,
                                           [5, 5],
                                           (2, 2),
                                           padding='SAME',
                                           activation_fn=lrelu,
                                           normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d(net,
                                           512,
                                           [5, 5],
                                           (2, 2),
                                           padding='SAME',
                                           activation_fn=lrelu,
                                           normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d(net,
                                           1024,
                                           [5, 5],
                                           (2, 2),
                                           padding='SAME',
                                           activation_fn=lrelu,
                                           normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.flatten(net)

            net = tf.contrib.layers.fully_connected(net,
                                                    2 * latent_dim,
                                                    activation_fn=tf.nn.tanh)
            z_mean = net[:, :latent_dim]
            z_sigma = tf.nn.softplus(net[:, latent_dim:])
            self.z = tf.distributions.Normal(loc=z_mean, scale=z_sigma)

        assert self.z.reparameterization_type == tf.distributions.FULLY_REPARAMETERIZED

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            net = tf.contrib.layers.fully_connected(self.z.sample(),
                                                    4 * 4 * 1024,
                                                    activation_fn=None,
                                                    normalizer_fn=tf.contrib.layers.batch_norm)
            net = tf.reshape(net, [-1, 4, 4, 1024])

            net = tf.contrib.layers.conv2d_transpose(net,
                                                     512,
                                                     [5, 5],
                                                     (2, 2),
                                                     padding='SAME',
                                                     activation_fn=lrelu,
                                                     normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d_transpose(net,
                                                     256,
                                                     [5, 5],
                                                     (2, 2),
                                                     padding='SAME',
                                                     activation_fn=lrelu,
                                                     normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d_transpose(net,
                                                     128,
                                                     [5, 5],
                                                     (2, 2),
                                                     padding='SAME',
                                                     activation_fn=lrelu,
                                                     normalizer_fn=tf.contrib.layers.batch_norm)

            net = tf.contrib.layers.conv2d_transpose(net,
                                                     3,
                                                     [5, 5],
                                                     (2, 2),
                                                     padding='SAME',
                                                     activation_fn=tf.nn.sigmoid)

            self.reconstruction = net

            self.cost = tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
            normdist = tf.distributions.Normal(loc=np.zeros(latent_dim, dtype=np.float32),
                                               scale=np.ones(latent_dim, dtype=np.float32))
            self.KL_divergence = tf.reduce_mean(tf.distributions.kl_divergence(self.z, normdist))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = optimizer.minimize(self.cost + self.KL_divergence)

        self.sess.run(tf.global_variables_initializer())

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.cost)
            tf.summary.scalar('KL_divergence', self.KL_divergence)
            tf.summary.image('original_image', self.x)
            # tf.summary.tensor_summary('latent_vector', self.z)
            tf.summary.image('reconstructed_image', self.reconstruction)
            # tf.summary.image('diff_image', tf.subtract(self.reconstruction, self.x)
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

        self.saver = tf.train.Saver()

    def resize_image(self, x):
        return self.sess.run(self.x, feed_dict={self.x: x})

    def encode(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x})

    def decode(self, z):
        return self.sess.run(self.reconstruction, feed_dict={self.z: z})

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x})

    def partial_fit(self, x):
        cost, _, kl_divergence = self.sess.run(
            (self.cost, self.optimizer, self.KL_divergence),
            feed_dict={self.x: x})
        return cost, kl_divergence

    def get_summary(self, x):
        summary = self.sess.run(
            (self.merged_summary_op), feed_dict={self.x: x}
        )
        return summary

    def save(self, epoch):
        self.saver.save(self.sess, '{}vae-cnn'.format(self.logdir), global_step=epoch + 1)

    def restore(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.logdir)
        global_step = 0
        if latest_checkpoint:
            self.saver.restore(self.sess, latest_checkpoint)
            global_step = int(latest_checkpoint.split("/")[-1].split("-")[-1])
        return global_step
