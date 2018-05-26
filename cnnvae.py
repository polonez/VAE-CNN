import tensorflow as tf


def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class CNNVAE(object):
    def __init__(self, lr=1e-4, logdir='/tmp/cnnvae/'):
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.lr = lr
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.conv2d(self.x, 64, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 128, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 256, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.layers.conv2d(net, 256, [5, 5], (2, 2), padding='SAME',
                                   kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net, training=True)
            net = lrelu(net)

            net = tf.contrib.layers.flatten(net)
            self.z = tf.contrib.layers.fully_connected(net,
                                                       100,
                                                       activation_fn=tf.nn.tanh,
                                                       weights_regularizer=tf.contrib.layers.l2_regularizer(1e-8))

        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            net = tf.contrib.layers.fully_connected(self.z,
                                                    4 * 4 * 256,
                                                    activation_fn=None)
            net = tf.reshape(net, [-1, 4, 4, 256])
            net = tf.layers.batch_normalization(net)

            net = tf.layers.conv2d_transpose(net, 256, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))
#                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-8))
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 128, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 64, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net)
            net = tf.nn.relu(net)

            net = tf.layers.conv2d_transpose(net, 3, [5, 5], (2, 2), 'SAME',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.05))
            net = tf.layers.batch_normalization(net)
            net = tf.nn.sigmoid(net)

            self.reconstruction = net

            self.cost = tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.optimizer = optimizer.minimize(self.cost)

        self.sess.run(tf.global_variables_initializer())

        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.cost)
            tf.summary.image('original_image', self.x)
            tf.summary.image('reconstructed_image', self.reconstruction)
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(
                logdir,
                graph=tf.get_default_graph())

    def resize_image(self, x):
        return self.sess.run(self.x, feed_dict={self.x: x})

    def encode(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x})

    def decode(self, z):
        return self.sess.run(self.reconstruction, feed_dict={self.z: z})

    def reconstruct(self, x):
        return self.sess.run(self.reconstruction, feed_dict={self.x: x})

    def partial_fit(self, x, epoch):
        cost, _, summary = self.sess.run(
            (self.cost, self.optimizer, self.merged_summary_op),
            feed_dict={self.x: x})
        self.summary_writer.add_summary(summary, global_step=epoch)
        return cost
