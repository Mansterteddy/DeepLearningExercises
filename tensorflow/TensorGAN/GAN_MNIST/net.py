import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

import config
import util

def log(x):
    '''
    Sometimes discriminator outputs can reach values close to (or even slightly less than) zero due to numerical rounding.
    This just make sure that we exclude those values so that we don't up with NaNs during optimization
    '''
    return tf.log(tf.maximum(x, 1e-5))

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

with tf.variable_scope('D'):
    D_W1 = tf.Variable(xavier_init([784, 128]))
    D_b1 = tf.Variable(tf.zeros(shape=[128]))

    D_W2 = tf.Variable(xavier_init([128, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

with tf.variable_scope('G'):
    G_W1 = tf.Variable(xavier_init([100, 128]))
    G_b1 = tf.Variable(tf.zeros(shape=[128]))

    G_W2 = tf.Variable(xavier_init([128, 784]))
    G_b2 = tf.Variable(tf.zeros(shape=[784]))

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

class GAN(object):
    def __init__(self, config):
        
        self.config = config

        self.Z = tf.placeholder(tf.float32, shape=[None, 100])
        self.G_sample = generator(self.Z)

        self.X = tf.placeholder(tf.float32, shape=[None, 784])
        self.D_real, self.D_logit_real = discriminator(self.X)
        self.D_fake, self.D_logit_fake = discriminator(self.G_sample)

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.D_loss = -tf.reduce_mean(log(self.D_real) + log(1.-self.D_fake))
        self.G_loss = -tf.reduce_mean(log(self.D_fake))

        self.D_optim = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.d_params)
        self.G_optim = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.g_params)

    def train(self):

        mnist = input_data.read_data_sets(self.config.mnist_path, one_hot=True)
        batch_size = self.config.batch_size
        input_dim = self.config.input_dim

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            save_index = 0

            for iter in range(100000):

                batch_X, _ = mnist.train.next_batch(batch_size)
                batch_Z_1 = sample_Z(batch_size, input_dim)
                batch_Z_2 = sample_Z(batch_size, input_dim)

                _, D_loss_curr = sess.run([self.D_optim, self.D_loss], feed_dict={self.X: batch_X, self.Z: batch_Z_1})
                _, G_loss_curr = sess.run([self.G_optim, self.G_loss], feed_dict={self.Z: batch_Z_2})

                if iter % 1000 == 0:
                    print("iter", iter)
                    print("D Loss: ", D_loss_curr)
                    print("G Loss: ", G_loss_curr)

                    test_Z = sample_Z(16, input_dim)
                    samples = sess.run(self.G_sample, feed_dict={self.Z: test_Z})

                    fig = util.plot(samples)
                    save_filename = './out/' + str(save_index).zfill(3)
                    save_index += 1
                    plt.savefig(save_filename, bbox_inches='tight')
                    plt.close()
    