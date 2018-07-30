import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import matplotlib.pyplot as plt
import numpy as np

import config
import util

mnist = input_data.read_data_sets(config.mnist_path, one_hot=True)
x_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def log(x):
    '''
    Sometimes discriminator outputs can reach values close to (or even slightly less than) zero due to numerical rounding.
    This just make sure that we exclude those values so that we don't up with NaNs during optimization
    '''
    return tf.log(tf.maximum(x, 1e-5))

with tf.variable_scope('D'):
    D_W1 = tf.Variable(xavier_init([x_dim + y_dim, config.hidden_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[1]))

    D_W2 = tf.Variable(xavier_init([config.hidden_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))

def discriminator(x, y):
    # Concatenate x and y
    inputs = tf.concat(axis=1, values=[x, y])

    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

with tf.variable_scope('G'):
    G_W1 = tf.Variable(xavier_init([config.input_dim + y_dim, config.hidden_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[config.hidden_dim]))

    G_W2 = tf.Variable(xavier_init([config.hidden_dim, x_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
    
def generator(z, y):
    # COncatenate z and y
    inputs = tf.concat(axis=1, values=[z, y])

    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

class CGAN(object):
    def __init__(self, config):
        self.config = config

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, y_dim])
        self.z = tf.placeholder(tf.float32, shape=[None, config.input_dim])

        self.G_sample = generator(self.z, self.y)

        self.D_real, self.D_logit_real = discriminator(self.x, self.y)
        self.D_fake, self.D_logit_fake = discriminator(self.G_sample, self.y)

        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_real, labels=tf.ones_like(self.D_logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.zeros_like(self.D_logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logit_fake, labels=tf.ones_like(self.D_logit_fake)))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.D_solver = tf.train.AdamOptimizer().minimize(self.D_loss, var_list=self.d_params)
        self.G_solver = tf.train.AdamOptimizer().minimize(self.G_loss, var_list=self.g_params)

    def train(self):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            if not os.path.exists('./out/'):
                os.makedirs('./out/')

            save_index = 0

            for iter in range(100000):
                
                batch_x, batch_y = mnist.train.next_batch(config.batch_size)
                batch_z = sample_Z(config.batch_size, config.input_dim)

                _, D_loss_cur = sess.run([self.D_solver, self.D_loss], feed_dict={self.x: batch_x, self.z: batch_z, self.y: batch_y})
                _, G_loss_cur = sess.run([self.G_solver, self.G_loss], feed_dict={self.z: batch_z, self.y: batch_y})

                if iter % 1000 == 0:
                    print("iter: ", iter)
                    print("D_loss: ", D_loss_cur)
                    print("G_loss: ", G_loss_cur)

                    samples_num = 16

                    z_sample = sample_Z(samples_num, config.input_dim)
                    y_sample = np.zeros(shape=[samples_num, y_dim])
                    y_sample[:, 7] = 1
                    
                    samples = sess.run(self.G_sample, feed_dict={self.z: z_sample, self.y: y_sample})

                    fig = util.plot(samples)
                    save_filename = './out/' + str(save_index).zfill(3)
                    save_index += 1
                    plt.savefig(save_filename, bbox_inches='tight')
                    plt.close()
    