import numpy as np
import tensorflow as tf

import util

def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))

        return tf.matmul(input, w) + b

def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        print("Activate minibatch")
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3

def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([input, minibatch_features], 1)

def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=step, var_list=var_list)
    return optimizer

def log(x):
    '''
    Sometimes discriminator outputs can reach vakyes close to (or even slightly less than) zero due to numerical rounding.
    This just make sure that we exclude those values so that we don't up with NaNs during optimization
    '''
    return tf.log(tf.maximum(x, 1e-5))

class GAN(object):
    def __init__(self, config):
        # This defines the generator network - it takes samples from a noise distribution as input, and passes them thourgh an MLP.
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(config.batch_size, 1))
            self.G = generator(self.z, config.hidden_size)

        '''
        The discriminator tries to tell the difference between samples from the true data distribution (self.x) and the generated samples (self.z).
        '''
        self.x = tf.placeholder(tf.float32, shape=(config.batch_size, 1))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x, config.hidden_size, config.minibatch)

        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, config.hidden_size, config.minibatch)

        # Define loss for discriminator and generator networks
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

def train(model, data, gen, config):

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(config.num_steps + 1):
            # update discriminator
            x = data.sample(config.batch_size)
            z = gen.sample(config.batch_size)
            loss_d, _ = session.run([model.loss_d, model.opt_d], {model.x: np.reshape(x, (config.batch_size, 1)), model.z: np.reshape(z, (config.batch_size, 1))})

            # update generator
            z = gen.sample(config.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {model.z: np.reshape(z, (config.batch_size, 1))})

            if step % config.log_every == 0:
                log_info = "step " + str(step) + " loss_d: " + str(loss_d) + " loss_g: " + str(loss_g) 
                print(log_info)

        samps = util.samples(model, session, data, gen.range, config.batch_size)
        util.plot_distributions(samps, gen.range)
