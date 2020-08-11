import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype(np.float32)
y_data = 3.0 * x_data + 1.0
print("x_data: ", x_data)
print("y_data: ", y_data)

# Define trainable variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

# Define graph operations
y = tf.multiply(W, x_data) + b

# Define loss
loss = tf.reduce_mean(tf.square(y - y_data))

# Define optimizer for training
train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)

# Define the operation that initializes variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    # Initialization
    sess.run(init)

    # Start training
    training_iters = 1000
    for step in range(training_iters):
        if step % 20 == 0 or (step + 1) == training_iters:
            print(step, sess.run(W), sess.run(b))

        _ = sess.run([train_optimizer])
