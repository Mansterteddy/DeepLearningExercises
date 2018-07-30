from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class Model:

	def __init__(self):
		
		self.datadir = "/Users/manster/Documents/Dataset/MNIST_data/"
		
	def read_mnist(self):

		print("Load dataset")

		# Load dataset
		mnist = input_data.read_data_sets(self.datadir, one_hot=True)
		
		# Check data shape
		print(mnist.train.images.shape)
		print(mnist.train.labels.shape)
		print(mnist.test.images.shape)
		print(mnist.test.labels.shape)

		return mnist

	def forward(self):
		# Should be implemented in child classes
		assert False

	def train(self):
		# Should be implemented in child classes
		assert False



class Softmax(Model):

	def __init__(self):
		Model.__init__(self)

	def weight_variable(self, shape):
		W = tf.truncated_normal(shape, stddev=0.1)
		#W = tf.Variable(tf.zeros([784, 10]))
		return tf.Variable(W)

	def bias_variable(self, shape):
		b = tf.constant(0.1, shape=shape)
		return tf.Variable(b)

	def forward(self, x):

		# Define network trainable parameters
		W = self.weight_variable([784, 10])
		b = self.bias_variable([10])
		#W = tf.Variable(tf.zeros([784, 10]))
		#b = tf.Variable(tf.zeros([10]))

		# Define graph operations
		y_ = tf.nn.softmax(tf.matmul(x, W) + b)

		return y_

	def train(self):

		print("Start training")
		mnist = self.read_mnist()
		print("Data reading done")

		# Construct placeholders as the input 
		x = tf.placeholder(tf.float32, [None, 784])
		y = tf.placeholder(tf.float32, [None, 10])

		y_ = self.forward(x)

		# Define loss -- cross entropy
		with tf.name_scope('loss'):
			cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_), reduction_indices=[1]))

		# Define optimizer for training
		with tf.name_scope('optimizer'):
			train_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

		# Define accuracy
		with tf.name_scope("accuracy"):
			correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))		

    	# Define the operation that initializes variables
		init = tf.global_variables_initializer()

		# Launch the graph
		with tf.Session() as sess:
			# Initialization
			sess.run(init)

			batch_size = 100
			training_iters = 10000

			for i in range(training_iters):
				# Load a batch of data
				batch_x, batch_y = mnist.train.next_batch(batch_size)

				# Feed data into placeholder, run optimizer
				_ = sess.run([train_optimizer], feed_dict={x: batch_x, y: batch_y})
				
				if i % 100 == 0:
					print("Training iterations: ", i)

					# Evaluate the trained model
					accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
					print("Testing accuracy: ", accuracy_val)

			# Evaluate final trained model
			accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
			print("Testing accuracy: ", accuracy_val)


class DeepNN(Model):

	def __init__(self):
		Model.__init__(self)

	def conv2d(self, x, W):
		"""conv2d returns a 2d convolution layer with full stride."""
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		"""max_pool_2x2 downsamples a feature map by 2X."""
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def weight_variable(self, shape):
		"""weight_variable generates a weight variable of a given shape."""
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		"""bias_variable generates a bias variable of a given shape."""
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)


	def forward(self, x):
		"""
		deepnn builds the graph for a deep net for classifying digits.

		Args:
			x: an input tensor

		Returns:
		A tuple (y, k)
		"""

		# Reshape to use within a convolutional neural net
		with tf.name_scope('reshape'):
			x_image = tf.reshape(x, [-1, 28, 28, 1])

		# First convolutioanl layer - maps one grapyscale image to 32 feature maps
		with tf.name_scope('conv1'):
			W_conv1 = self.weight_variable([5, 5, 1, 32])
			b_conv1 = self.bias_variable([32])
			h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

		# Pooling layer - downsample by 2X.
		with tf.name_scope('pool1'):
			h_pool1 = self.max_pool_2x2(h_conv1)

		# Second convolutional layer -- maps 32 feature maps to 64
		with tf.name_scope('conv2'):
			W_conv2 = self.weight_variable([5, 5, 32, 64])
			b_conv2 = self.bias_variable([64])
			h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

		# Second pooling layer
		with tf.name_scope('pool2'):
			h_pool2 = self.max_pool_2x2(h_conv2)

		# Fully connected layer 1 -- after 2 round of downsapling, our 28x28 image 
		# is down to 7x7x64 feature maps -- maps this to 1024 features
		with tf.name_scope('fc1'):
			W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
			b_fc1 = self.bias_variable([1024])

			h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
			h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Dropout - controls the complexity of the model, prevents co-adaptation of features.
		with tf.name_scope('dropout'):
			keep_prob = tf.placeholder(tf.float32)
			h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# Map the 1024 features to 10 classes, one for each digit
		with tf.name_scope('fc2'):
			W_fc2 = self.weight_variable([1024, 10])
			b_fc2 = self.bias_variable([10])

			y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		return y_conv, keep_prob


	def train(self):
		
		print("Start training")
		mnist = self.read_mnist()
		print("Data reading done")

		# Construct placeholders as the input 
		x = tf.placeholder(tf.float32, [None, 784])
		y = tf.placeholder(tf.float32, [None, 10])

		y_, keep_prob = self.forward(x)

		# Define loss -- cross entropy
		with tf.name_scope('loss'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)
			cross_entropy = tf.reduce_mean(cross_entropy)

		# Define optimizer for training
		with tf.name_scope('optimizer'):
			train_optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

		# Define accuracy
		with tf.name_scope("accuracy"):
			correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
			correct_prediction = tf.cast(correct_prediction, tf.float32)
			accuracy = tf.reduce_mean(correct_prediction)	

    	# Define the operation that initializes variables
		init = tf.global_variables_initializer()

		# Launch the graph
		with tf.Session() as sess:
			# Initialization
			sess.run(init)

			batch_size = 50
			training_iters = 20000

			for i in range(training_iters):
				# Load a batch of data
				batch = mnist.train.next_batch(batch_size)

				# Feed data into placeholder, run optimizer
				_ = sess.run([train_optimizer], feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
				
				if i % 100 == 0:
					print("Training iterations: ", i)
					
					#Train accuracy
					train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
					print("Training accuracy: ", train_accuracy)

					# Evaluate the trained model
					accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
					print("Testing accuracy: ", accuracy_val)

			# Evaluate final trained model
			accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
			print("Testing accuracy: ", accuracy_val)

