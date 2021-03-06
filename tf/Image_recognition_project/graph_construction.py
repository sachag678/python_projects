import tensorflow as tf

def classifier():
	"""This method creates a 3 layer convolutional-nn using max-pool layers, dropout, tanh activation function
	and AdamOptimizer

	params: None

	return: 

	:x: A placeholder which contains the 32x32x3 images

	:y_: A placeholder which contains the one hot encoded labels 

	:model: The CNN model

	:train_op: The function that will minimize the cross entropy of the predicted output 
	and the correct output using the AdamOptimizer

	:accuracy: The function that will calculate the accuracy using the perdicted output
	and correct output

	:keep_prob: The probabality used to determine how much to dropout during training and 
	testing
	"""
	
	x = tf.placeholder(tf.float32,[None,32,32,3])
	y_ = tf.placeholder(tf.float32,[None, 10])

	W_conv1 = weight_variable([5, 5, 3, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,32,32,3])

	h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([8 * 8 * 64, 2048])
	b_fc1 = bias_variable([2048])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([2048, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

	#train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#initialize model
	model = tf.global_variables_initializer()

	return x,y_,model,train_op,accuracy,keep_prob


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')