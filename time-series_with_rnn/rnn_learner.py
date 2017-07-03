import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def learn():
	"""Trains a RNN to learn a sine wave"""
	n_steps = 20
	n_inputs = 1
	n_neurons = 100
	n_outputs = 1

	X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
	y = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

	basic_cell = tf.contrib.rnn.OutputProjectionWrapper(
		tf.contrib.rnn.BasicRNNCell(num_units = n_neurons, activation = tf.nn.relu), 
		output_size=n_outputs)

	outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype = tf.float32, swap_memory=True)

	loss = tf.reduce_mean(tf.square(outputs-y))
	optimizer = tf.train.AdamOptimizer(0.001)

	train_op = optimizer.minimize(loss)

	model = tf.global_variables_initializer()

	n_iterations = 1000
	batch_size = 50

	t_instance = np.linspace(12.2, 12.2 + 0.1 * (n_steps + 1), n_steps + 1)

	with tf.Session() as sess:
		model.run()
		for iteration in range(n_iterations):

			X_batch, y_batch = next_batch(batch_size, n_steps)
			sess.run(train_op, feed_dict = {X:X_batch, y: y_batch})
			if iteration % 100 ==0:
				mse = loss.eval(feed_dict = {X:X_batch, y: y_batch})
				print(iteration, "\tMSE: ", mse)

		X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
		y_pred = sess.run(outputs, feed_dict={X: X_new})

	plt.title("Testing the model", fontsize=14)
	plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
	plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
	plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10,label="prediction")
	plt.legend(loc="upper left")
	plt.show()


def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
	t_min, t_max = 0, 30
	resolution = 0.1
	t0 = np.random.rand(batch_size, 1)*(t_max - t_min - n_steps * resolution)
	Ts = t0 + np.arange(0., n_steps + 1) * resolution
	ys = time_series(Ts)
	return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

if __name__ == '__main__':
	learn()
