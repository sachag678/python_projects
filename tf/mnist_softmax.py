from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#setup
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#softmax
y = tf.nn.softmax(tf.matmul(x,W)+b)

#cross entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.matmul(x,W)+b))

#training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

#test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_accuracy = np.zeros([1000,1])
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	_, train_accuracy[i] = sess.run([train_step,accuracy], feed_dict={x:batch_xs,y_:batch_ys})
	#print(train_accuracy)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

plt.plot(train_accuracy)
plt.show()