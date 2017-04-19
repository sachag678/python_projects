"""Sacha Gunaratne May 2017 v1.0"""
import tensorflow as tf
import numpy as np

def logistic_classifier(learning_rate):
	"""Single layer neural network or logisitc regression model for classification - able to handle more than 2 classes due to the softmax function 
	   which is the generalized version of the logistic function
	"""
	#setup of logistic regression function
	feature_len = 561
	output_len = 6

	x = tf.placeholder(tf.float32,[None,feature_len])
	#W = tf.Variable(tf.zeros([561,6]))
	W = tf.Variable(tf.random_normal([feature_len,output_len],mean =0,stddev=2/(feature_len+output_len)))
	b = tf.Variable(tf.random_normal([output_len],mean =0,stddev=2/(feature_len+output_len)))

	#output of nn
	y = tf.nn.softmax(tf.matmul(x,W)+b)

	#correct labels and cost function
	y_ = tf.placeholder(tf.float32,[None, output_len])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=tf.matmul(x,W)+b))

	#initialize model
	model = tf.global_variables_initializer()

	#train
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	
	#accuracy
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return (x,y_,model,train_op,accuracy)

def knn(num_neighbours):
	"""Uses a KNN algorithm to calculate the k closest neigbours toa given data point and then uses a majority
	"""
	#input_data
	feature_len = 561
	output_len = 6

	xtrain = tf.placeholder(tf.float32,[None,feature_len])
	ytrain = tf.placeholder(tf.float32,[None, output_len])
	xtest = tf.placeholder(tf.float32, [feature_len])
	ytest = tf.placeholder(tf.float32, [output_len])

	#euclidean dist
	distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(xtrain, xtest)), reduction_indices=1)))

	#indices and values of top 5 k
	_, indices = tf.nn.top_k(distance, k=num_neighbours, sorted = False)

	#initialize model
	model = tf.global_variables_initializer()

	#get the top K neighbours labels and distances
	nearest_neighbors = [tf.argmax(ytrain[indices[i]], 0) for i in range(num_neighbours)]
	nearest_neighbors_dist = [tf.argmax(distance[indices[i]], 0) for i in range(num_neighbours)]

	#make it a tensor
	neighbors_tensor = tf.stack(nearest_neighbors)
	nn_dist_tensor = tf.stack(nearest_neighbors_dist)

	#gets the unique values and the counts
	uniq, idx, count = tf.unique_with_counts(neighbors_tensor)

	#somehow update the count variable with weights in the following manner
	#count[idx]= count[idx]/(nn_dist_tensor[idx])**2

	#gets the index of the max count
	train_op = tf.slice(uniq, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

	#accuracy
	accuracy = tf.cast(tf.equal(train_op, tf.argmax(ytest,0)),'float32')

	return (xtrain,ytrain,xtest,ytest,accuracy,model,train_op)


def two_layer_net(learning_rate):
	"""Two? layer feed forward nerual network with one hidden layer, one input layer and one output layer"""
	#initial input and weights and bias
	#best hidden layer num is what?
	feature_len = 561
	output_len = 6
	num_hidden_dim = 15
	
	x = tf.placeholder(tf.float32,[None,feature_len])
	W1 = tf.Variable(tf.random_normal([feature_len,num_hidden_dim],mean =0,stddev=2/(feature_len+num_hidden_dim)))
	b1 = tf.Variable(tf.random_normal([num_hidden_dim],mean =0,stddev=2/(feature_len+num_hidden_dim)))

	#output to hidden layer
	z1 = tf.matmul(x,W1)+b1

	#activation function from the hidden node - can use tanh, softmaz, relu
	a = tf.nn.tanh(z1)

	#output from hidden layer using new weights and biases
	W2 = tf.Variable(tf.random_normal([num_hidden_dim,output_len],mean =0,stddev=2/(num_hidden_dim+output_len)))
	b2 = tf.Variable(tf.random_normal([output_len],mean =0,stddev=2/(num_hidden_dim+output_len)))
	z2 = tf.matmul(a,W2)+b2

	#output of nn
	y = tf.nn.softmax(z2)

	#correct labels and cost function
	y_ = tf.placeholder(tf.float32,[None, output_len])

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=z2))

	#initialize model
	model = tf.global_variables_initializer()

	#train
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
	
	#accuracy
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	return (x,y_,model,train_op,accuracy)



