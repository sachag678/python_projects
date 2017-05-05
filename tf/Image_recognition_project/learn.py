"""Sacha Gunaratne May 2017 v1.0"""
import tensorflow as tf
import numpy as np
import random

from graph_construction import classifier
from data_preprocessing import load_data, one_hot_encode_labels, normalize
import matplotlib.pyplot as plt

def train_model(num_epochs, batch_size, learning_rate):
	"""takes a some parameters, trains a con nueral net model,
	   calculates accuracy during training and prints it out. Prints out
	   final accuracy on a test set as well"""

	#load data
	num_classes = 10
	images_train,labels_train = load_data('train_32x32.mat')
	tr_data = normalize(images_train)
	tr_labels = one_hot_encode_labels(labels_train,num_classes)

	images_test,labels_test = load_data('test_32x32.mat')
	te_data = normalize(images_test)
	te_labels = one_hot_encode_labels(labels_test,num_classes)

	x,y_,model,train_op,accuracy,keep_drop = classifier(learning_rate, True)

	with tf.Session() as sess:
			sess.run(model)

			count = 0
			#get sample length
			sample_length = tr_data.shape[0]
			#get number of batches
			num_batches = int(sample_length/batch_size)

			#make the containers to hold test and train accuracy
			train_accuracy = np.zeros([num_epochs*num_batches,1])
			test_accuracy = np.zeros([num_epochs,1])
			print(Hello)
			for epoch in range(num_epochs):
				
				#get shuffled index
				shuffled_indexes = np.arange(sample_length)
				np.random.shuffle(shuffled_indexes)
				#shuffle data
				shuffled_train_data = tr_data[shuffled_indexes]
				shuffled_train_labels = tr_labels[shuffled_indexes]

				for i in range(num_batches):
				
					#gather batches
					start = i*batch_size
					end = (i+1)*batch_size

					#make sure dont access part of an array that doesn't exist(small overlap of data from previous batch will occur - artifact of uneven data set and even batch size)
					if end > sample_length:
						end = sample_length
					if start > sample_length-batch_size:
						start = sample_length-batch_size

					batch_x,batch_y = shuffled_train_data[start:end][:],shuffled_train_labels[start:end][:]
					print(batch_x.shape)
					_,train_accuracy[count] = sess.run([train_op, accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob: 0.5})

					#prints out the accuracy every 7 batches (So that an even amount gets printed out based on num_batches)
					print_out = int(num_batches/7)
					if i%print_out==0:
						print("epoch%d, train_step %d, training accuracy %g"%(epoch, i, train_accuracy[count]))
				
					count +=1

				test_accuracy[epoch] = sess.run(accuracy, feed_dict={x: te_data, y_: te_labels, keep_drop: 1.0})	
				print("epoch: %d, test accuracy: %g"%(epoch, test_accuracy[epoch]))
				print("--------------------------------------------------------------")

	#calculates average accuracy at five points#plots train accuracy
	train_line, = plt.plot(train_accuracy,'r.', label = 'train accuracy')
	test_line, = plt.plot([e*num_batches for e in range(num_epochs)], test_accuracy,'b-', label = 'test accuracy')

	plt.legend(loc = 'lower right')
	plt.xlabel('Batch Number')
	plt.ylabel('Accuracy')
	plt.title('Prediction Accuracy vs Batch Number')
	#plt.legend(handles=[])
	plt.show()

if __name__ == '__main__':

	# raw_batch_size = input("Please input batch size (suggested 100): " )
	# raw_num_epochs = input("Please input number of epochs (suggested 5-7): ")
	# raw_learning_rate = input("Please input learning rate (suggested 0.5-0.3): ")

	# while (float(raw_learning_rate)>1 or float(raw_learning_rate) <0):
	# 	print('Learning rate must be in between 0-1 inclusive')
	# 	raw_learning_rate = input("Please input learning rate (suggested 0.5): ")

	# learning_rate = float(raw_learning_rate)
	# batch_size = int(raw_batch_size)
	# num_epochs = int(raw_num_epochs)

	# train_model(num_epochs,batch_size,learning_rate)
	train_model(10,100,0.5)