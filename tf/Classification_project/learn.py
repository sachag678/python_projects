"""Sacha Gunaratne May 2017 v1.0"""
import tensorflow as tf
import numpy as np
import random

from graph_constructor import knn, logistic_classifier,two_layer_net
from data_preprocessing import load_data, one_hot_encode_labels, normalize
import matplotlib.pyplot as plt

def train_model(learn_model, num_epochs, batch_size, learning_rate):
	"""takes a some parameters, trains a specified model,
	   calculates accuracy during training and prints it out. Prints out
	   final accuracy on a test set as well"""

	#load data
	num_classes = 6
	tr_data = normalize(load_data('X_train.txt'))
	tr_labels = one_hot_encode_labels(load_data('y_train.txt'),num_classes)
	te_data = normalize(load_data('X_test.txt'))
	te_labels = one_hot_encode_labels(load_data('y_test.txt'),num_classes)

	#determine which learn method to run
	if learn_model =='logistic' or learn_model == '2-layer':

		if learn_model=='logistic':
			x,y_,model,train_op,accuracy = logistic_classifier(learning_rate)
		else:
			x,y_,model,train_op,accuracy = two_layer_net(learning_rate)

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
					_,train_accuracy[count] = sess.run([train_op, accuracy], feed_dict={x:batch_x,y_:batch_y})

					#prints out the accuracy every 7 batches (So that an even amount gets printed out based on num_batches)
					print_out = int(num_batches/7)
					if i%print_out==0:
						print("epoch%d, train_step %d, training accuracy %g"%(epoch, i, train_accuracy[count]))
				
					count +=1

				test_accuracy[epoch] = sess.run(accuracy, feed_dict={x: te_data, y_: te_labels})	
				print("epoch: %d, test accuracy: %g"%(epoch, test_accuracy[epoch]))
				print("--------------------------------------------------------------")

		#calculates average accuracy at five points
		
		#plots train accuracy
		train_line, = plt.plot(train_accuracy,'r.', label = 'train accuracy') 
		test_line, = plt.plot([e*num_batches for e in range(num_epochs)], test_accuracy,'b-', label = 'test accuracy')

		plt.legend(loc = 'lower right')
		plt.xlabel('Batch Number')
		plt.ylabel('Accuracy')
		plt.title('Prediction Accuracy vs Batch Number')
		#plt.legend(handles=[])
		plt.show()

	if learn_model =='knn':

		#get input from user
		raw_k_val = input("Please enter odd k value (recommend 7) or enter 0 to run a list of k values: ")

		if int(raw_k_val)==0:
			#ran multiple k's to decide which was best for this data set (Keep odd so that there are no ties)
			num_neighbours = [1,3,5,7,9,11,13,15]
			#num_neighbours = [1,3]
		else:
			num_neighbours = [int(raw_k_val)]

		#will contain the accuracy for each value of k
		train_accuracy = np.zeros((len(num_neighbours)))

		for index, k in enumerate(num_neighbours):
			#retrieve object for graph_constructor
			x,y_,xtest, ytest,accuracy, model,train_op = knn(k)
			with tf.Session() as sess:
				sess.run(model)
				# loop over test data
				for i in range(len(te_data)):
					# Get nearest neighbor to each row of test data which represents one multi dimensional data point
					train_accuracy[index] += sess.run(accuracy, feed_dict={x: tr_data, y_: tr_labels, xtest: te_data[i, :], ytest: te_labels[i]})

					if i%200==0:
						print(str(i) + ' out of ' + str(len(te_data)) + ' have been tested')

			print("k = {}, Accuracy: {} ".format(k, train_accuracy[index]/len(te_data)))

		#only plot if there is more than one k value
		if int(raw_k_val) == 0: 
			plt.plot(num_neighbours, train_accuracy/len(te_data), 'ro')
			plt.xlabel('K - value')
			plt.ylabel('Accuracy')
			plt.title('Prediction Accuracy vs Batch Number')
			plt.show()


if __name__ =='__main__':

	model = input("Please enter model(knn,2-layer,logistic): ")
	
	if(model=="knn"):
		train_model('knn',0,0,0)
	else:
		raw_batch_size = input("Please input batch size (suggested 100): " )
		raw_num_epochs = input("Please input number of epochs (suggested 5-7): ")
		raw_learning_rate = input("Please input learning rate (suggested 0.5-0.3): ")

		while (float(raw_learning_rate)>1 or float(raw_learning_rate) <0):
			print('Learning rate must be in between 0-1 inclusive')
			raw_learning_rate = input("Please input learning rate (suggested 0.5): ")

		learning_rate = float(raw_learning_rate)
		batch_size = int(raw_batch_size)
		num_epochs = int(raw_num_epochs)

		if(model=="logistic"):
			train_model('logistic',num_epochs,batch_size,learning_rate)
		if(model=="2-layer"):
			train_model('2-layer',num_epochs,batch_size,learning_rate)
