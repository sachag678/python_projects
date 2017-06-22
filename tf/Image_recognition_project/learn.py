"""Sacha Gunaratne May 2017 v1.0"""
import tensorflow as tf
import numpy as np
import random

from graph_construction import classifier
from data_preprocessing import load_data, one_hot_encode_labels, normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_model(num_epochs, batch_size):
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
	
	#split test set into validation set and test set 50/50
	num_feat_val_set = 500
	val_data , val_labels = te_data[:num_feat_val_set], te_labels[:num_feat_val_set]
	testing_data_and_labels = generate_data(te_data[num_feat_val_set:], te_labels[num_feat_val_set:])

	x,y_,model,train_op,accuracy,keep_prob = classifier(True)

	exit = False

	plt.ion()

	with tf.Session() as sess:
			sess.run(model)

			count = 0
			#training 
			#get sample length
			sample_length = tr_data.shape[0]
			#get number of batches
			num_batches = int(sample_length/batch_size)

			#make the containers to hold test and train accuracy
			train_accuracy = np.zeros([num_epochs*num_batches,1])
			val_accuracy = np.zeros([num_epochs*num_batches,1])
			test_accuracy = 0

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

					_,train_accuracy[count] = sess.run([train_op, accuracy], feed_dict={x:batch_x,y_:batch_y,keep_prob: 0.5})

					val_accuracy[count] = sess.run(accuracy, feed_dict={x: val_data, y_: val_labels, keep_prob: 1.0})	

					#prints out the accuracy every 7 batches (So that an even amount gets printed out based on num_batches)
					print_out = int(num_batches/7)
					if i%print_out==0:
						print("epoch: %d, train_step %d, training accuracy %g"%(epoch, i, train_accuracy[count]))
						print("epoch: %d, test accuracy: %g"%(epoch, val_accuracy[count]))

						#plotting
						train_line, = plt.plot(count*i,train_accuracy[count],'bo', label='train_accuracy')
						val_line, = plt.plot(count*i,val_accuracy[count],'ro',label = 'validation_accuracy')
						plt.xlabel('Batch Number')
						plt.ylabel('Accuracy')
						plt.title('Validation & train accuracy vs Batch Number')
						plt.pause(0.05)

					#exit training if there is increased validation error (decrease in test accuracy)
					if val_accuracy[count]<(val_accuracy[count-1]-0.1):
						exit = True
						break

					count +=1

				#exit training if there is increased validation error (decrease in test accuracy)
				if exit:
					print("Overfitting occuring: exiting training phase")
					break
				print("--------------------------------------------------------------")

			#calculate test accuracy on unseen test set
			for _ in range(te_data.shape[0]-num_feat_val_set):
				batch = next(testing_data_and_labels)
				test_accuracy += sess.run(accuracy, feed_dict={x: np.reshape(batch[0], (1,32,32,3)), y_: np.reshape(batch[1], (1,10)), keep_prob: 1.0})

	print("Testing Accuracy: %g" %(float(test_accuracy)/float(te_data.shape[0]-num_feat_val_set)))

	# while True:
	# 	plt.pause(0.05)

def generate_data(data, labels):
	"""uses a generator to solve the OOM problem for the testing"""
	for x in range(data.shape[0]):
		yield (data[x],labels[x])

if __name__ == '__main__':

	# raw_batch_size = input("Please input batch size (suggested 50): " )
	# raw_num_epochs = input("Please input number of epochs (suggested 10): ")
	
	# batch_size = int(raw_batch_size)
	# num_epochs = int(raw_num_epochs)

	# train_model(num_epochs,batch_size)
	train_model(2,50)