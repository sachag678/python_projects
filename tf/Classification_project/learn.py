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

		train_accuracy = np.zeros([num_epochs,1])

		with tf.Session() as sess:
			sess.run(model)

			for epoch in range(num_epochs):
				#random sampling
				sample_length = tr_data.shape[0]
				start = random.randint(0,sample_length)
				end = start+batch_size
				
				#gather samples
				batch_x,batch_y = tr_data[start:end][:],tr_labels[start:end][:]
				_,train_accuracy[epoch] = sess.run([train_op, accuracy], feed_dict={x:batch_x,y_:batch_y})

				#prints out the accuracy every n batches and plots it to the graph
				if epoch%50==0:
					print("epoch %d, training accuracy %g"%(epoch, train_accuracy[epoch]))
				
			print("test accuracy %g"%sess.run(accuracy, feed_dict={x: te_data, y_: te_labels}))

		#calculates average accuracy at five points
		avg_acc = np.zeros([5])
		for i in range(5):
			avg_acc[i] = np.mean(train_accuracy[i*int(num_epochs/5):(i+1)*int(num_epochs/5)])

		gap = int(int(num_epochs/5)/2)
		avg_acc_x = [i*int(num_epochs/5)+gap for i in range(5)]
		#plots train accuracy
		plt.plot(train_accuracy,'r.',avg_acc_x, avg_acc,'b-',avg_acc_x, avg_acc,'bo' )
		plt.xlabel('Batch Number')
		plt.ylabel('Accuracy')
		plt.title('Prediction Accuracy vs Batch Number')
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
		raw_batch_size = input("Please input batch size (suggested 200): " )
		raw_num_epochs = input("Please input number of epochs (suggested 2000): ")
		raw_learning_rate = input("Please input learning rate (suggested 0.5): ")

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
