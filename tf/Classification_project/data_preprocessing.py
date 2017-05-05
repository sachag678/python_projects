"""Sacha Gunaratne May 2017 v1.0"""
import numpy as np

def normalize(batch):
	"""Takes in a batch of data and normalizes it to zero mean and unit variance"""

	#creates numpy array of type float
	normalized_batch = np.array(batch,dtype='f')
	#gets batch size and num-features
	_ ,num_features = normalized_batch.shape
	#runs through array and normalizes
	for i in range(num_features):
		normalized_batch[:,i] = ((normalized_batch[:,i]-np.mean(normalized_batch[:,i]))/np.std(normalized_batch[:,i]))
	return normalized_batch

def load_data(filename):
	"""takes a filename and reads data and outputs a list"""
	data = []
	with open(filename) as f:
		for line in f:
		 	data.append([float(x) for x in line.split()]) #outputs a 2-D list
			#data2 = [float(x) for line in f for x in line.split()] #outputs a 1-D list
	return data

def one_hot_encode_labels(batch, num_classes):
	"""takes a list of labels and creates a one-hot vector for each row
	   eg:
	   if y[0:1] = 2 && num_classes = 6:
	   		then y[0:1] = [0,0,1,0,0,0] 
	"""
	labels = np.zeros((len(batch),num_classes))
	for j in range(len(batch)):
		for i in range(num_classes):
			if(batch[j]==[i]):
				labels[j][i-1]=1

	return labels

if __name__ =='__main__':
	#testing
	#load data
	data = load_data('X_train.txt')
	labels = load_data('y_train.txt')
	#convert to normalized
	normalized_batch = normalize(data)
	ohc_lables = one_hot_encode_labels(labels,6)
	#look at shape
	print(normalized_batch.shape)
	#print(labels[:][:])
	print(ohc_lables.shape[0])

	print(normalized_batch.dtype)

