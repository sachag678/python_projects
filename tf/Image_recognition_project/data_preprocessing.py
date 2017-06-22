"""Sacha Gunaratne 3rd May 2017 v.10"""
import scipy.io as sio
from skimage.color import rgb2grey
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def normalize(images):
	"""This function normalizes the image files and returns normalized images(0 mean and unit variance)"""
	output = np.zeros((images.shape[3],32,32,3))
	for num_image in range(images.shape[3]):
		output[num_image,:,:,:] = (images[:,:,:,num_image] - np.mean(images[:,:,:,num_image]))/np.std(images[:,:,:,num_image])

	return np.float32(output)


def load_data(filename):
	"""loads data from a .mat file and returns a list of images"""	
	data = sio.loadmat(filename)
	x = data['X']
	y = data['y']

	return x,y

def one_hot_encode_labels(batch, num_classes):
	"""takes a list of labels and creates a one-hot vector for each row
	   eg:
	   if y[0:1] = 2 && num_classes = 6:
	   		then y[0:1] = [0,0,1,0,0,0] 
	"""
	# labels = np.zeros((len(batch),num_classes))
	# for j in range(len(batch)):
	# 	for i in range(num_classes):
	# 		if(batch[j]==[i]):
	# 			labels[j][i-1]=1

	# return labels
	encoder = OneHotEncoder()
	return encoder.fit_transform(batch).toarray()


if __name__ =='__main__':
	images,labels = load_data('test_32x32.mat')
	ohc_labels = one_hot_encode_labels(labels,10)
	print(type(ohc_labels))
	normalized_images = normalize(images)
	print(normalized_images[:13000].shape)
	#print(normalized_images.shape[0])
	#print(normalized_images.dtype)
	# new_normalized = np.float32(normalized_images)
	# print(type(new_normalized[1,1,1,1]))
	# print(new_normalized.shape)


