"""Sacha Gunaratne 3rd May 2017 v.10"""
import scipy.io as sio
from skimage.color import rgb2grey
import numpy as np

def normalize(images):
	"""This function normalizes the image files and returns normalized images(0 mean and unit variance)"""
	output = np.zeros((images.shape))
	for num_image in range(images.shape[3]):
		output[:,:,:,num_image] = (images[:,:,:,num_image] - np.mean(images[:,:,:,num_image]))/np.std(images[:,:,:,num_image])

	return output


def load_data(filename):
	"""loads data from a .mat file and returns a list of images"""	
	data = sio.loadmat(filename)
	x = data['X']
	y = data['y']

	return x,y

if __name__ =='__main__':
	images,labels = load_data('test_32x32.mat')
	normalized_images = normalize(images)
	print(normalized_images.shape)