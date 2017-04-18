import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def func_1():
	x = tf.placeholder('float', [None, None])
	y = x*2

	with tf.Session() as session:
		x_data = [[1,2,3],
				  [3,4,5]]
		result = session.run(y, feed_dict={x: x_data})
		print(result)

def func2():
	img_data = mpimg.imread('MarshOrchid.jpg')

	image = tf.placeholder('uint8',[None,None,None])
	slice1 = tf.slice(image, [100,45,0], [3000,-1,-1])

	with tf.Session() as session:
		result = session.run(slice1, feed_dict={image:img_data})
	#greyscaled
	plt.imshow((result[:,:,0]/3+result[:,:,1]/3+result[:,:,2]/3))
	plt.show()

func2()