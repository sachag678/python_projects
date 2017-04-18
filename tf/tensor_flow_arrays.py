import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

filename = "MarshOrchid.jpg"

image = mpimg.imread(filename)
height, width, depth = image.shape

x = tf.Variable(image, name='x')

model = tf.global_variables_initializer()

with tf.Session() as session:
	#2 times makes it upside down
	for i in range(2):
		xshape = tf.shape(x)
		result = session.run(xshape)
		x = tf.reverse_sequence(x, [result[1]]*result[0],1,batch_dim=0)
		x = tf.transpose(x, perm=[1,0,2])
		session.run(model)
		result = (session.run(x))


#print(image.shape)

plt.imshow(result)
plt.show()