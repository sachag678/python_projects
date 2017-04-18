import tensorflow as tf
import numpy as np

def func_1():
	"""creates variables and constants and outputs in session"""
	hello = tf.constant('Hello Tensor')

	x = tf.constant([35.76,40,45],name='x')
	y = tf.Variable(x+5,name='y')

	model = tf.global_variables_initializer()

	with tf.Session() as session:
		session.run(model)
		print(session.run(y))

def func_2():
	"""can store large np arrays as well"""
	data = np.random.randint(1000, size=10000)

	x = tf.constant(data,name='x')
	y = tf.Variable(5*x^2-3*x+15,name = 'y')

def func_3():
	"""can update the variable as you go along - importnant for ML"""
	x = tf.Variable(0, name='x')

	model = tf.global_variables_initializer()

	with tf.Session() as session:
	    for i in range(5):
	        session.run(model)
	        x = x + 1
	        print(session.run(x))

def func_4():

	data = np.random.randint(1000)
	x = tf.Variable(data, name='x')

	model = tf.global_variables_initializer()

	with tf.Session() as session:
	    for i in range(2,7):
	        session.run(model)
	        data = np.random.randint(1000)
	        x = (x+data)/(i)
	        print(session.run(x))
	        

func_4()