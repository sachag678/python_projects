import unittest
import functions

class TestFunctions(unittest.TestCase):

	#have to call these starting with test
	def test_add(self):
		self.assertEqual(4,functions.add(2,2))
		self.assertEqual(0,functions.add(-2,2))
		self.assertEqual(-2,functions.add(-1,-1))

	def test_multiply(self):
		self.assertEqual(4,functions.multiply(2,2))
		self.assertEqual(-4,functions.multiply(-2,2))
		self.assertEqual(1,functions.multiply(-1,-1))

	def test_divide(self):
		self.assertEqual(2,functions.divide(2,1))
		# two different options to test Exceptions
		#using context manager
		with self.assertRaises(ValueError):
			functions.divide(5,0)

		#other option using assertRaises and pass the arguments to it.
		self.assertRaises(ValueError,functions.divide,5,0)

#this allows me to run the .py file without using python -m unittest test_functions.py
if __name__ == '__main__':
	unittest.main()
