import unittest
from advanced_python_1 import get_input, is_not_number

class TestCheckWeather(unittest.TestCase):
	"""Tests advanced_python_1.py"""

	def test_get_input(self):
		self.assertEqual(get_input(),('5','-10'))

	def test_is_not_number(self):
		self.assertTrue(is_not_number('gh'))

	def test_is_not_number_next(self):
		self.assertFalse(is_not_number('-5.4'))

if __name__ == '__main__':
	unittest.main()