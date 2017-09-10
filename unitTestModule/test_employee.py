import unittest
from employee import Employee
from unittest.mock import patch


class TestEmployee(unittest.TestCase):

    #use setup classes to call database once and get data
    @classmethod
    def setUpClass(cls):
        print('setupClass')

    def setUp(self):
        #instance attributes
        print('setup')
        self.emp_1 = Employee('Corey','Schafer',52500)
        self.emp_2 =Employee('Sue','Smith',63000)

    def tearDown(self):
        print('teardown')


    def test_email(self):
        print('test_email')
        self.assertEqual(self.emp_1.email, 'Corey.Schafer@email.com')
        self.assertEqual(self.emp_2.email, 'Sue.Smith@email.com')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.email, 'John.Schafer@email.com')
        self.assertEqual(self.emp_2.email, 'Jane.Smith@email.com')

    def test_fullname(self):
        print('test_fullname')
        self.assertEqual(self.emp_1.fullname, 'Corey Schafer')
        self.assertEqual(self.emp_2.fullname, 'Sue Smith')

        self.emp_1.first = 'John'
        self.emp_2.first = 'Jane'

        self.assertEqual(self.emp_1.fullname, 'John Schafer')
        self.assertEqual(self.emp_2.fullname, 'Jane Smith')

    def test_apply_raise(self):
        print('test_apply_raise')
        self.emp_1.apply_raise()
        self.emp_2.apply_raise()

        self.assertEqual(self.emp_1.pay, 55125)
        self.assertEqual(self.emp_2.pay, 66150)

    def test_monthly(self):
        print('monthly_test')
        #use mock to call to a fake object
        #set the mock values to specific True or false
        #set the response text
        with patch('employee.requests.get') as mocked_get:
            mocked_get.return_value.ok = True
            mocked_get.return_value.text = 'Success'

            schedule = self.emp_1.monthly_schedule('May')
            mocked_get.assert_called_with('http://company.com/Schafer/May')
            self.assertEqual(schedule,'Success')

if __name__ == '__main__':
    unittest.main()