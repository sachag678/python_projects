def func1():
	#run through a list and output its index as well - use enumerate
	cities = ['Colombo','Ottawa','Toronto', 'Rangoon']

	for i,city in enumerate(cities):
		print(i,city)

def func2():
	#printing the values from two lists at the same time
	xl = [1,2,3]
	yl = [2,4,6]
	#zip creates tuples of two lists (1,2), (2,4), (3,6)
	for x, y in zip(xl,yl):
		print(x,y)

def func3():
	#swap variable values (generally used temp variables)
	# y=-10
	# x=10
	x,y = 10,-10
	print('Before x is ', x, ' and y is', y)
	#tmp = y
	#y=x
	#x=tmp
	#tuple unpacking
	x, y = y, x
	print('After x is ', x, ' and y is', y)

def func4():
	#how to determine if a person is in the dict or not
	ages = {
		'Mary': 31,
		'Kohn': 32
	}
	#gets the age in the ages, but if its not there the default value is 
	age = ages.get('Dick','unknown')
	print('Dick is ', age, ' years old')

def func5():
	#searching a list for a specific value
	needle = 'd'
	haystack = ['a','b','c']

	#generally use a for loop and an if statement which has a else statement
	for letter in haystack:
		if needle==letter:
			print('Found')
			break
	else: # If no break occured this will be executed
		print('Not Found')

def func6():
	#file read
	f = open('tensor_flow_readme.txt')
	for line in f:
		print(line)
	f.close()

	#with statement allows us to not close the file - using context
	with open('tensor_flow_readme.txt') as f:
		for line in f:
			print(line)

def func7():
	print('Converting')
	try:
		print(int('1'))
	except:
		print('Failed')
	else:
		print('Passed')
	finally: # finally will always execute before the program crashes even if except is not there or exception occurs
		print('Done!')

def func8():
	my_dict = {"j":4,"h":5}
	for key,value in my_dict.iteritems(): #returns one item at a time instead of all if you use items()
		print(my_dict[key])
func7()