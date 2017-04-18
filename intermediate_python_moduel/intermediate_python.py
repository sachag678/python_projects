class Groceries:
	def __init__(self):
		"""initializes the empty grocery list"""
		self.grocery_list = {}

	def add_item(self, item):
		"""adds an item to the grocery_list and sets it value to false"""
		self.grocery_list.update({item: False})

	def remove_item(self, item):
		"""removes an item from grocery_list"""
		del self.grocery_list[item]
	
	def check_item(self, item):
		"""sets a value of an item in the grocery_list to true"""
		self.grocery_list[item] = True

	def items_remaining(self):
		"""Returns the number of unsold items"""
		return len([item for (item, sold) in self.grocery_list.items() if sold == False])
		

#testing
g = Groceries()

#adds items to the bag
g.add_item('Bag')
g.add_item('table')
g.add_item('book')
print(g.grocery_list)

#removes an item
g.remove_item('table')
print(g.grocery_list)

#sells and item
g.check_item('Bag')
print(g.grocery_list)

#counts number of unsold items
print(g.items_remaining())