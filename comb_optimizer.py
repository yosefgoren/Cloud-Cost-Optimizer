class CombOptim:
	def __init__(self, k: int, price_calc):
		"""'price_calc' is a function: (Offer)-->float"""
		self.optim_set = OptimumSet()
		self.reset_sel = ResetSelector()
		self.search_algo = SearchAlgorithm(price_calc)
	
	def run(self):
		while not self.isDone():
			start_node = self.reset_sel.getStartNode()
			path = self.search_algo.run(start_node)
			self.optim_set.update(path)
			self.reset_sel.update(path)
		return [node.offer for node in self.optim_set.returnBest()]

	def isDone(self)->bool:
		#TODO
		pass

class Node:
	def __init__(self, parent):
		self.price = None
		self.offer = None
		self.sons = None
		
	def getAllSons():
		#TODO
		pass
	def getPrice():
		#TODO
		pass
	def asOffer():
		#TODO
		pass
	def hashCode():
		#TODO
		pass

class OptimumSet:
	def __init__(self, k: int):
		"""the table holds the best k seen so far in terms of price."""
		self.table = [None]*k
	
	def update(self, node: list):
		"""if parameter 'node' has better price than anyone in the table, insert 'node' to the table
			, possibly removing the worst node currently in the table."""
		#TODO
		pass

	def returnBest(self):
		return self.table


class SearchAlgorithm:
	def __init__(self, price_calc):
		self.price_calc

	def run(self, start_node: Node)->list:
		"""returns the list of nodes visited in the run"""
		#to get price of Node:
		# price = self.price_calc(node.getOffer())
		#TODO
		pass

class ResetSelector:
	def __init__(self):
		#add root to DS...
		#TODO
		pass

	def getStartNode(self):
		#TODO
		pass

	def update(self, path: list):
		#TODO
		pass