import copy

from fleet_classes import Offer

class CombOptim:
	price_calc_func = None

	def __init__(self, k: int, price_calc, initial_seperated):
		"""'price_calc' is a function: (Offer)-->float"""
		CombOptim.price_calc_func = price_calc
		self.optim_set = OptimumSet(10)
		self.reset_sel = ResetSelector(k)
		self.search_algo = SearchAlgorithm()
		self.root = CombOptim.calc_root(initial_seperated)

	@staticmethod
	def calc_root(initial_seperated):
		partitions = list(map(lambda i: separate_partitions(i), initial_seperated))
		return Node(partitions)

	def get_root(self):
		return self.root

	@staticmethod
	def price_calc(offer):
		return CombOptim.price_calc_func(offer)
	
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
	node_cache = {}

	def __init__(self, partitions):
		self.partitions = copy.deepcopy(partitions)
		self.offers = []
		self.price = 0
		self.sons = None

		offers_init = [Offer(p, None) for p in partition2(self.partitions)]
		for offer in offers_init:
			self.offers.append(CombOptim.price_calc(offer))

		for offer in self.offers:
			self.price += offer.total_price
		print(self.price)

	def getAllSons(self):
		if self.sons == None:
			self.sons = []
			for partition in self.partitions:
				partition = partition[0]  # we assume this layer has no meaning
			pass



		# 	#num_modules = len(modules)
		# 	for first_module_idx in range(num_modules):
		# 		for second_module_idx in range(num_modules):
		# 			if first_module_idx != second_module_idx:
		# 				#TODO: create new module set from 'modules' and 'second_module_idx', 'first_module_idx'. put it in son_comb:
		# 				son_comb = None
		# 				hash = self.hashCode(son_comb)
		# 				if not hash in self.node_cache:
		# 					#TODO: create new node from son_comb, put it in son_node:
		# 					son_node = None
		# 					self.node_cache[hash] = son_node
		# 				else:
		# 					#take existing node from cache:
		# 					son_node = self.node_cache[hash]
		# 				self.sons.append(son_node)
		# return self.sons

	def getPrice(self):
		return self.price

	def hashCode(comb: list):
		#TODO
		pass

class OptimumSet:
	def __init__(self, k: int):
		"""the table holds the best k seen so far in terms of price."""
		self.table = [None]*k
	
	def update(self, visited_nodes: list):
		"""if parameter 'node' has better price than anyone in the table, insert 'node' to the table
			, possibly removing the worst node currently in the table."""
		#TODO
		pass

	def returnBest(self):
		return self.table

class SearchAlgorithm:
	def __init__(self):
		pass

	def run(self, start_node: Node)->list:
		"""returns the list of nodes visited in the run"""
		#to get price of Node:
		# price = self.price_calc(node.getOffer())
		#TODO
		pass

class ResetSelector:
	def __init__(self, k: int):
		#add root to DS...
		#TODO
		pass

	def getStartNode(self):
		#TODO
		pass

	def update(self, path: list):
		#TODO
		pass

	def getExpoitationBias(self)->float:
		#TODO
		pass

	def calcTotalScore(self, comb: Node)->float:
		#TODO
		pass

	def totalScoreAux(self, exporation_score: float, exploitation_score: float)->float:
		#TODO
		pass

	def calcExporationScore(comb: Node)->float:
		#TODO
		pass

	def calcExpoitationScore(comb: Node)->float:
		#TODO
		pass