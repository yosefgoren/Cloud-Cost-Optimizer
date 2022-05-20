import copy

from BBAlgorithm import separate_partitions
from fleet_classes import Offer
import numpy as np

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
        self.offer = self.__calc_offer()
        self.price = self.offer.total_price
        print(self.price)
        self.sons = None

    def __calc_offer(self):
        modules = []
        for group in self.partitions:
            combination = group[0]  # each group has 1 combination
            for module in combination:
                modules.append(module)

        return CombOptim.price_calc_func(Offer(modules, None))

    def getPrice(self):
        return self.price

    def getOffer(self):
        return self.offer

    # offers_init = [Offer(p, None) for p in partition2(self.partitions)]
    # for offer in offers_init:
    # 	self.offers.append(CombOptim.price_calc(offer))
    #
    # for offer in self.offers:
    # 	self.price += offer.total_price
    # print(self.price)

    def calcAllSons(self):
        if self.sons is None:
            self.sons = []
            for i, group in enumerate(self.partitions):
                combination = group[0]  # each group has 1 combination
                for j, module1 in enumerate(combination):
                    for k, module2 in enumerate(combination):
                        if j < k:
                            new_combination = copy.deepcopy(combination)
                            new_module = copy.deepcopy(module1 + module2)
                            del new_combination[max(j, k)]
                            del new_combination[min(j, k)]
                            new_combination.append(new_module)

                            new_partition = copy.deepcopy(self.partitions)
                            new_partition[i][0] = new_combination

                            self.sons.append(Node(new_partition))

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

    def hashCode(comb: list):
        # TODO
        pass


class OptimumSet:
    def __init__(self, k: int):
        """the table holds the best k seen so far in terms of price."""
        self.table = [None] * k

    def update(self, visited_nodes: list):
        """if parameter 'node' has better price than anyone in the table, insert 'node' to the table
			, possibly removing the worst node currently in the table."""
        # TODO
        pass

    def returnBest(self):
        return self.table


class SearchAlgorithm:
    def __init__(self):
        self.temperature = 0
        self.temperature_increment_pace = 1

    def run(self, start_node: Node) -> list:
        """returns the list of nodes visited in the run"""
        path = []
        next_node = start_node

        while True:
            path.append(next_node)
            next_node = self.get_next(next_node)
            self.update_temperature()
            if next_node is None:
                return path

    def get_next(self, node: Node) -> Node:
        """get the chosen son to continue to in the next iteration"""
        node.calcAllSons()
        improves, downgrades = SearchAlgorithm.split_sons_to_improves_and_downgrades(node.sons, node.getPrice())
        if (downgrades.shape[0] != 0) and self.is_choosing_downgrades():
            return SearchAlgorithm.get_son_by_weights(downgrades)
        else:
            return SearchAlgorithm.get_son_by_weights(improves)

    @staticmethod
    def get_son_by_weights(sons):
        """get the choosen son by the weights of the price diffs"""
        if sons.shape[0] == 0:
            return None

        index = sampleFromWeighted(np.array(sons[:, 1], dtype=np.float))
        return sons[index, 0]

    @staticmethod
    def split_sons_to_improves_and_downgrades(all_sons, cur_node_price):
        """split the sons to 2 ndarray of improves and downgrades. first column is sons, second column is the son
		corrsponding pricr diff"""
        improves = []
        downgrades = []

        for son in all_sons:
            price_diff = son.getPrice() - cur_node_price
            if price_diff > 0:
                improves.append([son, price_diff])
            else:
                downgrades.append([son, price_diff])

        return np.array(improves), np.array(downgrades)

    def is_choosing_downgrades(self):
        """return if we will choose a downgrade son"""
        prob_for_downgrade = 0.1 - 1.0 / (10 * np.exp(1 / (np.power(self.temperature, 0.9))))
        return (np.random.choice([0, 1], p=[1 - prob_for_downgrade, prob_for_downgrade])) == 1

    def update_temperature(self):
        """change the temperature according to the self.temperature_increment_pace"""
        self.temperature += self.temperature_increment_pace


class ResetSelector:
    def __init__(self, k: int):
        # add root to DS...
        # TODO
        pass

    def getStartNode(self):
        # TODO
        return CombOptim.calc_root()

    def update(self, path: list):
        # TODO
        pass

    def getExpoitationBias(self) -> float:
        # TODO
        pass

    def calcTotalScore(self, comb: Node) -> float:
        # TODO
        pass

    def totalScoreAux(self, exporation_score: float, exploitation_score: float) -> float:
        # TODO
        pass

    def calcExporationScore(comb: Node) -> float:
        # TODO
        pass

    def calcExpoitationScore(comb: Node) -> float:
        # TODO
        pass


def sampleFromWeighted(weight_arr: np.ndarray) -> int:
    sum_array = weight_arr.sum()
    weight_arr = weight_arr / sum_array
    index = np.random.choice(weight_arr.shape[0], p=weight_arr)
    return index
