# from msilib.schema import Component
# from importlib_metadata import Pair
import numpy as np
from numpy import ndarray
# from urllib3 import Retry
from fleet_classes import Offer
from math import inf
import copy
from BBAlgorithm import separate_partitions

class KeyMannager:
    def __init__(self, unique_identifier_func):
        """an instance of this class will be able to take elements and assign each element a unique int key,
            the 'unique_identifier_func' function is used to determined how an element differs from other elements."""
        self.id_func = unique_identifier_func
        self.counter = 0
        self.key_mappings = {}
    
    def __call__(self, element)->int:
        element_id = self.id_func(element)
        
        if not element_id in self.key_mappings:
            self.key_mappings[element_id] = self.counter
            self.counter += 1

        return self.key_mappings[element_id]
        
        
class CombOptim:
    def __init__(self, k: int, price_calc, initial_seperated):
        """'price_calc' is a function: (Offer)-->float"""
        CombOptim.price_calc_func = price_calc
        self.root = CombOptim.calc_root(initial_seperated)
        self.optim_set = OptimumSet(10)
        self.reset_sel = ResetSelector(k,self.get_num_components(),self.root)
        self.search_algo = SearchAlgorithm()

        CombOptim.getComponentKey = KeyMannager(lambda componenet: componenet.component_name)
        """given a component, return a unique key associated with it, based on it's name."""
    
        CombOptim.getModuleKey = KeyMannager(lambda module: tuple([self.getComponentKey(component) for component in module].sort()))
        """give a unique key to a module - a module is a set of components and is distinct from another
        module if they do not have the same sets of components."""
    
        CombOptim.getCombinationAsKey = KeyMannager(lambda combination: tuple([self.getModuleKey(module) for module in combination].sort()))
        """given a comination - meaning a set (unordered) of modules, return a unique key associated with it."""

        CombOptim.getGroupSetAsKey = KeyMannager(lambda group_set: tuple([self.getCombinationAsKey(group[0]) for group in group_set].sort()))
        """given a set of groups (the parameter is of type list, but the order is not considered
        to be a diffirentiating factor) return a unique key associated with the group set.
        Note how 'group[0]' is the one (and only) combination within the group."""
	
    @staticmethod
    def calc_root(initial_seperated):
        partitions = list(map(lambda i: separate_partitions(i), initial_seperated))
        return Node(partitions)

    def get_num_components(self):
        num_of_comp = 0
        for group in self.root.partitions:
            combination = group[0]  # each group has 1 combination
            num_of_comp += len(combination)

        return num_of_comp

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

    def hashCode(self)->int:
        return CombOptim.getGroupSetAsKey(self.partitions)
      
      
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

class OptimumSet:
    def __init__(self, k: int):
        """the table holds the best k seen so far in terms of price.
            requires that the elements inserted will have the method 'getPrice' which should
            return a float."""
        self.k = k
        self.table = []
    
    def update(self, visited_nodes: list):
        """considers the list of new nodes, such that the resulting set of nodes will be the 'k' best nodes
            seen at any update. The ordering the nodes is given by their 'getPrice()' method."""
        candidates = self.table + visited_nodes
        candidates.sort(key=lambda node: node.getPrice())
        self.table = candidates[:self.k]

    def returnBest(self):
        """returns the 'k' nodes with the best price seen so far.
        If not seen 'k' nodes yet, returns a list shorter than 'k'."""
        return self.table


class ResetSelector:
    class Candidate():
        def __init__(self, node: Node):
            """The 'self.reachable_bonus' is a variable used in calculating the exploitation score for
                this candidate.
                 * Each node that can be reached from this candidate has a 'reachable_bonus' associated
                with it and the candidate.
                 * At any givem time, the candidate will save the maximum 'reachable_bonus' that it gets from 
                any nodes that have been reached in runs starting from itself."""
            self.node = node
            self.total_score = node.getPrice()
            self.reachable_bonus = 0
            self.hash = None

    def __init__(self, k: int, num_componants: int, root: Node):
        """ The reset-selector remembers a list of the best candidates (candidate nodes) seen so far,
            list is saved at: self.top_candidates.
            The parameter 'k' is the maximum allowed size for the candidate list."""
        self.top_candidates = [ResetSelector.Candidate(root)]
        self.k = k
        self.num_componants = num_componants

        #reachable_bonus_formula_base is calculated here so we only have to calculate it once.
        self.reachable_bonus_base = 0.1**(1.0/num_componants)
        
        #hyperparameters:
        self.exploitation_score_price_bias = 0.5
        self.exploration_score_depth_bias = 0.5

    def getStartNode(self)->Node:
        """this method represents the main functionality of the reset-selector: based on all data seen so far
            - the reset-selector will return the the node it thinks the next run should start from."""
        scores = np.array([candidate.total_score for candidate in self.top_candidates])
        selected_node_idx = sampleFromWeighted(scores)
        return self.top_candidates[selected_node_idx].node

    def update(self, path: list):
        """'path' is a list of nodes seen in the last run the serach algorithm.
            this method will update in state of the reset selector - to consider the nodes seen in last search run.

            The order of nodes in 'path' is exprected to be the same order as the nodes were seen in the search.
            
            Calling this method will also cause the reset-selector to re-calculate the total scores for each 
            of the candidates saved within it."""
        #consider all nodes seen in last path as candidates:
        candidate_dict = {candidate.node.hashCode():candidate for candidate in self.top_candidates}
        best_reachable_bonus = 0
        for node in reversed(path):
            #add the new node to set of candidates if it's not already there:
            node_hash = node.hashCode()
            if not node_hash in candidate_dict:
                candidate_dict[node_hash] = ResetSelector.Candidate(node)
            node_candidate = candidate_dict[node_hash]
            
            #update the best reachable bonus for the rest of the path:
            best_reachable_bonus = max(node.getPrice(), best_reachable_bonus, node_candidate.reachable_bonus)
            #update the candidate's reachable_bonus to be the best bonus seen for it:
            node_candidate.reachable_bonus = best_reachable_bonus
            #apply diminishing effect to 'best_reachable_bonus':
            best_reachable_bonus *= self.reachable_bonus_base
        
        #update the list of top candidates and re-calculate total scores for all candidates currently saved:
        self.top_candidates = best_reachable_bonus.values()
        self.updateTotalScores()
        
        #sort the list of top candidates and throw away the candidates that are not in the top k:
        self.top_candidates.sort(key=lambda candidate: candidate.total_score)
        self.top_candidates = self.top_candidates[:self.k]
        
    def updateTotalScores(self)->list:
        """updates the total scores (floats) of all candidates in 'self.top_candidates'."""
        ration_scores = self.calcRationScores()
        tation_scores = self.calcTationScores()
        tation_bias = self.getCurrentTationBias()

        total_scores = tation_bias*tation_scores + (1-tation_bias)*ration_scores
        for idx in range(self.top_candidates):
            self.top_candidates[idx].total_score = total_scores[idx]

    def getCurrentTationBias(self)->float:
        """get the current exploitation bias, this represents the current preference of the algorithm for exploitation
            over exploration."""
        return 0.5
        #TODO: we probably want an implementation based on how much time the algorithm has to run,
        #	 s.t. when there is little time left the exploitation bias is close to 1.

    def normalizeArray(arr: ndarray)->ndarray:
        return (arr-arr.min())/(arr.max()-arr.min()) #minmax normalization
        # return arr/np.sum(arr) #normalise according to L1
        #return arr/np.linalg.norm(arr)#normalize according to L2

    def calcRationScores(self)->list:
        """calculates the exploration scores of all candidates in 'self.top_candidates' and returns scores
            in list of floats in same order."""
        uniqueness_scores = self.calcUniquenessScores()
        depth_scores = self.calcDepthScores()
        exploration_scores = self.normalizeArray(self.exploration_score_depth_bias*depth_scores
            +(1-self.exploration_score_depth_bias)*uniqueness_scores)

        return exploration_scores

    def calcDepthScores(self)->ndarray:
        """Calculate the 'depth score' for each candidate in 'self.top_candidates'.
            The deeper the candidate's node - the higher the depth score."""
        depths = np.array([c.node.getDepth() for c in self.top_candidates])
        return self.normalizeArray((depths-self.num_componants)*(depths-self.num_componants))

    def calcUniquenessScores(self)->ndarray:
        """Calculate the 'uniqueness score' for each candidate in 'self.top_candidates'.
            This score will be highest for nodes that are very different from the other nodes in 'top_candidates'."""
        return self.normalizeArray(self.combinationDistancesFormula([c.node for c in self.top_candidates]))

    # def getModuleDemandVector

    def combinationDistancesFormula(node_list: list)->ndarray:
        """Implementation of formula for calculating 'distance' for all nodes to all other nodes.
            The input is a list of combinations, and the output is an array of floats where the i'th float
            represents the average 'distance' of i'th node from the rest of the nodes in the input list.
            
            Input is in the form of list<Node>."""
        #TODO
        pass

    def calcTationScores(self)->ndarray:
        """calculates the explotation scores of all candidates in 'self.top_candidates' and returns scores
            in an array of floats with a corresponding order."""
        reachable_bonus_scores = self.normalizeArray(np.array([c.reachable_bonus for c in self.top_candidates]))
        price_scores = self.normalizeArray(np.array([c.node.price for c in self.top_candidates]))
        exploitation_scores = self.normalizeArray(self.exploitation_score_price_bias*price_scores
            +(1-self.exploitation_score_price_bias)*reachable_bonus_scores)
    
        return exploitation_scores



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


def sampleFromWeighted(weight_arr: np.ndarray) -> int:
    sum_array = weight_arr.sum()
    weight_arr = weight_arr / sum_array
    index = np.random.choice(weight_arr.shape[0], p=weight_arr)
    return index
