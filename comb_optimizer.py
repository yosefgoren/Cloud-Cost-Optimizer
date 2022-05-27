# from msilib.schema import Component
# from importlib_metadata import Pair
import sqlite3

import numpy as np
from numpy import ndarray, ones_like
# from urllib3 import Retry
import time
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
    def __init__(self, candidate_list_size: int, price_calc, initial_seperated, time_per_region,
                 region, exploitation_score_price_bias, exploration_score_depth_bias,
                 exploitation_bias, output_path, verbose = True):
        self.verbose = verbose
        Node.node_cache = {}

        CombOptim.getComponentKey = KeyMannager(lambda componenet: componenet.component_name)
        """given a component, return a unique key associated with it, based on it's name."""

        CombOptim.getModuleKey = KeyMannager(
            lambda module: tuple(sorted([self.getComponentKey(component) for component in module])))
        """give a unique key to a module - a module is a set of components and is distinct from another
        module if they do not have the same sets of components."""

        CombOptim.getCombinationAsKey = KeyMannager(
            lambda combination: tuple(sorted([self.getModuleKey(module) for module in combination])))
        """given a comination - meaning a set (unordered) of modules, return a unique key associated with it."""

        CombOptim.getGroupSetAsKey = KeyMannager(
            lambda group_set: tuple(sorted([self.getCombinationAsKey(group[0]) for group in group_set])))
        """given a set of groups (the parameter is of type list, but the order is not considered
        to be a diffirentiating factor) return a unique key associated with the group set.
        Note how 'group[0]' is the one (and only) combination within the group."""

        """'price_calc' is a function: (Offer)-->float"""
        CombOptim.price_calc_func = price_calc
        self.root = CombOptim.calc_root(initial_seperated)
        self.optim_set = OptimumSet(1)
        self.reset_sel = ResetSelector(candidate_list_size,self.get_num_components(),self.root,exploitation_score_price_bias,  exploration_score_depth_bias,exploitation_bias,self.verbose)
        self.search_algo = SearchAlgorithm()
        self.start_time = time.time()
        self.time_per_region = time_per_region
        self.region = region
        self.exploitation_score_price_bias = exploitation_score_price_bias
        self.exploration_score_depth_bias = exploration_score_depth_bias
        self.exploitation_bias=exploitation_bias
        self.output_path=output_path
        self.conn = sqlite3.connect(output_path)

    def finish_stats_operation(self):
        self.conn.commit()
        self.conn.close()

    def insert_stats(self, iteration):
        returned_best = self.optim_set.returnBest()
        best_node = returned_best[0]
        best_price = best_node.getPrice()
        depth_best = best_node.getDepth()
        query = "INSERT INTO STATS (INSERT_TIME, NODES_COUNT, BEST_PRICE, DEPTH_BEST, ITERATION, REGION_SOLUTION)\
                          VALUES ({insert_time}, {NODES_COUNT}, {BEST_PRICE}, {DEPTH_BEST}, {ITERATION}, '{region}')".format(insert_time=time.time(), NODES_COUNT=len(Node.node_cache), BEST_PRICE=best_price, \
                                DEPTH_BEST=depth_best, ITERATION=iteration, region=self.region)
        if self.verbose:
            print(query)
        self.conn.execute(query)

    def create_stats_table(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS STATS
        (INSERT_TIME    REAL     NOT NULL,
        NODES_COUNT   INT     NOT NULL,
        BEST_PRICE  REAL    NOT NULL,
        DEPTH_BEST  INT NOT NULL,
        ITERATION  INT NOT NULL,
        REGION_SOLUTION TEXT    NOT NULL);''')

    @staticmethod
    def calc_root(initial_seperated):
        partitions = list(map(lambda i: separate_partitions(i), initial_seperated))
        return Node(partitions, 0)

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
        if self.root.getPrice() == np.inf:
            print("CombOptim.run: infinite price for root, returning empty result.")
            return []
        # print("comb optimizer starting run.")
        self.create_stats_table()
        i = 1
        while not self.isDone():
            start_node = self.reset_sel.getStartNode()
            # start_node = self.root #for debugging without reset sel...
            path = self.search_algo.run(start_node)
            if len(path) != 0:
                self.optim_set.update(path)
                self.reset_sel.update(path)

            self.insert_stats(i)
            i += 1

        self.finish_stats_operation()

        return [node.getOffer() for node in self.optim_set.returnBest()]

    def isDone(self)->bool:
        return time.time()-self.start_time > self.time_per_region

class Node:
    node_cache = {}

    def __init__(self, partitions, node_depth: int):
        self.node_depth = node_depth
        self.partitions = copy.deepcopy(partitions)
        self.offer = self.__calc_offer()
        if self.offer is not None:
            self.price = self.offer.total_price
        else:
            self.price = np.inf

        self.sons = None
        Node.node_cache[self.hashCode()] = self
        #print(f"Node.__init__;\
        # hash: {self.hashCode()}\
        # , depth: {self.getDepth()}\
        # , total_score: {self.price}")

    def __calc_offer(self):
        modules = []
        for group in self.partitions:
            combination = group[0]  # each group has 1 combination
            for module in combination:
                modules.append(module)

        return CombOptim.price_calc_func(Offer(modules, None))

    def getDepth(self):
        return self.node_depth

    def getPrice(self):
        return self.price

    def getOffer(self):
        return self.offer

    def hashCode(self)->int:
        return CombOptim.getGroupSetAsKey(self.partitions)

    @staticmethod
    def hashCodeOfPartition(partition)->int:
        return CombOptim.getGroupSetAsKey(partition)
      
      
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

                            if Node.hashCodeOfPartition(new_partition) in Node.node_cache:
                                self.sons.append(Node.node_cache[Node.hashCodeOfPartition(new_partition)])
                            else:
                                self.sons.append(Node(new_partition, self.getDepth() + 1))

class OptimumSet:
    def __init__(self, k: int):
        """the table holds the best k seen so far in terms of price.
            requires that the elements inserted will have the method 'getPrice' which should
            return a float."""
        self.k = k
        self.table = [] # contain hashcode
    
    def update(self, visited_nodes: list):
        """considers the list of new nodes, such that the resulting set of nodes will be the 'k' best nodes
            seen at any update. The ordering the nodes is given by their 'getPrice()' method."""
        candidates = self.table + [node.hashCode() for node in visited_nodes if (not (node.hashCode() in self.table))]
        candidates.sort(key=lambda hashcode: Node.node_cache[hashcode].getPrice())
        self.table = candidates[:self.k]

    def returnBest(self):
        """returns the 'k' nodes with the best price seen so far.
        If not seen 'k' nodes yet, returns a list shorter than 'k'."""
        return [Node.node_cache[hashcode] for hashcode in self.table]


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
            self.total_score = None
            self.subtree_price_penalty = -self.node.getPrice()
            self.hash = None

    def __init__(self, candidate_list_size: int, num_componants: int, root: Node , exploitation_score_price_bias , exploration_score_depth_bias,exploitation_bias, verbose=True):
        """ The reset-selector remembers a list of the best candidates (candidate nodes) seen so far,
            list is saved at: self.top_candidates.
            The parameter 'k' is the maximum allowed size for the candidate list."""
        self.verbose = verbose
        self.top_candidates = [ResetSelector.Candidate(root)]
        self.candidate_list_size = int(candidate_list_size)
        self.num_componants = num_componants

        #reachable_bonus_formula_base is calculated here so we only have to calculate it once.
        self.penalty_base = 10**(1.0/num_componants)
        
        #hyperparameters:
        self.exploitation_score_price_bias = exploitation_score_price_bias
        self.exploration_score_depth_bias = exploration_score_depth_bias
        self.exploitation_bias=exploitation_bias

        self.updateTotalScores()

    def getStartNode(self)->Node:
        """this method represents the main functionality of the reset-selector: based on all data seen so far
            - the reset-selector will return the the node it thinks the next run should start from."""
        scores_list = [candidate.total_score for candidate in self.top_candidates]
        scores_arr = np.array(scores_list)
        try:
            selected_node_idx = sampleFromWeighted(scores_arr)
        except:
            print("sample from weighted raised err, scores list: ", scores_list)
            exit(1)
        selected_candidate = self.top_candidates[selected_node_idx]
        if self.verbose:
            print(f"ResetSelector.getStartNode;\
                hash: {selected_candidate.node.hashCode()}\
                , depth: {selected_candidate.node.getDepth()}\
                , total_score: {selected_candidate.total_score}")
        return selected_candidate.node

    def update(self, path: list):
        """'path' is a list of nodes seen in the last run the serach algorithm.
            this method will update in state of the reset selector - to consider the nodes seen in last search run.

            The order of nodes in 'path' is exprected to be the same order as the nodes were seen in the search.
            
            Calling this method will also cause the reset-selector to re-calculate the total scores for each 
            of the candidates saved within it."""
        #consider all nodes seen in last path as candidates:
        candidate_dict = {candidate.node.hashCode():candidate for candidate in self.top_candidates}
        last_candidate = None
        for node in reversed(path):
            #add the new node to set of candidates if it's not already there:
            node_hash = node.hashCode()
            if not node_hash in candidate_dict:
                candidate_dict[node_hash] = ResetSelector.Candidate(node)
            candidate = candidate_dict[node_hash]
            
            #update the subtree penalty of the candidate base on path:
            if last_candidate != None:
                candidate.subtree_price_penalty = \
                    max(candidate.subtree_price_penalty, last_candidate.subtree_price_penalty*self.penalty_base)
            last_candidate = candidate
        
        #update the list of top candidates and re-calculate total scores for all candidates currently saved:
        self.top_candidates = [item for item in candidate_dict.values()]
        self.updateTotalScores()
        # if 0 in [c.total_score for c in self.top_candidates]:
        #     raise Exception("ResetSelector.update: error: a candidates has a total score of 0.")
        
        #sort the list of top candidates and throw away the candidates that are not in the top k:
        self.top_candidates.sort(key=lambda candidate: candidate.total_score)
        self.top_candidates = self.top_candidates[:self.candidate_list_size]
        
    def updateTotalScores(self)->list:
        """updates the total scores (floats) of all candidates in 'self.top_candidates'."""
        ration_scores = self.calcRationScores()
        tation_scores = self.calcTationScores()
        tation_bias = self.getCurrentTationBias()

        total_scores = tation_bias*tation_scores + (1-tation_bias)*ration_scores
        for idx in range(len(self.top_candidates)):
            self.top_candidates[idx].total_score = total_scores[idx]
        if np.all(total_scores < 1e-6):
            total_scores = np.ones_like(total_scores, dtype=np.float)

    def getCurrentTationBias(self)->float:
        """get the current exploitation bias, this represents the current preference of the algorithm for exploitation
            over exploration."""
        return self.exploitation_bias
        #TODO: we probably want an implementation based on how much time the algorithm has to run,
        #	 s.t. when there is little time left the exploitation bias is close to 1.

    @staticmethod
    def normalizeArray(arr: ndarray)->ndarray:
        #minmax normalization:
        diff = arr.max() - arr.min()
        if diff == 0:
            return ones_like(arr)/2
        else:
            return (arr-arr.min())/diff
        
        # return arr/np.sum(arr) #normalise according to L1
        #return arr/np.linalg.norm(arr)#normalize according to L2

    def calcRationScores(self)->list:
        """calculates the exploration scores of all candidates in 'self.top_candidates' and returns scores
            in list of floats in same order."""
        uniqueness_scores = self.calcUniquenessScores()
        depth_scores = self.calcDepthScores()
        exploration_scores = ResetSelector.normalizeArray(self.exploration_score_depth_bias*depth_scores
            +(1-self.exploration_score_depth_bias)*uniqueness_scores)

        return exploration_scores

    def calcDepthScores(self)->ndarray:
        """Calculate the 'depth score' for each candidate in 'self.top_candidates'.
            The deeper the candidate's node - the higher the depth score."""
        depths = np.array([c.node.getDepth() for c in self.top_candidates])
        return ResetSelector.normalizeArray((depths-self.num_componants)*(depths-self.num_componants))

    def calcUniquenessScores(self)->ndarray:
        """Calculate the 'uniqueness score' for each candidate in 'self.top_candidates'.
            This score will be highest for nodes that are very different from the other nodes in 'top_candidates'."""
        nodes_list = [c.node for c in self.top_candidates]
        distances = ResetSelector.combinationDistancesFormula(nodes_list)
        return ResetSelector.normalizeArray(distances)

    @staticmethod
    def combinationDistancesFormula(node_list: list)->ndarray:
        """Implementation of formula for calculating 'distance' for all nodes to all other nodes.
            The input is a list of combinations, and the output is an array of floats where the i'th float
            represents the average 'distance' of i'th node from the rest of the nodes in the input list.
            
            Input is in the form of list<Node>."""
        #TODO
        return np.ones(len(node_list, ), dtype=float)

    def calcTationScores(self)->ndarray:
        """calculates the explotation scores of all candidates in 'self.top_candidates' and returns scores
            in an array of floats with a corresponding order."""
        reachable_bonus_scores = ResetSelector.normalizeArray(
            np.array([c.subtree_price_penalty for c in self.top_candidates])
        )
        price_scores = ResetSelector.normalizeArray(np.array([-c.node.price for c in self.top_candidates]))
        exploitation_scores = ResetSelector.normalizeArray(self.exploitation_score_price_bias*price_scores
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
            if next_node.getPrice() == np.inf:
                return path
            path.append(next_node)
            next_node = self.get_next(next_node)
            self.update_temperature()
            if next_node is None:
                return path

    def get_next(self, node: Node) -> Node:
        """get the chosen son to continue to in the next iteration"""
        node.calcAllSons()
        flag = self.is_choosing_downgrades()
        improves, downgrades = SearchAlgorithm.split_sons_to_improves_and_downgrades(node.sons, node.getPrice())
        if (downgrades.shape[0] != 0) and flag:
            return SearchAlgorithm.get_son_by_weights(downgrades)
        elif improves.shape[0] != 0:
            return SearchAlgorithm.get_son_by_weights(improves)
        else:
            return None

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
            price_diff = cur_node_price - son.getPrice()
            if price_diff > 0:
                improves.append([son, price_diff])
            else:
                downgrades.append([son, price_diff])

        return np.array(improves), np.array(downgrades)

    def is_choosing_downgrades(self):
        """return if we will choose a downgrade son"""
        prob_for_downgrade = 0.1 - 1.0 / (10 * np.exp(1.0 / (np.power(self.temperature, 0.9))))
        return (np.random.choice([0, 1], p=[1 - prob_for_downgrade, prob_for_downgrade])) == 1

    def update_temperature(self):
        """change the temperature according to the self.temperature_increment_pace"""
        self.temperature += self.temperature_increment_pace


def sampleFromWeighted(weight_arr: np.ndarray) -> int:
    if np.NaN in weight_arr:
        print(weight_arr)
        raise Exception("sampleFromWeighted: error: weight_arr contains NaN")

    sum_array = weight_arr.sum()
    if sum_array == 0:
        raise Exception("sampleFromWeighter: error: sum of weights is 0")
    weight_arr1 = weight_arr / sum_array
    index = np.random.choice(weight_arr1.shape[0], p=weight_arr1)

    return index
