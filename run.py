from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import numpy as np
import matplotlib.pyplot as plt

def make_trail_exp(name: str = "trail")->Experiment:
    N = 1
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [20]*N,
            "time_per_region":  [5]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.PROPORTIONAL]*N,
            "proportion_amount_node_sons_to_develop":   [0.1]*N,
            "get_next_mode":                            [GetNextMode.STOCHASTIC_ANNEALING]*N,
            "get_starting_node_mode":                   [GetStartNodeMode.RESET_SELECTOR]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(3, 3, 1, 32),
            "ram": NormDistInt(4, 4 ,1, 128),
            "net": NormDistInt(1, 1, 1, 4)
        },
        region = ["us-east-1"],
        force=True
    )

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    np.random.seed(42)
    e = make_trail_exp()
    e.run(force=True, bruteforce=True)
    # e = Experiment.load("trail")
    e.plot(normalize=False)