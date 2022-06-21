import multiprocessing
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
            "component_count":  [8]*N,
            "time_per_region":  [3]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.PROPORTIONAL]*N,
            "proportion_amount_node_sons_to_develop":   [0.1]*N,
            "get_next_mode":                            [GetNextMode.STOCHASTIC_ANNEALING]*N,
            "get_starting_node_mode":                   [GetStartNodeMode.RANDOM]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [1<<4]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [0]*N,
            "exploitation_bias":                [0]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(3, 3, 1, 32),
            "ram": NormDistInt(4, 4 ,1, 128),
            "net": NormDistInt(1, 1, 1, 4)
        },
        region = ["us-east-1"],
        force=True,
        use_existing_inputs="../experiments/trail"
    )

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    np.random.seed(42)
    e = make_trail_exp("t2")
    # e = Experiment.load("t2")
    e.run(bruteforce=True, multiprocess=1)

    #plot each sample by itself:
    # e = Experiment.load("trail")
    # cvs = e.get_plot_curves(normalize=False)
    # for i, xys in enumerate(cvs):
    #     xs, ys = invert_list(xys)
    #     plt.plot(xs, ys, label=str(i))
    # plt.legend()
    # plt.show()

    e.plot(y_variable="BEST_PRICE", normalize=False)
    # e.run(multiprocess=3, bruteforce=True)
    # e.plot(y_variable="BEST_PRICE", normalize=False)