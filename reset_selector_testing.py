from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import numpy as np
import matplotlib.pyplot as plt


def RSE_Generic(N: int, C: int, T: int, GSNM: GetStartNodeMode, tation_bias: float, name: str)->Experiment:
    print(f"created experiment named:\n{name}\n")
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.ALL]*N,
            "proportion_amount_node_sons_to_develop":   [1]*N,
            "get_next_mode":                            [GetNextMode.STOCHASTIC_ANNEALING]*N,
            "get_starting_node_mode":                   [GSNM]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [0.5]*N,
            "exploitation_bias":                [tation_bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = "us-east-1"
    )

def RSE_root(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.ROOT, 0, "RSE_root_restart")

def RSE_random(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RANDOM, 0, "RSE_random_restart")

def RSE_selector1_0(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 1.0, "RSE_reset_sel_1_0")

def RSE_selector0_3(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.3, "RSE_reset_sel_0_3")

def RSE_selector0_7(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.7, "RSE_reset_sel_0_7")

def RSE_selector0_0(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.0, "RSE_reset_sel_0_0")

RSE_names = [
    "RSE_root_restart",
    "RSE_random_restart",
    "RSE_reset_sel_1_0",
    "RSE_reset_sel_0_3",
    "RSE_reset_sel_0_7",
    "RSE_reset_sel_0_0"
]


def run_RSEs():
    N = 200
    C = 10
    T = 10

    e = RSE_root(N, C, T)
    e.run(multiprocess=2, retry=2)

    e = RSE_random(N, C, T)
    e.run(multiprocess=2, retry=2)

    e = RSE_selector1_0(N, C, T)
    e.run(multiprocess=2, retry=2)

    e = RSE_selector0_3(N, C, T)
    e.run(multiprocess=2, retry=2)

    e = RSE_selector0_7(N, C, T)
    e.run(multiprocess=2, retry=2)

    e = RSE_selector0_0(N, C, T)
    e.run(multiprocess=2, retry=2)

if __name__ == "__main__":
    #run:
    # run_RSEs()
    #plot:
    for name in RSE_names:
        e = Experiment.load(name)
        times, prices = e.get_plot_curves(normalize=False)[0]
        plt.plot(times, prices, label=name)
    plt.legend()
    plt.show()