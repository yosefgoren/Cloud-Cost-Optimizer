from audioop import mul
import re
from Experiment import *
from Distributions import NormDistInt

def create_tation_bias_experiment(N: int, C: int, T: int, Bias: float, force: bool = False)->Experiment:
    name = f"time-price_N-{N}_C-{C}_T-{T}_Bias-{Bias}"
    print(f"created experiment named:\n{name}\n")
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [Bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force = force,
        region = "us-east-2"
    )

def one_big_sample_experiment()->Experiment:
    N = 1
    name = f"one_big_sample_experiment"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [100]*N,
            "time_per_region":  [3600.0*5/20]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
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
        unique_sample_inputs = True
    )

def create_time_price_experiment(N: int, C: int, T: int)->Experiment:
    name = f"time-price_N-{N}_C-{C}_T-{T}"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
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
        unique_sample_inputs = True
    )

def create_candidate_size_experiement(N: int, C: int, T: int, Size: int)->Experiment:
    name = f"candidate-size_N-{N}_C-{C}_T-{T}_Size-{Size}"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [Size]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(3, 3, 1, 32),
            "ram": NormDistInt(4, 4 ,1, 128),
            "net": NormDistInt(1, 1, 1, 4)
        },
        region = "us-east-1"
    )

def create_trail_experiment()->Experiment:
    name = "trail"
    N=2
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [8]*N,
            "time_per_region":  [float(3)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
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
        region="us-east-1"
    )

import numpy as np
import matplotlib.pyplot as plt


def plot_multiple_curves(curves: list, z_axis: list, x_axis_name: str = "", y_axis_name: str = "", z_axis_name: str = ""):
    ax = plt.axes(projection='3d')
    for (xs, ys), z in zip(curves, z_axis):
        zs = np.ones_like(xs)*z
        ax.plot3D(xs, zs, ys)
    plt.show()

def plot_hyperparam_experiments(experiment_names: list, hyperparam_values: list, x_axis_name: str = "", y_axis_name: str = "", z_axis_name: str = ""):
    curves = []
    best_prices = []
    for name in experiment_names:
        e = Experiment.load(name)
        curves.append(e.get_plot_curves("INSERT_TIME", "BEST_PRICE", normalize=False)[0])
        best_prices.append(curves[-1][1][-1])
    # plot_multiple_curves(curves, hyperparam_values, z_axis_name)
    
    plt.plot(hyperparam_values, best_prices)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.show()


# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    
    sizes = [32*i for i in range(1, 11)]
    N = 100
    C = 18
    T = 20

    # for size in sizes:
    #     e = create_candidate_size_experiement(N, C, T, size)
    #     e.run(multiprocess=6, retry=2)

    candidate_experiment_naemes = [f"candidate-size_N-{N}_C-{18}_T-{T}_Size-{size}" for size in sizes]
    plot_hyperparam_experiments(candidate_experiment_naemes, sizes, "candidate list size", "price", "runtime")
    
    bias_values = [i*2/10.0 for i in range(5)]
    bias_experiment_names = [f"tation-bias_N-300_C-18_T-20_Bias-0.{i*2}" for i in range(5)]
    plot_hyperparam_experiments(bias_experiment_names, bias_values, "exploitation bias", "price", "runtime")
    
 