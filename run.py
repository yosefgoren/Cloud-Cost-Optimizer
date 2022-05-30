from audioop import mul
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


def plot_multiple_curves(curves: list, z_axis: list):
    ax = plt.axes(projection='3d')
    for (xs, ys), z in zip(curves, z_axis):
        xline = xs
        yline = ys
        zline = np.ones_like(xs)*z
        ax.plot3D(xline, zline, yline, 'gray')
    plt.show()

def plot_hyperparam_experiments(experiment_names: list, hyperparam_values: list):
    curves = []
    for name in experiment_names:
        e = Experiment.load(name)
        curves.append(e.get_plot_curves("INSERT_TIME", "BEST_PRICE", normalize=False)[0])
    plot_multiple_curves(curves, hyperparam_values)

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":

    plot_hyperparam_experiments()
    
 