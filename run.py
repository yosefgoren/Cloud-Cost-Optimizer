from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import numpy as np
import matplotlib.pyplot as plt

def create_tation_bias_experiment(N: int, C: int, T: int, bias: float, force: bool = False)->Experiment:
    name = f"tation-bias_N-{N}_C-{C}_T-{T}_B-{bias}"
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
            "get_starting_node_mode":                   [GetStartNodeMode.RESET_SELECTOR]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [0.5]*N,
            "exploitation_bias":                [bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force = force,
        region = "us-east-2"
    )


def make_trail_exp()->Experiment:
    N = 1
    return Experiment.create(
        experiment_name="trail",
        control_parameter_lists = {
            "component_count":  [5]*N,
            "time_per_region":  [5]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.ALL]*N,
            "proportion_amount_node_sons_to_develop":   [1]*N,
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
        region = ["us-east-1", "us-west-1"],
        force=True
    )

def plot_multiple_curves_3d(curves: list, z_axis: list, x_axis_name: str = "", y_axis_name: str = "", z_axis_name: str = ""):
    ax = plt.axes(projection='3d')
    for (xs, ys), z in zip(curves, z_axis):
        zs = np.ones_like(xs)*z
        ax.plot3D(xs, zs, ys)
    plt.show()

def plot_hyperparam_experiments_3d(experiment_names: list, hyperparam_values: list, x_axis_name: str = "", y_axis_name: str = "", z_axis_name: str = ""):
    curves = []
    best_prices = []
    for name in experiment_names:
        e = Experiment.load(name)
        curves.append(e.get_plot_curves("INSERT_TIME", "BEST_PRICE", normalize=False)[0])
        best_prices.append(curves[-1][1][-1])
    plot_multiple_curves_3d(curves, hyperparam_values, z_axis_name)


def tation_bias_experiment_series():
    detail = 8
    biasies = [float(i)/(detail-1) for i in range(detail)]
    N = 80
    C = 9
    T = 3

    #create & run:
    # for bias in biasies:
    #     e = create_tation_bias_experiment(N, C, T, bias)
        # e.run(multiprocess=2, retry=2)

    #plot:
    candidate_experiment_naemes = [f"tation-bias_N-{N}_C-{C}_T-{T}_B-{bias}" for bias in biasies]
    plot_hyperparam_experiments_3d(candidate_experiment_naemes, biasies, "candidate list size", "price", "runtime")
    
def reset_exp_series():
    detail = 8
    biasies = [float(i)/(detail-1) for i in range(detail)]
    N = 80
    C = 9
    T = 3

    #plot:
    candidate_experiment_naemes = [f"tation-bias_N-{N}_C-{C}_T-{T}_B-{bias}" for bias in biasies]
    plot_hyperparam_experiments_3d(candidate_experiment_naemes, biasies, "candidate list size", "price", "runtime")
    
 
# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    # reset_exp_series()
    # e = make_trail_exp()
    # e.run(multiprocess=1)
    e = Experiment.load("time-price_N-300_C-18_T-20")
    e.plot(regions_aggregate=Flags.NON_INCREASING, normalize=False)