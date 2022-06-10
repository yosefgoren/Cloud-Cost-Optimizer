from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import numpy as np
import matplotlib.pyplot as plt


def NNE_Generic(N: int, C: int, T: int, root_dir: str, NNM: GetNextMode, name: str)->Experiment:
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
            "get_next_mode":                            [NNM]*N,
            "get_starting_node_mode":                   [GetStartNodeMode.RESET_SELECTOR]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [0.5]*N,
            "exploitation_bias":                [0.7]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = "us-east-1",
        experiments_root_dir=root_dir
    )

def NNE_Greedy(*args)->Experiment:
    return NNE_Generic(*args, GetNextMode.GREEDY, "NNE_greedy")
def NNE_Stochastic(*args)->Experiment:
    return NNE_Generic(*args, GetNextMode.STOCHASTIC_ANNEALING, "NNE_stochastic")


NNEs = {
    "NNE_greedy" : NNE_Greedy,
    "NNE_stochastic" : NNE_Stochastic,
}

def run_NNEs(N: int, root_dir: str):
    C = 10
    T = 10
    mp = 4

    for NNE_gen in NNEs.values():
        e = NNE_gen(N, C, T, root_dir)
        e.run(multiprocess=mp, retry=2)

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    series_name = "NNE-N600-C20-T40"
    root_dir = "./experiments/"+series_name
    N = 4
    #run:
    # run_NNEs(N, root_dir)
    #plot:
    # get_cross_section_plots(N, N//2, series_name)

    for name in NNEs.keys():
        e = Experiment.load(name, experiments_root_dir=root_dir)
        times, prices = e.get_plot_curves(normalize=False, x_variable="INSERT_TIME", y_variable="BEST_PRICE")[0]
        plt.plot(times, prices, label=name)
    

    plt.title(f"NNE; all samples")
    plt.xlabel("time")
    plt.ylabel("best price")
    plt.legend()
    plt.show()